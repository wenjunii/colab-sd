"""SDXL conditioning and launch contracts; no model downloads or CUDA installs."""
import io
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from PIL import Image

import build_clean_toe as builder
import colab_launcher
import colab_server as server
import sdxl_stream as xl
import tensorrt_backend as trt
import workstation
from test_streaming import Par
from test_streaming import FakeWebSocket, until


class XLChecks(unittest.TestCase):
    def test_live_pooled_text_and_size_conditioning_follow_batch_and_compiled_replacement(self):
        native, compiled = Mock(return_value=('native',)), Mock(return_value=('compiled',))
        unet = xl.XLConditionedUNet(native, 768, 576)
        with self.assertRaisesRegex(RuntimeError, 'Prepare'):
            unet(torch.zeros(1, 4, 2, 2), torch.ones(1), torch.zeros(1, 77, 8))
        unet.set_conditioning(torch.ones(1, 4))
        replacement = unet.with_model(compiled)
        unet.set_conditioning(torch.full((1, 4), 3.0))
        sample = torch.zeros(2, 4, 2, 2)
        self.assertEqual(replacement(sample, torch.ones(2), torch.zeros(2, 77, 8)), ('compiled',))
        kwargs = compiled.call_args.kwargs['added_cond_kwargs']
        self.assertTrue(torch.all(kwargs['text_embeds'] == 3))
        self.assertEqual(kwargs['text_embeds'].shape, (2, 4))
        self.assertEqual(kwargs['time_ids'].tolist(), [[576, 768, 0, 0, 576, 768]] * 2)

    def test_proxy_captures_second_encoder_pool_for_prepare_and_update_prompt(self):
        encoded = (torch.ones(1, 77, 8), None, torch.full((1, 4), 2.0), None)
        pipe = SimpleNamespace(unet=Mock(), encode_prompt=Mock(return_value=encoded))
        proxy = xl.XLStreamPipe(pipe, 512, 512)
        self.assertIs(proxy.encode_prompt(prompt='forest', do_classifier_free_guidance=False), encoded)
        self.assertIs(proxy.unet.state.pooled, encoded[2])
        with self.assertRaises(ValueError):
            proxy.encode_prompt(prompt='forest', do_classifier_free_guidance=True)

    def test_loader_pins_xl_weights_and_uses_only_matching_tiny_vae(self):
        pipe = SimpleNamespace(unet=Mock(), vae=None, to=Mock(), set_progress_bar_config=Mock())
        tiny = object()
        diffusion = SimpleNamespace(StableDiffusionXLPipeline=SimpleNamespace(from_pretrained=Mock(return_value=pipe)),
                                   AutoencoderTiny=SimpleNamespace(from_pretrained=Mock(return_value=tiny)))
        core = SimpleNamespace(StreamDiffusion=Mock())
        with patch.dict(sys.modules, {'diffusers': diffusion, 'streamdiffusion': core}), \
                patch.object(torch.cuda, 'is_available', return_value=False):
            xl.SDXLStreamWrapper.load(512, 384)
        model_call = diffusion.StableDiffusionXLPipeline.from_pretrained.call_args
        self.assertEqual((model_call.args[0], model_call.kwargs['revision']), (xl.MODEL_ID, xl.MODEL_REVISION))
        self.assertTrue(model_call.kwargs['use_safetensors'])
        self.assertEqual(model_call.kwargs['variant'], 'fp16')
        vae_call = diffusion.AutoencoderTiny.from_pretrained.call_args
        self.assertEqual((vae_call.args[0], vae_call.kwargs['revision']), (xl.VAE_ID, xl.VAE_REVISION))
        self.assertIs(pipe.vae, tiny)
        settings = core.StreamDiffusion.call_args.kwargs
        self.assertIsInstance(settings['pipe'], xl.XLStreamPipe)
        self.assertEqual(settings['t_index_list'], [0])
        self.assertEqual(settings['cfg_type'], 'none')

    def test_prompt_blending_caches_and_restores_both_embedding_types(self):
        encode = Mock(side_effect=lambda **kw: (torch.full((1, 77, 8), 1. if kw['prompt'] == 'A' else 3.),
                                                None, torch.full((1, 4), 2. if kw['prompt'] == 'A' else 6.), None))
        pipe = SimpleNamespace(encode_prompt=encode)
        text = SimpleNamespace(stream=SimpleNamespace(pipe=pipe, batch_size=1, prompt_embeds=None,
                               unet=xl.XLConditionedUNet(Mock(), 512, 512)))
        image = SimpleNamespace(stream=SimpleNamespace(batch_size=2, prompt_embeds=None,
                                unet=xl.XLConditionedUNet(Mock(), 512, 512)))
        image.update_strength = lambda strength: image.stream.unet.set_conditioning(torch.zeros(1, 4))
        with patch.multiple(server, diffusion_model='sdxl-turbo', pipeline_ready=True, stream_txt2img=text,
                            stream_img2img=image, applied_prompt_state=None, prompt_embedding_cache={},
                            current_prompt_a='A', current_prompt_b='B', current_prompt_blend=0.0,
                            current_neg_prompt='', current_seed=42, current_strength=.6), patch.object(server, 'set_seed'):
            server.apply_prompts_and_blend('A', 'B', .5, '')
            server.apply_controls({'strength': .8, 'seed': 7})
            self.assertEqual(encode.call_count, 2)
            self.assertTrue(torch.all(text.stream.prompt_embeds == 2))
            self.assertTrue(torch.all(image.stream.prompt_embeds == 2))
            self.assertTrue(torch.all(text.stream.unet.state.pooled == 4))
            self.assertTrue(torch.all(image.stream.unet.state.pooled == 4))
            self.assertEqual(image.stream.prompt_embeds.shape[0], 2)

    def test_failed_xl_shared_core_never_substitutes_sd_turbo(self):
        text = SimpleNamespace(stream=SimpleNamespace(pipe=SimpleNamespace(dtype=torch.float32)))
        with patch.multiple(server, diffusion_model='sdxl-turbo', pipeline_ready=False, applied_prompt_state=None,
                            stream_txt2img=None, stream_img2img=None, prompt_embedding_cache={}), \
                patch.object(server, 'create_txt2img_pipeline', return_value=text), \
                patch.dict(sys.modules, {'streamdiffusion': SimpleNamespace(StreamDiffusion=Mock(side_effect=ValueError('failure')))}), \
                patch.object(server, 'StreamDiffusionWrapper') as old_wrapper:
            with self.assertRaisesRegex(RuntimeError, 'no other model'):
                server.load_pipeline()
            old_wrapper.assert_not_called()
            self.assertFalse(server.pipeline_ready)

    def test_launcher_checks_model_identity_and_applies_selection_to_child(self):
        session = colab_launcher.BridgeSession(diffusion_model='sdxl-turbo', tunnel=False)
        with patch.object(colab_launcher.socket, 'create_connection', side_effect=ConnectionRefusedError), \
                patch.object(session, 'wait_for_ready'), patch.object(session, '_spawn') as spawn:
            session.start()
            self.assertEqual(spawn.call_args.args[2]['BRIDGE_DIFFUSION_MODEL'], 'sdxl-turbo')
        session.processes['server'] = SimpleNamespace(poll=lambda: None)
        with patch.object(colab_launcher.urllib.request, 'urlopen', return_value=io.StringIO(json.dumps(
                {'ready': True, 'session_id': session.session_id, 'diffusion_model': 'sd-turbo'}))):
            with self.assertRaisesRegex(RuntimeError, 'different model'):
                session.wait_for_ready()
        with self.assertRaises(ValueError):
            colab_launcher.BridgeSession(backend='waypoint', diffusion_model='sdxl-turbo')

    def test_fresh_server_reports_selected_model_before_gpu_load(self):
        result = subprocess.run([sys.executable, '-B', '-c', 'import colab_server as s; print(s.health().body.decode())'],
                  cwd=Path(server.__file__).parent, capture_output=True, text=True, timeout=30,
                  env={**os.environ, 'BRIDGE_BACKEND': 'sd-turbo', 'BRIDGE_DIFFUSION_MODEL': 'sdxl-turbo'})
        self.assertEqual(result.returncode, 0, result.stderr)
        status = json.loads(result.stdout)
        self.assertEqual(status['model'], 'SDXL-Turbo')
        self.assertEqual(status['diffusion_model'], 'sdxl-turbo')
        self.assertFalse(status['ready'])

    def test_td_displays_model_name_from_server(self):
        pars = SimpleNamespace(Modelname=Par('--'), Status=Par(''))
        storage = {}
        component = SimpleNamespace(par=pars, fetch=lambda key, default=None: storage.get(key, default),
                                    store=lambda key, value: storage.__setitem__(key, value))
        namespace = {'parent': lambda: component, 'op': lambda name: None}
        exec(builder.WEBSOCKET_CODE, namespace)
        namespace['onReceiveText'](None, 0, json.dumps({'type': 'stats', 'model': 'SDXL-Turbo'}))
        self.assertEqual(pars.Modelname.val, 'SDXL-Turbo')

    def test_workstation_model_option_reaches_existing_streamdiffusion_environment(self):
        with tempfile.TemporaryDirectory(prefix='colab-sd-test-xl-') as directory:
            runtime = Path(directory).resolve()
            python = workstation.runtime_python(runtime)
            python.parent.mkdir(parents=True)
            python.touch()
            wrapper = runtime / 'StreamDiffusion/utils/wrapper.py'
            wrapper.parent.mkdir(parents=True)
            wrapper.touch()
            (runtime / 'workstation.json').write_text(json.dumps({'managed_by': 'colab-sd', 'ready': True,
                                                                 'revision': workstation.UPSTREAM_REVISION}))
            with patch.object(workstation, 'gpu_check'), patch.object(workstation, 'BridgeSession') as launch, patch('builtins.print'):
                self.assertEqual(workstation.main(['start', '--runtime', str(runtime), '--model', 'sdxl-turbo']), 0)
                self.assertEqual(launch.call_args.kwargs['diffusion_model'], 'sdxl-turbo')
                self.assertEqual(launch.call_args.kwargs['root'], runtime / 'StreamDiffusion')


class XLCompilerChecks(unittest.TestCase):
    def test_compiled_xl_graph_keeps_pooled_text_and_size_as_live_inputs(self):
        class Native(torch.nn.Module):
            config = SimpleNamespace(in_channels=4, cross_attention_dim=8, addition_embed_type='text_time',
                                     addition_time_embed_dim=2, projection_class_embeddings_input_dim=16)
            device, dtype = torch.device('cpu'), torch.float32
            attn_processors = {}
            def set_attn_processor(self, processors):
                pass
            def forward(self, sample, timestep, encoder_hidden_states=None, added_cond_kwargs=None, **kwargs):
                pool, size = added_cond_kwargs['text_embeds'], added_cond_kwargs['time_ids']
                return (sample * .5 + pool.mean() + size.mean() / 10000,)
        class TorchTensorRTModule(torch.nn.Module):
            def __init__(self, graph):
                super().__init__()
                self.graph = graph
            def forward(self, *args):
                return self.graph(*args)
        compiler = Mock(side_effect=lambda graph, **kw: TorchTensorRTModule(graph))
        with tempfile.TemporaryDirectory(prefix='colab-sd-test-xl-trt-') as directory, \
                patch.dict(sys.modules, {'torch_tensorrt': SimpleNamespace(__version__='2.11', compile=compiler),
                    'tensorrt': SimpleNamespace(__version__='10.15'),
                    'diffusers.models.attention_processor': SimpleNamespace(AttnProcessor=Mock())}), \
                patch.object(trt, 'package_requirement'), patch.object(trt, 'save_engine_archive') as save, \
                patch.object(trt, 'load_engine_archive', return_value=None), \
                patch.object(torch.cuda, 'is_available', return_value=True), patch.object(torch.cuda, 'synchronize'), \
                patch.object(torch.cuda, 'get_device_capability', return_value=(8, 0)), \
                patch.object(torch.cuda, 'get_device_name', return_value='CPU fixture'), patch('builtins.print'):
            adapter = trt.compile_unet(Native(), 32, 16, 2, directory)
            inputs = compiler.call_args.kwargs['inputs']
            self.assertEqual(len(inputs), 5)
            self.assertEqual(inputs[3].shape, (2, 4))
            self.assertEqual(inputs[4].tolist(), [[16, 32, 0, 0, 16, 32]] * 2)
            first = adapter(*inputs[:3], added_cond_kwargs={'text_embeds': inputs[3], 'time_ids': inputs[4]})[0]
            second = adapter(*inputs[:3], added_cond_kwargs={'text_embeds': inputs[3] + 1, 'time_ids': inputs[4]})[0]
            self.assertTrue(torch.allclose(second - first, torch.ones_like(first)))
            self.assertEqual(len(save.call_args.args[3]), 5)
            self.assertEqual(len(trt.alternate_inputs(inputs)), 5)
            with self.assertRaisesRegex(ValueError, 'conditioning'):
                adapter(*inputs[:3])


class CancelledControllerChecks(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_cannot_abandon_controller_while_worker_finishes(self):
        started, release = threading.Event(), threading.Event()
        def block():
            started.set()
            if not release.wait(5):
                raise RuntimeError('fixture worker timed out')
        busy = server.model_executor.submit(block)
        backend = SimpleNamespace(release_controls=Mock())
        ws = FakeWebSocket()
        with patch.multiple(server, access_token='', active_client=None, is_world_backend=True,
                            is_krea_backend=False, world_backend=backend):
            task = asyncio.create_task(server.websocket_stream_endpoint(ws))
            try:
                await until(lambda: started.is_set() and server.active_client is ws)
                task.cancel()
                await asyncio.sleep(.02)  # first cancellation enters cleanup
                self.assertIs(server.active_client, ws)
                task.cancel()  # ASGI scopes can cancel again during cleanup
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertIs(server.active_client, ws)
                release.set()
                await until(lambda: server.active_client is None)
                backend.release_controls.assert_called()
            finally:
                release.set()
                await asyncio.wrap_future(busy)
