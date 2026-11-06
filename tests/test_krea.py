"""Krea lifecycle/transport contracts; no model weights or CUDA installation."""
import ast
import asyncio
from contextlib import nullcontext
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import build_clean_toe as builder
import colab_launcher
import colab_server as server
import generate_krea_notebook as notebook
import krea_assets as assets
from krea_backend import KreaBackend, gpu_preflight, validate_krea_controls
import krea_setup
from test_streaming import FakeWebSocket, until
from test_waypoint import Par


def fake_backend(clip_blocks=64):
    def generate(state, **kwargs):
        count = 9 if kwargs['block_idx'] == 0 else 12
        state.values.update(videos=[np.zeros((count, 480, 832, 3), dtype=np.uint8)],
                            prompt_embeds='cached-embedding')
        # Reproduce the upstream decoder's mutation, so reset must restore it.
        pipe.vae.clear_cache = lambda: None
        return state
    pipe = Mock(side_effect=generate, vae=SimpleNamespace(clear_cache=Mock()))
    torch = SimpleNamespace(Generator=Mock(return_value=Mock()), inference_mode=nullcontext)
    return KreaBackend(pipe, lambda: SimpleNamespace(values={}), torch, clip_blocks)


class KreaChecks(unittest.TestCase):
    def test_invalid_patch_is_atomic_and_only_fixed_size_is_accepted(self):
        backend = fake_backend()
        for data in ({'prompt_a': 'new', 'krea_paused': 1}, {'width': 1024, 'height': 768},
                     {'mode': 'world'}, {'seed': True}, {'seed': -1}, {'prompt_a': ' '},
                     {'prompt_a': 'x' * 4097}, {'krea_reset': 'yes'}):
            with self.subTest(data=str(data)[:60]), self.assertRaises(ValueError):
                backend.configure(data)
            self.assertEqual(backend.prompt, '')
            self.assertFalse(backend.can_generate)
        self.assertEqual(validate_krea_controls({'prompt_a': ' a city '}), {'prompt_a': 'a city'})

    def test_hardware_preflight_rejects_small_or_busy_gpu_before_downloads(self):
        cuda = SimpleNamespace(is_available=lambda: True, get_device_capability=lambda: (8, 0),
                               mem_get_info=lambda: (35 * 1024**3, 40 * 1024**3),
                               get_device_name=lambda: 'A100')
        torch = SimpleNamespace(cuda=cuda)
        self.assertEqual(gpu_preflight(torch)['gpu'], 'A100')
        for capability, free, total in ((7, 39, 40), (8, 22, 24), (8, 20, 40)):
            cuda.get_device_capability = lambda: (capability, 0)
            cuda.mem_get_info = lambda: (free * 1024**3, total * 1024**3)
            with self.assertRaises(RuntimeError):
                gpu_preflight(torch)

    def test_generation_waits_for_prompt_and_holds_only_one_chunk(self):
        backend = fake_backend()
        self.assertIsNone(backend.next_frame())
        backend.pipe.assert_not_called()
        backend.configure({'prompt_a': 'a city'})
        for remaining in range(8, -1, -1):
            frame, duration = backend.next_frame()
            self.assertEqual(frame.shape, (480, 832, 3))
            self.assertEqual(len(backend.frames), remaining)
        self.assertEqual(backend.pipe.call_count, 1)
        self.assertIsNone(backend.state.values['videos'])
        backend.next_frame()
        self.assertEqual(len(backend.frames), 11)
        kwargs = backend.pipe.call_args.kwargs
        self.assertEqual((kwargs['num_inference_steps'], kwargs['width'], kwargs['height']), (4, 832, 480))
        self.assertIsNone(kwargs['prompt'])
        self.assertEqual(kwargs['prompt_embeds'], 'cached-embedding')
        self.assertFalse(kwargs['update_prompt_embeds'])

    def test_prompt_changes_invalidate_embedding_at_next_chunk(self):
        backend = fake_backend()
        backend.configure({'prompt_a': 'city'})
        backend.next_frame()
        backend.configure({'prompt_a': 'forest'})
        for _ in range(8):
            backend.next_frame()
        self.assertEqual(backend.pipe.call_count, 1)
        backend.next_frame()
        self.assertEqual(backend.pipe.call_args.kwargs['prompt'], ['forest'])
        self.assertIsNone(backend.pipe.call_args.kwargs['prompt_embeds'])
        self.assertTrue(backend.pipe.call_args.kwargs['update_prompt_embeds'])

    def test_pause_reset_and_rollover_reuse_weights_and_restore_vae_cache(self):
        backend = fake_backend(clip_blocks=1)
        original_clear = backend._clear_vae_cache
        backend.configure({'prompt_a': 'city', 'seed': 7})
        backend.next_frame()
        backend.configure({'krea_paused': True})
        self.assertIsNone(backend.next_frame())
        self.assertEqual(backend.progress()['world_phase'], 'paused')
        backend.configure({'krea_paused': False})
        for _ in range(9):
            backend.next_frame()
        self.assertEqual((backend.clip_index, backend.block_index), (1, 1))
        self.assertEqual([call.args[0] for call in backend.torch.Generator.return_value.manual_seed.call_args_list], [7, 8])
        self.assertEqual(backend.pipe.call_args.kwargs['block_idx'], 0)
        self.assertGreaterEqual(original_clear.call_count, 2)
        backend.configure({'krea_reset': True})
        self.assertEqual((backend.clip_index, backend.block_index, len(backend.frames)), (0, 0, 0))
        self.assertIs(backend.pipe.vae.clear_cache, original_clear)
        backend.next_frame()
        backend.release_controls()
        self.assertFalse(backend.can_generate)
        self.assertEqual(len(backend.frames), 0)

    def test_bad_chunk_stops_retries_until_explicit_reset(self):
        backend = fake_backend()
        backend.configure({'prompt_a': 'city'})
        backend.pipe.side_effect = ValueError('GPU failed')
        with self.assertRaisesRegex(ValueError, 'reset'):
            backend.next_frame()
        self.assertIsNone(backend.next_frame())
        self.assertEqual(backend.pipe.call_count, 1)
        backend.configure({'prompt_a': 'forest'})
        self.assertFalse(backend.can_generate)
        backend.configure({'krea_reset': True})
        self.assertTrue(backend.can_generate)

    def test_float_rgb_output_is_encoded_as_bgr_and_invalid_shapes_stop_generation(self):
        backend = fake_backend()
        backend.configure({'prompt_a': 'city'})
        def output(state, **kwargs):
            state.values['videos'] = [pixels]
            return state
        backend.pipe.side_effect = output
        pixels = np.zeros((1, 480, 832, 3), dtype=np.float32)
        pixels[..., 0] = 1
        frame, _ = backend.next_frame()
        self.assertEqual(tuple(frame[0, 0]), (0, 0, 255))
        for pixels in (np.zeros((13, 480, 832, 3), dtype=np.uint8),
                       np.zeros((1, 8, 8, 3), dtype=np.uint8),
                       np.full((1, 480, 832, 3), np.nan, dtype=np.float32)):
            backend.configure({'krea_reset': True})
            with self.assertRaises(ValueError):
                backend.next_frame()
            self.assertFalse(backend.can_generate)

    def test_loader_requests_int8_local_components_and_only_text_encoder_offload(self):
        transformer = SimpleNamespace(is_loaded_in_8bit=True)
        pipe = SimpleNamespace(vae=SimpleNamespace(clear_cache=Mock()), text_encoder=object(),
                               tokenizer=Mock(return_value={'input_ids': np.ones((1, 8), dtype=np.int64)}),
                               scheduler=object(), transformer=transformer,
                               load_components=Mock(), update_components=Mock())
        diffusers = SimpleNamespace(AutoModel=SimpleNamespace(from_pretrained=Mock(return_value=transformer)),
                    BitsAndBytesConfig=Mock(), ModularPipeline=SimpleNamespace(from_pretrained=Mock(return_value=pipe)))
        accelerate = SimpleNamespace(cpu_offload=Mock())
        with patch.dict('sys.modules', {'diffusers': diffusers, 'accelerate': accelerate,
                'diffusers.modular_pipelines': SimpleNamespace(PipelineState=lambda: SimpleNamespace(values={}))}), \
                patch('krea_backend.gpu_preflight'), patch.object(assets, 'download_assets', return_value=('model', 'pipe')):
            KreaBackend.load(Path('owned-runtime'))
            kwargs = diffusers.AutoModel.from_pretrained.call_args.kwargs
            self.assertTrue(kwargs['local_files_only'])
            self.assertEqual(kwargs['device_map'], 'cuda')
            diffusers.BitsAndBytesConfig.assert_called_once_with(load_in_8bit=True)
            self.assertEqual([call.kwargs['names'] for call in pipe.load_components.call_args_list],
                             [['tokenizer', 'scheduler'], ['text_encoder', 'vae']])
            self.assertIs(accelerate.cpu_offload.call_args.args[0], pipe.text_encoder)
            transformer.is_loaded_in_8bit = False
            with self.assertRaisesRegex(RuntimeError, 'did not activate'):
                KreaBackend.load(Path('owned-runtime'))

    def test_pinned_runtime_config_is_copied_without_mutating_cached_source(self):
        with tempfile.TemporaryDirectory(prefix='colab-sd-test-krea-') as directory:
            root = Path(directory).resolve()
            snapshot = root / 'snapshot'
            snapshot.mkdir()
            model_index = {name: [None, None, {'repo': 'upstream', 'revision': None}]
                           for name in ('transformer', 'text_encoder', 'tokenizer', 'vae', 'scheduler')}
            files = {'modular_model_index.json': json.dumps(model_index), 'modular_blocks.py': '# source'}
            hashes = {}
            for name, content in files.items():
                (snapshot / name).write_text(content)
                hashes[name] = hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
            with patch.object(assets, 'RUNTIME_HASHES', hashes):
                derived = assets.prepare_pipeline(snapshot, root / 'wan', root)
                config = json.loads((derived / 'modular_model_index.json').read_text())
                self.assertEqual(config['text_encoder'][2]['repo'], str((root / 'wan').resolve()))
                self.assertEqual(config['transformer'], [None, None])
                self.assertEqual((snapshot / 'modular_model_index.json').read_text(), files['modular_model_index.json'])
                (snapshot / 'modular_blocks.py').write_text('# corrupt')
                with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
                    assets.prepare_pipeline(snapshot, root / 'wan', root)

    def test_setup_refuses_an_existing_bridge_before_environment_mutations(self):
        with patch.object(krea_setup.platform, 'system', return_value='Linux'), \
                patch.object(krea_setup.platform, 'machine', return_value='x86_64'), \
                patch.object(krea_setup.socket, 'create_connection', return_value=nullcontext()), \
                patch.object(krea_setup, 'ensure_environment') as environment:
            with self.assertRaisesRegex(RuntimeError, 'left running'):
                krea_setup.setup(Path('unused'))
            environment.assert_not_called()

    def test_setup_installs_only_in_its_environment_then_reuses_packages(self):
        with tempfile.TemporaryDirectory(prefix='colab-sd-test-krea-') as directory:
            root = Path(directory).resolve()
            python = root / 'venv/bin/python'
            (root / 'cloudflared').touch()
            with patch.object(krea_setup.platform, 'system', return_value='Linux'), \
                    patch.object(krea_setup.platform, 'machine', return_value='x86_64'), \
                    patch.object(krea_setup, 'require_unused_port'), \
                    patch.object(krea_setup, 'ensure_environment', return_value=(python, False)), \
                    patch.object(krea_setup.subprocess, 'run') as run:
                krea_setup.setup(root)
                commands = [c.args[0] for c in run.call_args_list]
                self.assertEqual(len(commands), 4)  # Torch, runtime, pip check, GPU/import preflight
                self.assertTrue(all(c[0] == str(python) for c in commands))
                self.assertFalse(any('world_engine' in arg or 'realtime-video.git' in arg
                                     for command in commands for arg in command))
                run.reset_mock()
                krea_setup.setup(root)
                self.assertEqual(run.call_count, 2)
                self.assertFalse(any('install' in c.args[0] for c in run.call_args_list))

    def test_launcher_and_imported_server_select_fixed_krea_backend(self):
        session = colab_launcher.BridgeSession(backend='krea')
        self.assertEqual(session.output_size, (832, 480))
        with self.assertRaises(ValueError):
            colab_launcher.BridgeSession(backend='krea', tensorrt=True)
        session.processes['server'] = SimpleNamespace(poll=lambda: None)
        with patch.object(colab_launcher.urllib.request, 'urlopen', return_value=io.StringIO(json.dumps(
                {'ready': True, 'backend': 'waypoint', 'session_id': session.session_id}))):
            with self.assertRaisesRegex(RuntimeError, 'different backend'):
                session.wait_for_ready()
        result = subprocess.run([sys.executable, '-B', '-c',
                                 'import colab_server as s; print(s.health().body.decode())'],
                cwd=notebook.PROJECT_DIR, env={**os.environ, 'BRIDGE_BACKEND': 'krea', 'ENABLE_TENSORRT': '0'},
                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        status = json.loads(result.stdout)
        self.assertEqual((status['backend'], status['mode'], status['width'], status['height']), ('krea', 'krea', 832, 480))
        self.assertFalse(status['ready'])

    def test_td_handshake_sends_krea_controls_without_world_or_sd_options(self):
        pars = SimpleNamespace(Mode=Par('world'), Resolution=Par('1280x720'), Status=Par(''),
                               Prompta=Par('forest'), Kreapaused=Par(False), Kreainfo=Par(''), Seed=Par(42))
        storage, messages = {}, []
        component = SimpleNamespace(par=pars, fetch=lambda key, default=None: storage.get(key, default),
                                    store=lambda key, value: storage.__setitem__(key, value))
        ws = SimpleNamespace(par=SimpleNamespace(active=Par(True)), sendText=lambda data: messages.append(json.loads(data)))
        namespace = {'parent': lambda: component, 'op': lambda name: ws if name == 'websocket1' else None}
        exec(builder.WEBSOCKET_CODE, namespace)
        namespace['onReceiveText'](ws, 0, json.dumps({'type': 'welcome', 'protocol': 1, 'ready': True,
             'backend': 'krea', 'supported_modes': ['krea'], 'mode': 'krea', 'width': 832, 'height': 480,
             'resolution_locked': True, 'krea_clip': 0, 'krea_block': 1, 'krea_clip_blocks': 64}))
        self.assertEqual(pars.Mode.val, 'krea')
        control = messages[-1]
        self.assertEqual(set(control), {'type', 'mode', 'width', 'height', 'prompt_a', 'seed', 'krea_paused', 'krea_reset'})
        self.assertTrue(storage['server_ready'])
        namespace['send_control'](krea_reset=True)
        self.assertTrue(messages[-1]['krea_reset'])
        self.assertIn('chunk 1/64', pars.Kreainfo.val)

    def test_notebook_sources_match_and_cells_compile(self):
        book = notebook.make_notebook()
        for cell in book['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
        setup = ast.parse(''.join(book['cells'][2]['source']))
        sources = next(ast.literal_eval(node.value) for node in setup.body
                       if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                       and node.targets[0].id == 'sources')
        for name in notebook.SOURCES:
            self.assertEqual(sources[name], (notebook.PROJECT_DIR / name).read_text(encoding='utf-8'))
        self.assertIn("backend='krea'", ''.join(book['cells'][3]['source']))


class KreaTransportChecks(unittest.IsolatedAsyncioTestCase):
    async def test_prompt_and_ack_gates_credit_window_pause_reset_and_disconnect(self):
        backend, ws = fake_backend(), FakeWebSocket()
        with patch.multiple(server, is_world_backend=False, is_krea_backend=True, krea_backend=backend,
                            pipeline_ready=True, current_mode='krea', output_width=832, output_height=480,
                            active_client=None, access_token='', current_prompt_a='default', current_seed=42), \
                patch.object(server, 'get_gpu_telemetry', return_value=('--', '--')):
            task = asyncio.create_task(server.websocket_stream_endpoint(ws))
            async def send(data):
                await ws.incoming.put({'text': json.dumps(data)})
            try:
                await until(lambda: ws.text)
                self.assertEqual(ws.text[0]['supported_modes'], ['krea'])
                await send({'type': 'client_options', 'frame_ack': True, 'frame_window': 1})
                await send({'type': 'ping', 'sent_at': 1})
                await until(lambda: any(row.get('type') == 'pong' for row in ws.text))
                backend.pipe.assert_not_called()
                await send({'type': 'control', 'prompt_a': 'forest'})
                await until(lambda: ws.binary)
                await asyncio.sleep(.1)
                self.assertEqual(len({chunk[0] for chunk in ws.binary}), 1)
                self.assertEqual(backend.pipe.call_count, 1)
                await send({'type': 'control', 'krea_paused': True})
                await until(lambda: backend.paused)
                await send({'type': 'frame_ack', 'frame_id': ws.binary[0][0]})
                await asyncio.sleep(.1)
                self.assertEqual(len({chunk[0] for chunk in ws.binary}), 1)
                await send({'type': 'control', 'krea_reset': True, 'krea_paused': False})
                await until(lambda: backend.pipe.call_count == 2)
            finally:
                await ws.close()
                await asyncio.wait_for(task, 3)
            self.assertIsNone(server.active_client)
            self.assertFalse(backend.can_generate)
            self.assertEqual(len(backend.frames), 0)
