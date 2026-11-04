"""SDXL-to-Waypoint contracts; no pretrained weights or CUDA installation."""
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

import build_clean_toe as builder
import colab_launcher
import colab_server as server
import sdxl_stream
import waypoint_seed as seeds
import waypoint_setup
import workstation
from waypoint_backend import WaypointBackend
from test_streaming import Par
from test_waypoint import fake_backend
from test_waypoint_seed import consume_batch


class XLSeedChecks(unittest.TestCase):
    def test_model_pins_match_standalone_xl_and_unknown_models_are_rejected(self):
        self.assertEqual((seeds.XL_MODEL_ID, seeds.XL_MODEL_REVISION),
                         (sdxl_stream.MODEL_ID, sdxl_stream.MODEL_REVISION))
        self.assertEqual(set(seeds.SEED_MODELS), set(waypoint_setup.PROMPT_SEED_MODELS))
        with self.assertRaises(ValueError):
            seeds.PromptSeedGenerator.load('arbitrary/repo')
        with self.assertRaises(ValueError):
            WaypointBackend.load(prompt_seeds=True, prompt_seed_model='arbitrary/repo')

    def test_loader_uses_pinned_xl_pipeline_without_loading_sd_or_tiny_vae(self):
        xl, sd = Mock(), Mock()
        with patch.dict('sys.modules', {'diffusers': SimpleNamespace(
                StableDiffusionXLPipeline=xl, StableDiffusionPipeline=sd)}):
            generator = seeds.PromptSeedGenerator.load('sdxl-turbo')
            settings = xl.from_pretrained.call_args.kwargs
            self.assertEqual(xl.from_pretrained.call_args.args, (seeds.XL_MODEL_ID,))
            self.assertEqual(settings['revision'], seeds.XL_MODEL_REVISION)
            self.assertTrue(settings['use_safetensors'])
            self.assertEqual(settings['variant'], 'fp16')
            self.assertFalse(settings['add_watermarker'])
            sd.from_pretrained.assert_not_called()
            self.assertEqual(generator.model, 'sdxl-turbo')
            self.assertEqual(generator.label, 'SDXL-Turbo')
            xl.from_pretrained.side_effect = RuntimeError('out of memory')
            with self.assertRaisesRegex(RuntimeError, 'out of memory'):
                seeds.PromptSeedGenerator.load('sdxl-turbo')
            sd.from_pretrained.assert_not_called()

    def test_both_embedding_types_blend_cache_and_change_noise_without_reencoding(self):
        pipe = Mock(return_value=SimpleNamespace(images=np.ones((1, 432, 768, 3))))
        pipe.encode_prompt.side_effect = lambda text, **kw: (
            torch.full((1, 77, 8), 1. if text == 'A' else 3.), None,
            torch.full((1, 4), 2. if text == 'A' else 6.), None)
        random = Mock()
        torch_module = SimpleNamespace(inference_mode=torch.inference_mode, Generator=Mock(return_value=random))
        generator = seeds.PromptSeedGenerator(pipe, torch_module, model='sdxl-turbo')
        for alpha, number, token, pooled in ((.25, 42, 1.5, 3.), (.75, 43, 2.5, 5.)):
            rgb = generator.generate('A', 'B', alpha, number)
            self.assertEqual(rgb.shape, (432, 768, 3))
            self.assertEqual(rgb.dtype, np.uint8)
            self.assertTrue(torch.all(pipe.call_args.kwargs['prompt_embeds'] == token))
            self.assertTrue(torch.all(pipe.call_args.kwargs['pooled_prompt_embeds'] == pooled))
        self.assertEqual(pipe.encode_prompt.call_count, 2)
        generator.generate('A', 'C', .25, 44)
        self.assertEqual(pipe.encode_prompt.call_count, 3)
        self.assertEqual(set(generator.embeddings), {'A', 'C'})
        self.assertEqual(pipe.call_args.kwargs['num_inference_steps'], 1)
        self.assertEqual(pipe.call_args.kwargs['guidance_scale'], 0)
        self.assertEqual([call.args[0] for call in random.manual_seed.call_args_list], [42, 43, 44])

    def test_invalid_xl_pixels_report_selected_model(self):
        pipe = Mock(return_value=SimpleNamespace(images=np.full((1, 432, 768, 3), np.nan)))
        pipe.encode_prompt.return_value = (torch.zeros(1, 77, 8), None, torch.zeros(1, 4), None)
        torch_module = SimpleNamespace(inference_mode=torch.inference_mode, Generator=Mock())
        with self.assertRaisesRegex(ValueError, 'SDXL-Turbo returned an invalid seed'):
            seeds.PromptSeedGenerator(pipe, torch_module, model='sdxl-turbo').generate('A', '', 0, 42)

    def test_world_loader_propagates_model_and_disables_capability_when_opted_out(self):
        world = SimpleNamespace(WorldEngine=Mock(), CtrlInput=Mock())
        with patch.dict('sys.modules', {'world_engine': world, 'gemlite': Mock()}), \
                patch('waypoint_backend.engine_class_for_profile', return_value=world.WorldEngine), \
                patch('huggingface_hub.snapshot_download', return_value='pinned/local'), \
                patch.object(torch.cuda, 'is_available', return_value=True), \
                patch.object(torch.cuda, 'get_device_capability', return_value=(8, 0)), \
                patch.object(seeds.PromptSeedGenerator, 'load') as load:
            disabled = WaypointBackend.load(prompt_seed_model='sdxl-turbo')
            load.assert_not_called()
            self.assertIsNone(disabled.progress()['world_seed_model'])
            combined = WaypointBackend.load(prompt_seeds=True, prompt_seed_model='sdxl-turbo')
            load.assert_called_once_with('sdxl-turbo')
            self.assertEqual(combined.progress()['world_seed_model'], 'sdxl-turbo')
            self.assertEqual(combined.progress()['world_seed_model_label'], 'SDXL-Turbo')

    def test_sdxl_seeds_keep_auto_off_manual_pause_and_timer_behavior(self):
        base = fake_backend()
        calls = []
        def generate(*args):
            calls.append(backend.progress()['world_status'])
            return np.zeros((90, 160, 3), np.uint8)
        backend = WaypointBackend(base.engine, base.ctrl_type, base.torch,
            seed_generator=SimpleNamespace(generate=Mock(side_effect=generate)), prompt_seed_model='sdxl-turbo')
        backend.configure({'world_seed_source': 'sdturbo', 'prompt_a': 'forest', 'world_auto_seed': False})
        consume_batch(backend)
        backend._seed_deadline = 0
        consume_batch(backend)
        self.assertEqual(backend.seed_count, 1)
        backend.configure({'world_reseed': True})
        consume_batch(backend)
        self.assertEqual(backend.seed_count, 2)
        backend.configure({'world_paused': True, 'world_auto_seed': True})
        self.assertIsNone(backend.next_frame())
        backend.configure({'world_paused': False})
        backend._seed_deadline = 0
        consume_batch(backend)
        self.assertEqual(backend.seed_count, 3)
        self.assertTrue(all('SDXL-Turbo' in status for status in calls))
        self.assertEqual(backend.seed_image.shape, (4, 720, 1280, 3))

    def test_td_status_shows_selected_seed_model_and_menu_preserves_old_value(self):
        pars = SimpleNamespace(Status=Par(''), Worldseedinfo=Par(''))
        component = SimpleNamespace(par=pars)
        namespace = {'parent': lambda: component, 'op': lambda name: None}
        exec(builder.WEBSOCKET_CODE, namespace)
        namespace['onReceiveText'](None, 0, json.dumps({'type': 'stats', 'world_prompt_seeding': True,
            'world_seed_model_label': 'SDXL-Turbo', 'world_seed_count': 2, 'world_seed_ms': 123}))
        self.assertIn('SDXL-Turbo seeds: 2', pars.Worldseedinfo.val)
        self.assertIn('sdturbo "Prompt Model (A/B)"', builder.CPARM_CONTENT)

    def test_launcher_rejects_sd_seeds_when_xl_seeds_were_requested(self):
        with tempfile.TemporaryDirectory(prefix='colab-sd-seed-launch-') as directory:
            session = colab_launcher.BridgeSession(root=directory, backend='waypoint', tunnel=False,
                server_environment={'WAYPOINT_PROMPT_SEEDS': '1', 'WAYPOINT_PROMPT_SEED_MODEL': 'sdxl-turbo'})
            response = json.dumps({'ready': True, 'session_id': session.session_id,
                                   'backend': 'waypoint', 'world_seed_model': 'sd-turbo'})
            with patch.object(colab_launcher.socket, 'create_connection', side_effect=ConnectionRefusedError), \
                    patch.object(session, '_spawn'), patch.object(colab_launcher.urllib.request, 'urlopen',
                                                               return_value=io.StringIO(response)):
                with self.assertRaisesRegex(RuntimeError, 'prompt seeds were requested'):
                    session.start()

    def test_server_reads_seed_model_from_launch_environment(self):
        with patch.dict(os.environ, {'WAYPOINT_PROMPT_SEEDS': '1', 'WAYPOINT_PROMPT_SEED_MODEL': 'sdxl-turbo'}), \
                patch.multiple(server, is_world_backend=True, is_krea_backend=False, enable_tensorrt=False,
                               pipeline_ready=False, world_backend=None, pipeline_revision=0), \
                patch.object(WaypointBackend, 'load', return_value=object()) as load:
            server.load_pipeline()
            self.assertTrue(load.call_args.kwargs['prompt_seeds'])
            self.assertEqual(load.call_args.kwargs['prompt_seed_model'], 'sdxl-turbo')


class XLWorkstationSeedChecks(unittest.TestCase):
    def test_setup_persists_model_and_start_can_override_it_without_changing_installation(self):
        with tempfile.TemporaryDirectory(prefix='colab-sd-seed-setup-') as directory:
            root = Path(directory).resolve()
            with patch.object(workstation, 'check_platform'), patch.object(workstation.shutil, 'which', return_value='git'), \
                    patch.object(waypoint_setup, 'setup'), patch('builtins.print'):
                self.assertEqual(workstation.main(['setup', '--backend', 'waypoint', '--runtime', str(root),
                                  '--prompt-seeds', '--prompt-seed-model', 'sdxl-turbo']), 0)
            marker = root / 'workstation.json'
            before = marker.read_bytes()
            self.assertEqual(json.loads(before)['prompt_seed_model'], 'sdxl-turbo')
            python = workstation.runtime_python(root)
            python.parent.mkdir(parents=True)
            python.touch()
            (root / 'installed.json').write_text(json.dumps({'dependencies': [*waypoint_setup.DEPENDENCIES,
                                                                             *waypoint_setup.SEED_DEPENDENCIES]}))
            for flags, expected in (([], 'sdxl-turbo'), (['--prompt-seed-model', 'sd-turbo'], 'sd-turbo')):
                with patch.object(workstation, 'gpu_check'), patch.object(workstation, 'BridgeSession') as launcher, patch('builtins.print'):
                    self.assertEqual(workstation.main(['start', '--backend', 'waypoint', '--runtime', str(root), *flags]), 0)
                    self.assertEqual(launcher.call_args.kwargs['server_environment']['WAYPOINT_PROMPT_SEED_MODEL'], expected)
                self.assertEqual(marker.read_bytes(), before)

    def test_seed_model_selection_requires_prompt_seeding_and_waypoint_backend(self):
        with patch.object(workstation, 'setup_waypoint') as setup, patch.object(workstation, 'gpu_check') as gpu, \
                patch('sys.stderr'):
            self.assertEqual(workstation.main(['setup', '--backend', 'waypoint', '--prompt-seed-model', 'sdxl-turbo']), 1)
            self.assertEqual(workstation.main(['start', '--prompt-seed-model', 'sdxl-turbo']), 1)
            setup.assert_not_called()
            gpu.assert_not_called()
