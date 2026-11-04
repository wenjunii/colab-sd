"""Workstation Waypoint parity with CPU fixtures; never install GPU packages."""
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

import colab_launcher
import colab_server as server
import waypoint_setup
import workstation
from test_waypoint_seed import hybrid


class WaypointWorkstationChecks(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix='colab-sd-test-workstation-')
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name).resolve()
        self.assertEqual(self.root.parent, Path(tempfile.gettempdir()).resolve())
        self.runtime = self.root / 'waypoint'

    def prepare_runtime(self, prompt_seeds=True):
        self.runtime.mkdir()
        python = workstation.runtime_python(self.runtime)
        python.parent.mkdir(parents=True)
        python.touch()
        (self.runtime / 'workstation.json').write_text(json.dumps({
            'managed_by': 'colab-sd', 'backend': 'waypoint', 'revision': waypoint_setup.RUNTIME_REVISION,
            'ready': True, 'prompt_seeds': prompt_seeds, 'model_cache': str(self.root / 'models')}))
        (self.runtime / 'installed.json').write_text(json.dumps({
            'dependencies': [*waypoint_setup.DEPENDENCIES,
                             *(waypoint_setup.SEED_DEPENDENCIES if prompt_seeds else ())]}))
        return python

    def test_defaults_select_separate_runtimes_and_preserve_sd_cli(self):
        self.assertNotEqual(workstation.DEFAULT_RUNTIME, workstation.DEFAULT_WAYPOINT_RUNTIME)
        self.assertFalse(workstation.DEFAULT_WAYPOINT_RUNTIME.is_relative_to(workstation.DEFAULT_RUNTIME))
        with patch.object(workstation, 'setup_waypoint') as world, patch.object(workstation, 'setup') as sd:
            self.assertEqual(workstation.main(['setup', '--backend', 'waypoint', '--prompt-seeds']), 0)
            world.assert_called_once_with(workstation.DEFAULT_WAYPOINT_RUNTIME, True, None, prompt_seed_model='sd-turbo')
            sd.assert_not_called()
            self.assertEqual(workstation.main(['setup']), 0)
            sd.assert_called_once_with(workstation.DEFAULT_RUNTIME, 'cu130', workstation.TORCH_VERSION)

    def test_setup_reuses_pinned_installer_without_legacy_sd_or_tunnel(self):
        for system in ('Windows', 'Linux'):
            with self.subTest(system=system), patch.object(workstation.platform, 'system', return_value=system), \
                    patch.object(workstation.platform, 'machine', return_value='x86_64'), \
                    patch.object(workstation.shutil, 'which', return_value='git'), \
                    patch.object(waypoint_setup, 'setup') as install, patch('builtins.print'):
                workstation.setup_waypoint(self.runtime, prompt_seeds=True, model_cache=self.root / 'models')
                install.assert_called_once_with(self.runtime, prompt_seeds=True, install_tunnel=False)
                config = json.loads((self.runtime / 'workstation.json').read_text())
                self.assertTrue(config['ready'])
                self.assertTrue(config['prompt_seeds'])
                self.assertEqual(config['backend'], 'waypoint')
                self.assertFalse((self.runtime / 'StreamDiffusion').exists())
                self.assertEqual(list((self.root / 'models').iterdir()), [])

    def test_failed_setup_is_not_ready_and_preserves_models_and_pairing_token(self):
        self.prepare_runtime()
        models = self.root / 'models'
        models.mkdir()
        (models / 'weights').write_bytes(b'keep')
        token = workstation.pairing_token(self.runtime)
        with patch.object(workstation, 'check_platform'), \
                patch.object(workstation.shutil, 'which', return_value='git'), \
                patch.object(waypoint_setup, 'setup', side_effect=RuntimeError('install failed')):
            with self.assertRaisesRegex(RuntimeError, 'install failed'):
                workstation.setup_waypoint(self.runtime, prompt_seeds=True)
        self.assertFalse(json.loads((self.runtime / 'workstation.json').read_text())['ready'])
        self.assertEqual((models / 'weights').read_bytes(), b'keep')
        self.assertEqual(workstation.pairing_token(self.runtime), token)

    def test_setup_refuses_unmanaged_or_other_backend_folder_before_installation(self):
        self.runtime.mkdir()
        marker = self.runtime / 'workstation.json'
        cases = ({'managed_by': 'colab-sd', 'revision': workstation.UPSTREAM_REVISION},
                 {'managed_by': 'some-other-app', 'backend': 'waypoint'})
        for config in cases:
            marker.write_text(json.dumps(config))
            before = marker.read_bytes()
            with patch.object(workstation, 'check_platform'), patch.object(waypoint_setup, 'setup') as install:
                with self.assertRaisesRegex(RuntimeError, 'another application or backend'):
                    workstation.setup_waypoint(self.runtime)
                install.assert_not_called()
                self.assertEqual(marker.read_bytes(), before)
        marker.write_text(json.dumps({'managed_by': 'colab-sd', 'backend': 'waypoint'}))
        before = marker.read_bytes()
        with patch.object(workstation, 'check_platform'), patch.object(workstation, 'run_pip') as pip:
            with self.assertRaisesRegex(RuntimeError, 'another application or backend'):
                workstation.setup(self.runtime)
            pip.assert_not_called()
            self.assertEqual(marker.read_bytes(), before)

    def test_start_uses_native_world_backend_token_and_persistent_child_caches(self):
        python = self.prepare_runtime()
        for transport in ('lan', 'cloudflare'):
            for prompt_seeds in (False, True):
                marker = self.runtime / 'workstation.json'
                config = json.loads(marker.read_text())
                config['prompt_seeds'] = prompt_seeds
                marker.write_text(json.dumps(config))
                (self.runtime / 'installed.json').write_text(json.dumps({'dependencies': [*waypoint_setup.DEPENDENCIES,
                    *(waypoint_setup.SEED_DEPENDENCIES if prompt_seeds else ())]}))
                with self.subTest(transport=transport, seeds=prompt_seeds), \
                        patch.object(workstation, 'gpu_check') as gpu, \
                        patch.object(workstation, 'BridgeSession') as session, \
                        patch.object(workstation, 'cloudflared_binary', return_value='/cloudflared') as binary, \
                        patch('builtins.print'):
                    result = workstation.main(['start', '--backend', 'waypoint', '--runtime', str(self.runtime),
                                               '--transport', transport, '--port', '8765',
                                               '--advertise-host', '192.168.1.50'])
                    self.assertEqual(result, 0)
                    settings = session.call_args.kwargs
                    self.assertEqual((settings['backend'], settings['root'], settings['python_executable']),
                                     ('waypoint', self.runtime, python))
                    self.assertEqual(settings['output_size'], (1280, 720))
                    self.assertFalse(settings['tensorrt'])
                    self.assertEqual(settings['max_fps'], 20)
                    self.assertEqual(settings['startup_timeout'], 1800)
                    self.assertEqual(settings['port'], 8765)
                    self.assertEqual(settings['tunnel_protocol'], 'http2')
                    self.assertEqual(settings['access_token'], workstation.pairing_token(self.runtime))
                    env = settings['server_environment']
                    self.assertEqual(env['WAYPOINT_QUANT'], 'intw8a8')
                    self.assertEqual(env['WAYPOINT_COMPILE_PROFILE'], 'fast')
                    self.assertEqual(env['WAYPOINT_PROMPT_SEEDS'], '1' if prompt_seeds else '0')
                    self.assertEqual(env['HF_HUB_CACHE'], str(self.root / 'models'))
                    self.assertEqual(env['TORCHINDUCTOR_CACHE_DIR'], str(self.runtime / 'inductor-cache'))
                    self.assertIsNone(env['TRANSFORMERS_CACHE'])
                    gpu.assert_called_once_with(python, self.runtime, backend='waypoint')
                    if transport == 'lan':
                        binary.assert_not_called()
                    else:
                        binary.assert_called_once_with(self.runtime)
                    session.return_value.stop.assert_called_once()

    def test_override_profile_quantization_cache_and_fps_apply_only_to_launch(self):
        self.prepare_runtime()
        marker = self.runtime / 'workstation.json'
        before = marker.read_bytes()
        with patch.object(workstation, 'gpu_check'), patch.object(workstation, 'BridgeSession') as session, patch('builtins.print'):
            result = workstation.main(['start', '--backend', 'waypoint', '--runtime', str(self.runtime),
                '--quantization', 'bf16', '--compile-profile', 'max-autotune', '--max-fps', '12',
                '--model-cache', str(self.root / 'override-cache'), '--resolution', '1280x720'])
            self.assertEqual(result, 0)
            settings = session.call_args.kwargs
            self.assertEqual(settings['max_fps'], 12)
            env = settings['server_environment']
            self.assertEqual(env['WAYPOINT_QUANT'], 'bf16')
            self.assertEqual(env['WAYPOINT_COMPILE_PROFILE'], 'max-autotune')
            self.assertEqual(env['HF_HUB_CACHE'], str(self.root / 'override-cache'))
        self.assertEqual(marker.read_bytes(), before)

    def test_wrong_backend_options_or_stale_dependencies_stop_before_gpu_or_token(self):
        self.prepare_runtime()
        base = ['start', '--runtime', str(self.runtime)]
        for args in (['--backend', 'waypoint', '--tensorrt'], ['--backend', 'waypoint', '--resolution', '512x512'],
                     ['--backend', 'sd-turbo', '--quantization', 'bf16'], ['--backend', 'sd-turbo']):
            with self.subTest(args=args), patch.object(workstation, 'gpu_check') as gpu, \
                    patch.object(workstation, 'pairing_token') as token, patch('sys.stderr'):
                self.assertEqual(workstation.main(base + args), 1)
                gpu.assert_not_called()
                token.assert_not_called()
        (self.runtime / 'installed.json').write_text(json.dumps({'dependencies': []}))
        with patch.object(workstation, 'gpu_check') as gpu, patch('sys.stderr'):
            self.assertEqual(workstation.main(base + ['--backend', 'waypoint']), 1)
            gpu.assert_not_called()

    def test_waypoint_setup_can_skip_tunnel_download_for_lan(self):
        self.runtime.mkdir()
        with patch.object(waypoint_setup, 'ensure_environment', return_value=(Path('/isolated/python'), False)), \
                patch.object(waypoint_setup.subprocess, 'run'), patch.object(waypoint_setup.urllib.request, 'urlretrieve') as download:
            python, binary = waypoint_setup.setup(self.runtime, prompt_seeds=True, install_tunnel=False)
        self.assertEqual(python, Path('/isolated/python'))
        self.assertIsNone(binary)
        download.assert_not_called()

    def test_launcher_environment_is_child_only_and_cannot_replace_pairing_or_backend(self):
        environment = {'WAYPOINT_QUANT': 'bf16', 'TRANSFORMERS_CACHE': None,
                       'BRIDGE_BACKEND': 'sd-turbo', 'BRIDGE_ACCESS_TOKEN': 'wrong'}
        session = colab_launcher.BridgeSession(root=self.root, backend='waypoint', tunnel=False,
                    server_environment=environment, access_token='private-test-token')
        environment['WAYPOINT_QUANT'] = 'changed-after-construction'
        with patch.dict(os.environ, {'TRANSFORMERS_CACHE': 'legacy', 'WAYPOINT_QUANT': 'parent'}), \
                patch.object(colab_launcher.socket, 'create_connection', side_effect=ConnectionRefusedError), \
                patch.object(session, '_spawn') as spawn, patch.object(session, 'wait_for_ready'):
            session.start()
            child = spawn.call_args.args[2]
            self.assertEqual(child['WAYPOINT_QUANT'], 'bf16')
            self.assertNotIn('TRANSFORMERS_CACHE', child)
            self.assertEqual(child['BRIDGE_BACKEND'], 'waypoint')
            self.assertEqual(child['BRIDGE_ACCESS_TOKEN'], 'private-test-token')
            self.assertEqual(os.environ['WAYPOINT_QUANT'], 'parent')
            self.assertEqual(os.environ['TRANSFORMERS_CACHE'], 'legacy')


class AuthenticatedWorldChecks(unittest.TestCase):
    def test_authenticated_prompt_seed_world_video_and_probe_use_existing_transport(self):
        backend = hybrid()
        token = 'test-waypoint-token-with-more-than-32-characters'
        with patch.multiple(server, access_token=token, is_world_backend=True, is_krea_backend=False,
                            backend_name='waypoint', world_backend=backend, pipeline_ready=True,
                            current_mode='world', current_seed=42, output_width=1280, output_height=720,
                            active_client=None), patch.object(server, 'load_pipeline'), \
                patch.object(server, 'get_gpu_telemetry', return_value=('--', '--')), TestClient(server.app) as client:
            with client.websocket_connect('/ws/stream') as ws:
                self.assertEqual(ws.receive_json()['type'], 'auth_required')
                ws.send_json({'type': 'auth', 'token': 'incorrect'})
                self.assertEqual(ws.receive_json()['code'], 'unauthorized')
            backend.seed_generator.generate.assert_not_called()
            with client.websocket_connect('/ws/stream') as ws:
                self.assertEqual(ws.receive_json()['type'], 'auth_required')
                ws.send_json({'type': 'auth', 'token': token})
                welcome = ws.receive_json()
                self.assertEqual(welcome['supported_modes'], ['world'])
                self.assertTrue(welcome['world_prompt_seeding'])
                ws.send_json({'type': 'client_options', 'frame_ack': True, 'frame_window': 1})
                ws.send_json({'type': 'control', 'world_seed_source': 'sdturbo', 'prompt_a': 'a forest',
                              'world_auto_seed': False, 'world_seed_interval': 5, 'world_buttons': [87],
                              'world_mouse_x': .2})
                chunks = {}
                for _ in range(100):
                    message = ws.receive()
                    if message.get('bytes'):
                        data = message['bytes']
                        chunks[data[1]] = data[3:]
                        if len(chunks) == data[2]:
                            break
                image = cv2.imdecode(np.frombuffer(b''.join(chunks[i] for i in sorted(chunks)), np.uint8), 1)
                self.assertEqual(image.shape[:2], (720, 1280))
                self.assertEqual(backend.seed_generator.generate.call_count, 1)
                self.assertEqual(backend.buttons, [87])
                self.assertFalse(backend.auto_seed)
                self.assertEqual(backend.seed_interval, 5)
                with client.websocket_connect('/ws/ping') as probe:
                    self.assertEqual(probe.receive_json()['type'], 'auth_required')
                    probe.send_json({'type': 'auth', 'token': token})
                    self.assertEqual(probe.receive_json()['type'], 'probe_ready')
                    probe.send_json({'type': 'ping', 'sent_at': 123})
                    self.assertEqual(probe.receive_json(), {'type': 'pong', 'sent_at': 123})
            deadline = time.monotonic() + 2
            while server.active_client is not None and time.monotonic() < deadline:
                time.sleep(.01)  # disconnect cleanup awaits the owned worker
            self.assertIsNone(server.active_client)
            self.assertEqual(backend.buttons, [])
            self.assertEqual(len(backend.frames), 0)
