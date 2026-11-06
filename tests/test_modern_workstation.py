"""All-backend workstation parity without installing GPU packages or models."""
import ast
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

import workstation
import ltx_setup
import krea_setup
import colab_server as server
import check_ltx_stream
from test_ltx import fake_backend as ltx_fixture
from test_krea import fake_backend as krea_fixture


class ModernWorkstationChecks(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix='colab-sd-modern-workstation-')
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name).resolve()

    def prepare(self, backend):
        runtime = self.root / backend
        python = workstation.runtime_python(runtime)
        python.parent.mkdir(parents=True, exist_ok=True)
        python.touch()
        signature = workstation.modern_installer(backend).installation_signature()
        config = {'managed_by': 'colab-sd', 'backend': backend, 'ready': True,
                  'installation': signature, 'revisions': workstation.modern_revisions(backend),
                  'model_cache': str(runtime / 'models')}
        (runtime / 'workstation.json').write_text(json.dumps(config))
        (runtime / 'installed.json').write_text(json.dumps(signature))
        return runtime

    def test_setup_routes_to_separate_backends_and_rejects_mixed_setup_options(self):
        with patch.object(workstation, 'setup_modern') as install:
            for backend in ('krea', 'ltx'):
                self.assertEqual(workstation.main(['setup', '--backend', backend]), 0)
                install.assert_called_with(workstation.RUNTIMES[backend], backend, None)
            for args in (['--backend', 'krea', '--prompt-seeds'], ['--backend', 'ltx', '--torch-version', '2.13.0'],
                         ['--backend', 'krea', '--cuda', 'cu128']):
                install.reset_mock()
                with patch('sys.stderr'):
                    self.assertEqual(workstation.main(['setup', *args]), 1)
                install.assert_not_called()

    def test_setup_reuses_colab_installer_on_both_platforms_without_tunnel_or_parent_mutation(self):
        for backend in ('krea', 'ltx'):
            for system in ('Windows', 'Linux'):
                runtime = self.root / (backend + system)
                cache = self.root / (backend + system + '-cache')
                installer = workstation.modern_installer(backend)
                with self.subTest(backend=backend, system=system), \
                        patch.object(workstation.platform, 'system', return_value=system), \
                        patch.object(workstation.platform, 'machine', return_value='x86_64'), \
                        patch.object(installer, 'setup') as install, patch.object(workstation, 'run_pip') as legacy, \
                        patch.dict(os.environ, {'HF_HUB_CACHE': 'parent-cache', 'HF_TOKEN': 'fixture-secret'}), \
                        patch('builtins.print'):
                    workstation.setup_modern(runtime, backend, cache)
                    install.assert_called_once_with(runtime, install_tunnel=False,
                                                    project_dir=workstation.PROJECT, check_port=False)
                    legacy.assert_not_called()
                    config = json.loads((runtime / 'workstation.json').read_text())
                    self.assertTrue(config['ready'])
                    self.assertEqual(config['backend'], backend)
                    self.assertEqual(config['model_cache'], str(cache))
                    self.assertNotIn('fixture-secret', json.dumps(config))
                    self.assertEqual(os.environ['HF_HUB_CACHE'], 'parent-cache')
                    self.assertEqual(list(cache.iterdir()), [])

    def test_unmanaged_and_other_backend_runtime_folders_are_rejected_without_mutation(self):
        runtime = self.root / 'foreign'
        runtime.mkdir()
        marker = runtime / 'workstation.json'
        for config in ({'managed_by': 'someone-else', 'backend': 'ltx'},
                       {'managed_by': 'colab-sd', 'backend': 'waypoint'}):
            marker.write_text(json.dumps(config))
            before = marker.read_bytes()
            with patch.object(workstation, 'check_platform'), patch.object(ltx_setup, 'setup') as install:
                with self.assertRaisesRegex(RuntimeError, 'another application or backend'):
                    workstation.setup_modern(runtime, 'ltx')
                install.assert_not_called()
                self.assertEqual(marker.read_bytes(), before)
        marker.unlink()
        (runtime / 'keep.txt').write_text('reference')
        with patch.object(workstation, 'check_platform'), self.assertRaisesRegex(RuntimeError, 'not empty'):
            workstation.setup_modern(runtime, 'krea')
        self.assertEqual((runtime / 'keep.txt').read_text(), 'reference')

    def test_failed_setup_preserves_models_and_pairing_and_cannot_claim_ready(self):
        for backend in ('krea', 'ltx'):
            runtime = self.prepare(backend)
            cache = runtime / 'models'
            cache.mkdir()
            (cache / 'weights.fixture').write_bytes(b'preserve')
            token = workstation.pairing_token(runtime)
            with patch.object(workstation, 'check_platform'), \
                    patch.object(workstation.modern_installer(backend), 'setup', side_effect=RuntimeError('failed')):
                with self.assertRaisesRegex(RuntimeError, 'failed'):
                    workstation.setup_modern(runtime, backend)
            self.assertFalse(json.loads((runtime / 'workstation.json').read_text())['ready'])
            self.assertEqual((cache / 'weights.fixture').read_bytes(), b'preserve')
            self.assertEqual(workstation.pairing_token(runtime), token)

    def test_lan_and_tunnel_launch_use_shared_server_and_exact_backend_defaults(self):
        for backend, size, timeout in (('krea', (832, 480), 1800), ('ltx', (1024, 576), 3600)):
            runtime = self.prepare(backend)
            for transport in ('lan', 'cloudflare'):
                with self.subTest(backend=backend, transport=transport), \
                        patch.object(workstation, 'gpu_check') as gpu, \
                        patch.object(workstation, 'BridgeSession') as launch, \
                        patch.object(workstation, 'cloudflared_binary', return_value='cloudflared') as tunnel, \
                        patch('builtins.print'):
                    self.assertEqual(workstation.main(['start', '--backend', backend, '--runtime', str(runtime),
                        '--transport', transport, '--port', '8765', '--advertise-host', '192.168.1.50']), 0)
                    settings = launch.call_args.kwargs
                    self.assertEqual(settings['backend'], backend)
                    self.assertEqual(settings['root'], runtime)
                    self.assertEqual(settings['server_script'], workstation.PROJECT / 'colab_server.py')
                    self.assertEqual(settings['output_size'], size)
                    self.assertEqual(settings['startup_timeout'], timeout)
                    self.assertEqual(settings['max_fps'], 24)
                    self.assertEqual(settings['tunnel_protocol'], 'http2')
                    self.assertEqual(settings['access_token'], workstation.pairing_token(runtime))
                    self.assertEqual(settings['tunnel'], transport == 'cloudflare')
                    env = settings['server_environment']
                    self.assertEqual(env['HF_HUB_CACHE'], str(runtime / 'models'))
                    self.assertIsNone(env['TRANSFORMERS_CACHE'])
                    self.assertNotIn('HF_HOME', env)
                    self.assertNotIn('HF_TOKEN_PATH', env)
                    if backend == 'ltx':
                        self.assertEqual((env['LTX_CONTINUOUS'], env['LTX_CONTINUE_SCENE'], env['LTX_REFRESH_EVERY']),
                                         ('1', '1', '4'))
                        self.assertEqual(env['LTX_PERFORMANCE_PROFILE'], 'resident')
                    else:
                        self.assertEqual(env['KREA_INT8_PROFILE'], 'fast')
                        self.assertEqual(env['KREA_CLIP_BLOCKS'], '64')
                        self.assertEqual(env['DISABLE_SAGEATTENTION'], '1')
                    gpu.assert_called_once_with(workstation.runtime_python(runtime), runtime, backend=backend)
                    if transport == 'lan':
                        tunnel.assert_not_called()
                    else:
                        tunnel.assert_called_once_with(runtime)
                    launch.return_value.stop.assert_called_once()

    def test_start_overrides_are_child_only_and_do_not_change_setup(self):
        image = self.root / 'input.png'
        image.write_bytes(b'fixture; format validation happens on the GPU server before model downloads')
        for backend, options, expected in (
            ('ltx', ['--resolution', '768x512', '--clip-frames', '17', '--performance-profile', 'low_memory',
                     '--no-continuous', '--no-continue-scene', '--refresh-every', '0', '--input-image', str(image)],
             {'LTX_CLIP_FRAMES': '17', 'LTX_PERFORMANCE_PROFILE': 'low_memory', 'LTX_CONTINUOUS': '0',
              'LTX_CONTINUE_SCENE': '0', 'LTX_REFRESH_EVERY': '0', 'LTX_INPUT_IMAGE': str(image)}),
            ('krea', ['--int8-profile', 'balanced', '--clip-blocks', '3'],
             {'KREA_INT8_PROFILE': 'balanced', 'KREA_CLIP_BLOCKS': '3'})):
            runtime = self.prepare(backend)
            marker = runtime / 'workstation.json'
            before = marker.read_bytes()
            with patch.object(workstation, 'gpu_check'), patch.object(workstation, 'BridgeSession') as launch, \
                    patch.dict(os.environ, {'LTX_CONTINUOUS': 'parent', 'HF_HUB_CACHE': 'parent'}), patch('builtins.print'):
                self.assertEqual(workstation.main(['start', '--backend', backend, '--runtime', str(runtime),
                    '--model-cache', str(self.root / 'override'), *options]), 0)
                env = launch.call_args.kwargs['server_environment']
                self.assertTrue(expected.items() <= env.items())
                self.assertEqual(os.environ['LTX_CONTINUOUS'], 'parent')
                self.assertEqual(os.environ['HF_HUB_CACHE'], 'parent')
                self.assertEqual(env['HF_HUB_CACHE'], str(self.root / 'override'))
            self.assertEqual(marker.read_bytes(), before)

    def test_incompatible_controls_and_stale_signatures_fail_before_gpu_or_token(self):
        runtime = self.prepare('ltx')
        bad = (['--backend', 'ltx', '--tensorrt'], ['--backend', 'ltx', '--resolution', '512x288'],
               ['--backend', 'ltx', '--refresh-every', '33'], ['--backend', 'ltx', '--int8-profile', 'fast'],
               ['--backend', 'krea', '--resolution', '1024x768'], ['--backend', 'krea', '--clip-blocks', '65'],
               ['--backend', 'krea', '--no-continuous'], ['--backend', 'sd-turbo', '--clip-frames', '17'],
               ['--backend', 'waypoint', '--performance-profile', 'resident'])
        for options in bad:
            with self.subTest(options=options), patch.object(workstation, 'gpu_check') as gpu, \
                    patch.object(workstation, 'pairing_token') as token, patch('sys.stderr'):
                self.assertEqual(workstation.main(['start', '--runtime', str(runtime), *options]), 1)
                gpu.assert_not_called()
                token.assert_not_called()
        (runtime / 'installed.json').write_text('{}')
        with patch.object(workstation, 'gpu_check') as gpu, patch('sys.stderr'):
            self.assertEqual(workstation.main(['start', '--backend', 'ltx', '--runtime', str(runtime)]), 1)
            gpu.assert_not_called()

    def test_shared_installers_skip_port_and_tunnel_for_workstation_setup(self):
        for installer in (krea_setup, ltx_setup):
            root = self.root / installer.__name__
            python = workstation.runtime_python(root)
            with patch.object(installer.platform, 'system', return_value='Windows'), \
                    patch.object(installer.platform, 'machine', return_value='AMD64'), \
                    patch.object(installer, 'require_unused_port') as port, \
                    patch.object(installer, 'ensure_environment', return_value=(python, False)), \
                    patch.object(installer.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, stdout='')), \
                    patch.object(installer.urllib.request, 'urlretrieve') as download, \
                    patch.object(ltx_setup, 'ensure_source', return_value=root / 'source'), patch('builtins.print'):
                result = installer.setup(root, install_tunnel=False, project_dir=workstation.PROJECT, check_port=False)
                self.assertEqual(result, (python, None))
                port.assert_not_called()
                download.assert_not_called()
                self.assertEqual(json.loads((root / 'installed.json').read_text()), installer.installation_signature())

    def test_gpu_checks_and_launcher_import_need_no_host_torch_or_numpy(self):
        with patch.object(workstation.subprocess, 'run') as run:
            for backend in ('ltx', 'krea'):
                workstation.gpu_check(Path('fixture/python'), self.root, backend)
                ast.parse(run.call_args.args[0][2])
                self.assertIn('gpu_preflight', run.call_args.args[0][2])
                self.assertEqual(run.call_args.kwargs['cwd'], workstation.PROJECT)
        code = '''import builtins, sys
sys.path.insert(0, sys.argv[1])
original=builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch','cv2','numpy'):
        raise AssertionError('GPU/array dependency imported by host launcher: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
import workstation
from colab_launcher import BridgeSession
BridgeSession(backend='ltx',tunnel=False,output_size=(768,512))
workstation.main(['start','--help'])
'''
        subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code, str(workstation.PROJECT)],
                       capture_output=True, text=True, check=True, timeout=15)

    def test_sd_and_sdxl_cache_override_preserves_legacy_defaults_and_auth_location(self):
        runtime = self.root / 'sd'
        python = workstation.runtime_python(runtime)
        python.parent.mkdir(parents=True)
        python.touch()
        wrapper = runtime / 'StreamDiffusion/utils/wrapper.py'
        wrapper.parent.mkdir(parents=True)
        wrapper.touch()
        marker = runtime / 'workstation.json'
        marker.write_text(json.dumps({'managed_by': 'colab-sd', 'ready': True,
                                     'revision': workstation.UPSTREAM_REVISION}))
        before = marker.read_bytes()
        for model in ('sd-turbo', 'sdxl-turbo'):
            with patch.object(workstation, 'gpu_check'), patch.object(workstation, 'BridgeSession') as launch, \
                    patch('builtins.print'), patch.dict(os.environ, {'HF_TOKEN_PATH': 'existing-auth-location'}):
                args = ['start', '--runtime', str(runtime), '--model', model]
                self.assertEqual(workstation.main(args), 0)
                self.assertEqual(launch.call_args.kwargs['server_environment'], {})
                self.assertEqual(workstation.main(args + ['--model-cache', str(self.root / 'sd-models')]), 0)
                env = launch.call_args.kwargs['server_environment']
                self.assertEqual(env['TRANSFORMERS_CACHE'], str(self.root / 'sd-models'))
                self.assertEqual(env['HF_HUB_CACHE'], str(self.root / 'sd-models'))
                self.assertNotIn('HF_TOKEN_PATH', env)
                self.assertEqual(os.environ['HF_TOKEN_PATH'], 'existing-auth-location')
                self.assertEqual(launch.call_args.kwargs['diffusion_model'], model)
        self.assertEqual(marker.read_bytes(), before)

    def test_windows_host_build_tools_path_includes_scripts_without_parent_mutation(self):
        with patch.object(ltx_setup.platform, 'system', return_value='Windows'), \
                patch.dict(os.environ, {'PATH': 'parent-path'}):
            python = self.root / 'python.exe'
            environment = ltx_setup.setup_environment(python)
            self.assertEqual(environment['PATH'].split(os.pathsep)[:2],
                             [str(self.root), str(self.root / 'Scripts')])
            self.assertEqual(os.environ['PATH'], 'parent-path')


class ProbeAuthenticationChecks(unittest.IsolatedAsyncioTestCase):
    async def test_probe_authenticates_without_putting_the_token_in_output_or_errors(self):
        sent = []
        class Socket:
            def __init__(self):
                self.messages = iter([{'type': 'auth_required'}, {'type': 'error', 'code': 'unauthorized'}])
            async def recv(self):
                return json.dumps(next(self.messages))
            async def send(self, message):
                sent.append(json.loads(message))
        class Connection:
            async def __aenter__(self):
                return Socket()
            async def __aexit__(self, *args):
                pass
        fake = SimpleNamespace(connect=lambda *args, **kwargs: Connection())
        with tempfile.TemporaryDirectory(prefix='colab-sd-auth-probe-') as directory, \
                patch.dict(sys.modules, {'websockets': fake}):
            with self.assertRaisesRegex(RuntimeError, '--token-file'):
                await check_ltx_stream.probe('ws://127.0.0.1:8765/ws/stream', Path(directory))
            self.assertEqual(sent, [])
            with self.assertRaisesRegex(RuntimeError, 'authentication failed') as error:
                await check_ltx_stream.probe('ws://127.0.0.1:8765/ws/stream', Path(directory),
                                           access_token='fixture-secret')
            self.assertEqual(sent, [{'type': 'auth', 'token': 'fixture-secret'}])
            self.assertNotIn('fixture-secret', str(error.exception))
            self.assertEqual(list(Path(directory).iterdir()), [])


class AuthenticatedModernChecks(unittest.TestCase):
    def test_krea_and_ltx_authenticate_before_using_the_same_td_controls_and_video(self):
        token = 'fixture-token-for-workstation-more-than-32-characters'
        for name, backend, size in (('ltx', ltx_fixture(), (512, 512)), ('krea', krea_fixture(), (832, 480))):
            with self.subTest(backend=name), patch.multiple(server, access_token=token, backend_name=name,
                    is_world_backend=False, is_krea_backend=name == 'krea', is_ltx_backend=name == 'ltx',
                    ltx_backend=backend if name == 'ltx' else None, krea_backend=backend if name == 'krea' else None,
                    pipeline_ready=True, current_mode=name, current_seed=42, current_prompt_a='default',
                    output_width=size[0], output_height=size[1], active_client=None), \
                    patch.object(server, 'load_pipeline'), \
                    patch.object(server, 'get_gpu_telemetry', return_value=('--', '--')), TestClient(server.app) as client:
                with client.websocket_connect('/ws/stream') as ws:
                    self.assertEqual(ws.receive_json()['type'], 'auth_required')
                    ws.send_json({'type': 'auth', 'token': 'incorrect'})
                    self.assertEqual(ws.receive_json()['code'], 'unauthorized')
                with client.websocket_connect('/ws/stream') as ws:
                    self.assertEqual(ws.receive_json()['type'], 'auth_required')
                    ws.send_json({'type': 'auth', 'token': token})
                    self.assertEqual(ws.receive_json()['supported_modes'], [name])
                    ws.send_json({'type': 'client_options', 'frame_ack': True, 'frame_window': 1})
                    control = {'type': 'control', 'mode': name, 'prompt_a': 'a forest'}
                    if name == 'ltx':
                        control.update(ltx_generate=True, ltx_continuous=True, ltx_continue_scene=True, ltx_refresh_every=4)
                    ws.send_json(control)
                    chunks = {}
                    for _ in range(100):
                        message = ws.receive()
                        if message.get('bytes'):
                            raw = message['bytes']
                            chunks[raw[1]] = raw[3:]
                            if len(chunks) == raw[2]:
                                break
                    pixels = cv2.imdecode(np.frombuffer(b''.join(chunks[i] for i in sorted(chunks)), np.uint8), 1)
                    self.assertEqual(pixels.shape[:2], (size[1], size[0]))
                    ws.send_json({'type': 'control', name + '_paused': True})
                    for _ in range(20):
                        data = ws.receive_json()
                        if data.get('type') == 'configured' and data.get(name + '_paused'):
                            break
                    self.assertTrue(backend.paused)
            self.assertFalse(backend.can_generate)
