import os
import ast
import json
import hashlib
import io
import http.server
import threading
import zipfile
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import colab_setup


class ModelCacheChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='colab-sd-cache-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.mount = self.root / 'drive'
        (self.mount / 'MyDrive').mkdir(parents=True)
        self.drive = SimpleNamespace(mount=Mock())
        self.modules = patch.dict('sys.modules', {'google.colab': SimpleNamespace(drive=self.drive)})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.environment = patch.dict(os.environ, {
            'HF_HOME': str(self.root / 'private-auth'),
            'HF_TOKEN_PATH': str(self.root / 'private-auth/token'),
            **{key: 'previous-cache' for key in colab_setup.MODEL_CACHE_VARIABLES},
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def drive_cache(self, folder='StreamDiffusion/models'):
        return colab_setup.configure_model_cache(True, folder, mount_point=self.mount)

    def test_repeat_sessions_reuse_the_same_model_files_and_keep_auth_local(self):
        cache = self.drive_cache()
        weight = cache / 'existing-model.safetensors'
        weight.write_bytes(b'cached weight fixture')
        modified = weight.stat().st_mtime_ns
        self.assertEqual(self.drive_cache(), cache)
        self.assertEqual(weight.read_bytes(), b'cached weight fixture')
        self.assertEqual(weight.stat().st_mtime_ns, modified)
        for key in colab_setup.MODEL_CACHE_VARIABLES:
            self.assertEqual(os.environ[key], str(cache))
        self.assertEqual(os.environ['HF_HOME'], str(self.root / 'private-auth'))
        self.assertEqual(os.environ['HF_TOKEN_PATH'], str(self.root / 'private-auth/token'))
        self.drive.mount.assert_called_with(str(self.mount))

    def test_package_download_cache_persists_without_moving_auth_or_user_pip_settings(self):
        with patch.dict(os.environ, {'PIP_CACHE_DIR': 'user-selected-package-cache'}):
            cache = self.drive_cache()
            arguments = colab_setup.pip_cache_arguments()
            self.assertEqual(arguments[0], '--cache-dir')
            package_cache = Path(arguments[1])
            self.assertTrue(package_cache.is_relative_to(self.mount / 'MyDrive'))
            wheel = package_cache / 'fixture.whl'
            wheel.write_bytes(b'reusable package fixture')
            colab_setup.configure_model_cache(True, 'StreamDiffusion/ltx25-models', mount_point=self.mount)
            self.assertEqual(colab_setup.pip_cache_arguments(), arguments)
            self.assertEqual(wheel.read_bytes(), b'reusable package fixture')
            self.assertEqual(os.environ['PIP_CACHE_DIR'], 'user-selected-package-cache')
            colab_setup.configure_model_cache(False, local_cache=self.root / 'local')
            self.assertEqual(colab_setup.pip_cache_arguments(), ['--no-cache-dir'])
            self.assertTrue(wheel.is_file())
            self.assertTrue(cache.is_dir())

    def test_complete_pinned_snapshot_is_reused_without_hub_requests(self):
        cache = self.drive_cache()
        repo, revision = 'bridge-test/cache-only', 'b' * 40
        snapshot = cache / 'models--bridge-test--cache-only' / 'snapshots' / revision
        snapshot.mkdir(parents=True)
        content = b'cached weight fixture'
        (snapshot / 'weights.safetensors').write_bytes(content)
        (snapshot / '__init__.py').write_bytes(b'')
        expected = {'weights.safetensors': (len(content), hashlib.sha256(content).hexdigest()),
                    '__init__.py': (0, hashlib.sha256(b'').hexdigest())}
        with patch('huggingface_hub.snapshot_download', side_effect=AssertionError('Network forbidden')):
            self.assertEqual(colab_setup.cached_model_snapshot(repo, revision, expected), str(snapshot))
        self.assertIsNone(colab_setup.cached_model_snapshot(repo, revision, {'missing.json': (1, '')}))
        self.assertIsNone(colab_setup.cached_model_snapshot(repo, revision, {'weights.safetensors': (999, '')}))
        with self.assertRaises(ValueError):
            colab_setup.cached_model_snapshot(repo, revision, ['../escape'])

    def test_waypoint_notebook_can_reuse_cache_with_only_its_bundled_helpers(self):
        import generate_waypoint_notebook
        cache = self.drive_cache()
        revision = 'c' * 40
        snapshot = cache / 'models--bridge-test--waypoint' / 'snapshots' / revision
        snapshot.mkdir(parents=True)
        (snapshot / 'config.yaml').write_text('fixture: true')
        cell = ast.parse(''.join(generate_waypoint_notebook.make_notebook()['cells'][2]['source']))
        sources = next(ast.literal_eval(node.value) for node in cell.body
                       if isinstance(node, ast.Assign) and node.targets[0].id == 'sources')
        helpers = self.root / 'bundled-helpers'
        helpers.mkdir()
        for name, content in sources.items():
            (helpers / name).write_text(content, encoding='utf-8')
        # Isolated mode excludes the checkout and PYTHONPATH, just like a clean VM
        # that has only the notebook's emitted helper files.
        code = ('import sys; sys.path.insert(0, sys.argv[1]); '
                'from colab_setup import cached_model_snapshot; '
                'assert cached_model_snapshot("bridge-test/waypoint", sys.argv[2], '
                '["config.yaml"]) == sys.argv[3]')
        result = subprocess.run([sys.executable, '-I', '-B', '-c', code,
                                 str(helpers), revision, str(snapshot)], cwd=helpers,
                                env=os.environ.copy(), capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_ltx_two_fresh_runtime_folders_stage_existing_drive_weights_without_download(self):
        import numpy as np
        from safetensors.numpy import save
        import ltx_assets
        cache = self.drive_cache('StreamDiffusion/ltx25-models')
        snapshot = cache / ('models--' + ltx_assets.MODEL_ID.replace('/', '--')) / 'snapshots' / ltx_assets.MODEL_REVISION
        snapshot.mkdir(parents=True)
        content = save({'fixture': np.zeros(2, dtype=np.float32)})
        path = snapshot / 'model.safetensors'
        path.write_bytes(content)
        timestamp = path.stat().st_mtime_ns
        expected = {'model.safetensors': (len(content), hashlib.sha256(content).hexdigest())}
        with patch.object(ltx_assets, 'WEIGHTS', expected), \
                patch('huggingface_hub.snapshot_download', side_effect=AssertionError('Network forbidden')) as download, \
                patch('huggingface_hub.hf_hub_download', side_effect=AssertionError('Network forbidden')) as hub:
            for name in ('fresh-vm-one', 'fresh-vm-two'):
                staged = ltx_assets.download_models(self.root / name)
                self.assertEqual((staged / 'model.safetensors').read_bytes(), content)
                # Restarting the server on the same VM must not even revisit Drive.
                with patch.object(colab_setup, 'cached_model_snapshot', side_effect=AssertionError('Drive query forbidden')):
                    self.assertEqual(ltx_assets.download_models(self.root / name), staged)
            download.assert_not_called()
            hub.assert_not_called()
        self.assertEqual(path.stat().st_mtime_ns, timestamp)

    def test_streamdiffusion_setup_reuses_valid_environment_and_reinstalls_after_drift(self):
        runtime = self.root / 'runtime'
        runtime.mkdir()
        (runtime / 'cloudflared').write_bytes(b'fixture binary')
        previous = Path.cwd()
        try:
            with patch.object(colab_setup.platform, 'system', return_value='Linux'), \
                    patch.object(colab_setup, 'configure_model_cache'), \
                    patch.object(colab_setup.subprocess, 'run') as run, \
                    patch.object(colab_setup, 'core_packages_ready', return_value=True) as ready:
                colab_setup.setup_environment(runtime)
                first = sum('pip' in call.args[0] for call in run.call_args_list)
                self.assertEqual(first, 2)
                run.reset_mock()
                colab_setup.setup_environment(runtime)
                self.assertFalse(any('pip' in call.args[0] for call in run.call_args_list))
                ready.return_value = False
                colab_setup.setup_environment(runtime)
                self.assertEqual(sum('pip' in call.args[0] for call in run.call_args_list), 2)
        finally:
            os.chdir(previous)

    def test_real_pip_reuses_cached_wheel_in_a_fresh_process_with_server_offline(self):
        self.drive_cache()
        content = io.BytesIO()
        with zipfile.ZipFile(content, 'w') as archive:
            archive.writestr('bridge_cache_fixture/__init__.py', '')
            archive.writestr('bridge_cache_fixture-0.0.1.dist-info/METADATA',
                             'Metadata-Version: 2.1\nName: bridge-cache-fixture\nVersion: 0.0.1\n')
            archive.writestr('bridge_cache_fixture-0.0.1.dist-info/WHEEL',
                             'Wheel-Version: 1.0\nGenerator: cache-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
            archive.writestr('bridge_cache_fixture-0.0.1.dist-info/RECORD', '')
        payload, requests = content.getvalue(), []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append(self.path)
                self.send_response(200)
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('Content-Length', str(len(payload)))
                self.send_header('Cache-Control', 'public, max-age=31536000')
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]
        wheel = 'bridge_cache_fixture-0.0.1-py3-none-any.whl'
        url = f'http://127.0.0.1:{port}/{wheel}#sha256={hashlib.sha256(payload).hexdigest()}'
        environment = {key: value for key, value in os.environ.items() if not key.startswith('PIP_')}
        environment.update(PIP_CONFIG_FILE=os.devnull, PYTHONDONTWRITEBYTECODE='1',
                           TMP=str(self.root), TEMP=str(self.root), TMPDIR=str(self.root))

        def fetch(destination):
            result = subprocess.run([sys.executable, '-B', '-m', 'pip', '--disable-pip-version-check',
                *colab_setup.pip_cache_arguments(), '--trusted-host', '127.0.0.1', 'download',
                '--no-deps', '--no-index', '--dest', str(destination), url], env=environment,
                capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((destination / wheel).read_bytes(), payload)

        try:
            fetch(self.root / 'vm-one')
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        fetch(self.root / 'vm-two')
        self.assertEqual(len(requests), 1)

    def test_drive_package_cache_reaches_each_modern_installer(self):
        import krea_setup
        import ltx_setup
        import waypoint_setup
        self.drive_cache()
        expected = colab_setup.pip_cache_arguments()
        for installer in (waypoint_setup, krea_setup, ltx_setup):
            runtime = self.root / installer.__name__
            python = runtime / 'venv' / 'python'
            options = {'install_tunnel': False}
            if installer is not waypoint_setup:
                options['check_port'] = False
            with self.subTest(installer=installer.__name__), \
                    patch.object(installer, 'ensure_environment', return_value=(python, False)), \
                    patch.object(installer.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, stdout='')) as run, \
                    patch.object(ltx_setup, 'ensure_source', return_value=runtime / 'source'):
                installer.setup(runtime, **options)
                installs = [call.args[0] for call in run.call_args_list
                            if 'install' in call.args[0] and '--no-build-isolation' not in call.args[0]]
                self.assertTrue(installs)
                for command in installs:
                    self.assertNotIn('--no-cache-dir', command)
                    index = command.index('--cache-dir')
                    self.assertEqual(command[index:index + 2], expected)

    def test_disabling_drive_selects_local_cache_without_deleting_drive_models(self):
        cache = self.drive_cache()
        weight = cache / 'saved-model'
        weight.write_bytes(b'keep')
        self.drive.mount.reset_mock()
        local = colab_setup.configure_model_cache(False, local_cache=self.root / 'local')
        self.assertEqual(weight.read_bytes(), b'keep')
        self.assertTrue(local.is_dir())
        self.drive.mount.assert_not_called()
        self.assertTrue(all(os.environ[key] == str(local) for key in colab_setup.MODEL_CACHE_VARIABLES))

    def test_invalid_folders_cannot_escape_drive_or_trigger_authorization(self):
        for folder in ('../escape', '.', '', '  ', str(self.root.resolve())):
            with self.subTest(folder=folder), self.assertRaises(ValueError):
                self.drive_cache(folder)
        self.drive.mount.assert_not_called()
        self.assertEqual(os.environ['HF_HUB_CACHE'], 'previous-cache')

    def test_failed_authorization_or_storage_does_not_claim_a_temporary_fallback(self):
        self.drive.mount.side_effect = RuntimeError('authorization cancelled')
        with self.assertRaisesRegex(RuntimeError, 'authorization cancelled'):
            self.drive_cache()
        self.drive.mount.side_effect = None
        with patch('colab_setup.tempfile.NamedTemporaryFile', side_effect=OSError('Drive quota exceeded')):
            with self.assertRaisesRegex(OSError, 'quota exceeded'):
                self.drive_cache()
        self.assertEqual(os.environ['HF_HUB_CACHE'], 'previous-cache')

    def test_incomplete_mount_stops_instead_of_creating_a_fake_drive(self):
        with self.assertRaisesRegex(RuntimeError, 'not mounted'):
            colab_setup.configure_model_cache(True, mount_point=self.root / 'unmounted')
        self.assertFalse((self.root / 'unmounted').exists())

    def test_fresh_server_process_reads_cached_files_without_network_download(self):
        cache = self.drive_cache()
        repo = cache / 'models--bridge-test--cache-only'
        revision = 'a' * 40
        (repo / 'refs').mkdir(parents=True)
        (repo / 'refs/main').write_text(revision)
        snapshot = repo / 'snapshots' / revision
        snapshot.mkdir(parents=True)
        config = snapshot / 'config.json'
        config.write_text('{"cached": true}')
        environment = os.environ.copy()
        environment['HF_HUB_OFFLINE'] = '1'
        code = (
            'from pathlib import Path; import sys; '
            'from huggingface_hub import hf_hub_download; '
            'from huggingface_hub.constants import HF_HUB_CACHE; '
            'assert Path(HF_HUB_CACHE).resolve() == Path(sys.argv[1]).resolve(); '
            'path = hf_hub_download("bridge-test/cache-only", "config.json", local_files_only=True); '
            'assert Path(path).read_text() == \'{"cached": true}\''
        )
        result = subprocess.run([sys.executable, '-c', code, str(cache)], env=environment,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
