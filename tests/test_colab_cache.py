import os
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
