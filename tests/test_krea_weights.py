"""Exercise cache repair/staging with real tiny Safetensors, never model weights."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from safetensors.numpy import save_file, load_file

import krea_assets as assets
import krea_weights as cache


class KreaWeightChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='colab-sd-krea-weight-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source, self.target = self.root / 'source', self.root / 'model-files'
        self.source.mkdir()
        self.name = 'transformer/model.safetensors'
        path = self.source / self.name
        path.parent.mkdir()
        save_file({'fixture': np.arange(8, dtype=np.float32)}, path)
        self.valid = path.read_bytes()
        self.expected = {self.name: (len(self.valid), hashlib.sha256(self.valid).hexdigest())}

    def stage(self):
        return cache.stage_snapshot(self.source, self.target, assets.MODEL_ID, assets.MODEL_REVISION,
                                    self.expected, [])

    def test_valid_weights_stage_once_and_reuse_without_drive_reads_or_downloads(self):
        with patch('huggingface_hub.hf_hub_download') as download:
            self.stage()
            np.testing.assert_array_equal(load_file(self.target / self.name)['fixture'], np.arange(8))
            # A good local copy is independent of a later missing/invalid Drive file.
            (self.source / self.name).write_bytes(b'bad cache')
            with patch.object(cache, 'copy_checkpoint', side_effect=AssertionError('unnecessary Drive read')), \
                    patch.object(cache, 'validate_checkpoint', side_effect=AssertionError('unnecessary full rehash')):
                self.stage()
            download.assert_not_called()
        self.assertFalse(list(self.target.glob('.stage-*')))

    def test_pointer_html_and_truncated_downloads_repair_only_the_failed_shard(self):
        for invalid in (b'version https://git-lfs.github.com/spec/v1\n', b'<html>error</html>', self.valid[:8]):
            with self.subTest(content=invalid[:12]):
                (self.source / self.name).write_bytes(invalid)
                # Use a different target per case; existing good copies must be reused.
                self.target = self.root / ('stage-' + str(len(invalid)))
                def repair(repo, filename, **kwargs):
                    self.assertEqual((repo, filename), (assets.MODEL_ID, self.name))
                    self.assertEqual(kwargs, {'revision': assets.MODEL_REVISION, 'force_download': True})
                    (self.source / self.name).write_bytes(self.valid)
                    return str(self.source / self.name)
                with patch('huggingface_hub.hf_hub_download', side_effect=repair) as download:
                    self.stage()
                    download.assert_called_once()
                self.assertEqual((self.target / self.name).read_bytes(), self.valid)

    def test_same_size_corruption_is_detected_by_digest_and_repaired_once(self):
        invalid = bytearray(self.valid)
        invalid[-1] ^= 1
        (self.source / self.name).write_bytes(invalid)
        def repair(*args, **kwargs):
            (self.source / self.name).write_bytes(self.valid)
            return str(self.source / self.name)
        with patch('huggingface_hub.hf_hub_download', side_effect=repair) as download:
            self.stage()
            download.assert_called_once()
        self.assertEqual((self.target / self.name).read_bytes(), self.valid)

    def test_failed_repair_is_bounded_cleans_partial_and_preserves_other_models(self):
        (self.source / self.name).write_bytes(b'bad')
        other = self.source / 'shared.safetensors'
        other.write_bytes(b'preserve')
        with patch('huggingface_hub.hf_hub_download', return_value=str(self.source / self.name)) as download:
            with self.assertRaisesRegex(RuntimeError, 'still invalid after one download'):
                self.stage()
            download.assert_called_once()
        self.assertFalse((self.target / self.name).exists())
        self.assertFalse(list(self.target.glob('.stage-*')))
        self.assertEqual(other.read_bytes(), b'preserve')

    def test_missing_file_is_repaired_and_validated_without_deleting_cache(self):
        (self.source / self.name).unlink()
        def repair(*args, **kwargs):
            (self.source / self.name).write_bytes(self.valid)
            return str(self.source / self.name)
        with patch('huggingface_hub.hf_hub_download', side_effect=repair):
            self.stage()
        self.assertEqual((self.source / self.name).read_bytes(), self.valid)

    def test_modified_local_copy_is_revalidated_and_restored_from_good_drive_cache(self):
        self.stage()
        (self.target / self.name).write_bytes(b'broken local cache')
        with patch('huggingface_hub.hf_hub_download') as download:
            self.stage()
            download.assert_not_called()
        self.assertEqual((self.target / self.name).read_bytes(), self.valid)

    def test_invalid_receipt_rehashes_good_local_weights_without_rereading_drive(self):
        self.stage()
        (self.source / self.name).write_bytes(b'bad persistent cache')
        receipt = self.target / '.validated.json'
        for content in ('invalid JSON', '{}'):
            receipt.write_text(content)
            with patch.object(cache, 'validate_checkpoint', wraps=cache.validate_checkpoint) as validate, \
                    patch.object(cache, 'copy_checkpoint', side_effect=AssertionError('unneeded Drive copy')), \
                    patch('huggingface_hub.hf_hub_download') as download:
                self.stage()
                validate.assert_called_once()
                download.assert_not_called()
            self.assertIn(self.name, json.loads(receipt.read_text()))

    def test_interrupted_staging_preserves_completed_files_and_cleans_owned_partial(self):
        second = 'transformer/second.safetensors'
        (self.source / second).write_bytes(self.valid)
        self.expected[second] = self.expected[self.name]
        original = cache.copy_checkpoint
        def copy(source, partial, expected):
            if source.name == 'second.safetensors':
                partial.write_bytes(b'interrupted')
                raise OSError('disk interrupted')
            original(source, partial, expected)
        with patch.object(cache, 'copy_checkpoint', side_effect=copy), patch('huggingface_hub.hf_hub_download') as download:
            with self.assertRaisesRegex(OSError, 'interrupted'):
                self.stage()
            download.assert_not_called()
        self.assertEqual((self.target / self.name).read_bytes(), self.valid)
        self.assertFalse(list(self.target.glob('.stage-*')))
        with patch.object(cache, 'copy_checkpoint', wraps=original) as copy:
            self.stage()
            self.assertEqual(copy.call_count, 1)
            self.assertEqual(copy.call_args.args[0].name, 'second.safetensors')

    def test_low_disk_space_fails_before_copy_or_force_download(self):
        with patch.object(cache.shutil, 'disk_usage', return_value=type('Usage', (), {'free': 0})()), \
                patch.object(cache, 'copy_checkpoint') as copy, patch('huggingface_hub.hf_hub_download') as download:
            with self.assertRaisesRegex(RuntimeError, 'Persistent model weights were preserved'):
                self.stage()
            copy.assert_not_called()
            download.assert_not_called()

    def test_staging_rejects_paths_outside_owned_model_folder(self):
        self.expected = {'../escape.safetensors': self.expected[self.name]}
        with self.assertRaisesRegex(ValueError, 'outside its model folder'):
            self.stage()
        self.assertFalse((self.root / 'escape.safetensors').exists())

    def test_valid_digest_with_invalid_header_is_still_rejected(self):
        invalid = b'not a Safetensors file'
        (self.source / self.name).write_bytes(invalid)
        self.expected[self.name] = (len(invalid), hashlib.sha256(invalid).hexdigest())
        with patch('huggingface_hub.hf_hub_download', return_value=str(self.source / self.name)):
            with self.assertRaisesRegex(RuntimeError, 'still invalid after one download'):
                self.stage()

    def test_download_assets_loads_only_staged_model_paths_and_preserves_original_config(self):
        wan = self.root / 'wan'
        (wan / 'vae').mkdir(parents=True)
        wan_name = 'vae/diffusion_pytorch_model.safetensors'
        (wan / wan_name).write_bytes(self.valid)
        (wan / 'vae/config.json').write_text('{}')
        # Keep this integration fixture as small as possible, but parse real weights.
        index = {name: [None, None, {'repo': 'original', 'revision': None}]
                 for name in ('transformer', 'text_encoder', 'tokenizer', 'vae', 'scheduler')}
        original = json.dumps(index)
        (self.source / 'modular_model_index.json').write_text(original)
        hashes = {'modular_model_index.json': hashlib.sha256(original.encode()).hexdigest()}
        with patch('huggingface_hub.snapshot_download', side_effect=[str(self.source), str(wan)]) as download, \
                patch.object(assets, 'RUNTIME_HASHES', hashes), patch.object(cache, 'KREA_WEIGHTS', self.expected), \
                patch.object(assets, 'KREA_FILES', {'modular_model_index.json': (len(original), hashes['modular_model_index.json'])}), \
                patch.object(assets, 'WAN_FILES', {'vae/config.json': (2, hashlib.sha256(b'{}').hexdigest())}), \
                patch.object(cache, 'WAN_WEIGHTS', {wan_name: self.expected[self.name]}):
            staged, pipeline = assets.download_assets(self.root / 'runtime')
            self.assertTrue(staged.is_relative_to(self.root / 'runtime/model-files'))
            self.assertEqual((staged / self.name).read_bytes(), self.valid)
            config = json.loads((pipeline / 'modular_model_index.json').read_text())
            self.assertIn('/model-files/wan-', config['text_encoder'][2]['repo'].replace('\\', '/'))
            self.assertEqual((self.source / 'modular_model_index.json').read_text(), original)
            self.assertEqual([call.kwargs['revision'] for call in download.call_args_list],
                             [assets.MODEL_REVISION, assets.WAN_REVISION])
            self.assertNotIn('*.safetensors', download.call_args_list[0].kwargs['allow_patterns'])
            self.assertFalse(any('transformer' in pattern for pattern in assets.WAN_PATTERNS))
