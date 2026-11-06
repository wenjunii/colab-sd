"""Drive cache placeholders, metadata integrity and component readiness."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from safetensors.numpy import save_file

import colab_launcher
import colab_server as server
import krea_assets as assets
import krea_backend as backend
import krea_weights as cache


def expected(payload):
    return len(payload), hashlib.sha256(payload).hexdigest()


class MetadataChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='colab-sd-krea-metadata-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.repo = self.root / 'hub' / ('models--' + assets.MODEL_ID.replace('/', '--'))
        self.snapshot = self.repo / 'snapshots' / assets.MODEL_REVISION
        self.snapshot.mkdir(parents=True)
        self.blobs = self.repo / 'blobs'
        self.blobs.mkdir()
        self.local = self.root / 'local'
        self.weight_name = 'transformer/model.safetensors'
        fixture = self.root / 'fixture.safetensors'
        save_file({'fixture': np.arange(8, dtype=np.float32)}, fixture)
        self.weight = fixture.read_bytes()
        self.metadata_name = 'tokenizer/tokenizer.json'
        self.metadata = b'{"version":"1.0","model":{"type":"Unigram"}}'
        self.weights = {self.weight_name: expected(self.weight)}
        self.files = {self.metadata_name: expected(self.metadata)}
        for name, payload in ((self.weight_name, self.weight), (self.metadata_name, self.metadata)):
            digest = expected(payload)[1]
            (self.blobs / digest).write_bytes(payload)
            path = self.snapshot / name
            path.parent.mkdir(parents=True)
            path.write_text('../../../blobs/' + digest, encoding='ascii')
            self.assertEqual(path.stat().st_size, 79)  # matches the user's log

    def stage(self):
        return cache.stage_snapshot(self.snapshot, self.local, assets.MODEL_ID, assets.MODEL_REVISION,
                                    self.weights, self.files)

    def test_79_byte_weight_and_tokenizer_links_reuse_blobs_without_downloads(self):
        with patch('huggingface_hub.hf_hub_download') as download:
            self.stage()
            download.assert_not_called()
        self.assertEqual((self.local / self.weight_name).read_bytes(), self.weight)
        self.assertEqual((self.local / self.metadata_name).read_bytes(), self.metadata)
        self.assertEqual((self.snapshot / self.weight_name).stat().st_size, 79)

    def test_old_shared_blob_text_link_chain_is_resolved_inside_hub_cache(self):
        digest = self.weights[self.weight_name][1]
        shared_digest = 'a' * 64
        shared = self.repo.parent / 'blobs' / 'aa' / shared_digest
        shared.parent.mkdir(parents=True)
        shared.write_bytes(self.weight)
        (self.blobs / digest).write_text('../../blobs/aa/' + shared_digest)
        with patch('huggingface_hub.hf_hub_download') as download:
            self.stage()
            download.assert_not_called()
        self.assertEqual((self.local / self.weight_name).read_bytes(), self.weight)

    def test_bad_tokenizer_is_repaired_once_without_copying_valid_local_weights(self):
        self.stage()
        (self.local / self.metadata_name).write_bytes(b'bad local JSON')
        digest = self.files[self.metadata_name][1]
        (self.blobs / digest).write_bytes(b'bad persistent JSON')
        def repair(repo, filename, **kwargs):
            self.assertEqual((repo, filename), (assets.MODEL_ID, self.metadata_name))
            self.assertEqual(kwargs, {'revision': assets.MODEL_REVISION, 'force_download': True})
            (self.blobs / digest).write_bytes(self.metadata)
            return str(self.snapshot / self.metadata_name)
        with patch('huggingface_hub.hf_hub_download', side_effect=repair) as download, \
                patch.object(cache, 'copy_checkpoint', side_effect=AssertionError('valid weight recopied')):
            self.stage()
            download.assert_called_once()
        self.assertEqual((self.local / self.metadata_name).read_bytes(), self.metadata)

    def test_plain_corrupt_metadata_with_no_blob_is_bounded_to_one_retry(self):
        (self.snapshot / self.metadata_name).write_bytes(b'<html>bad download</html>')
        with patch('huggingface_hub.hf_hub_download', return_value=str(self.snapshot / self.metadata_name)) as download:
            with self.assertRaisesRegex(RuntimeError, 'asset still invalid after one download'):
                self.stage()
            download.assert_called_once()
        self.assertEqual((self.local / self.weight_name).read_bytes(), self.weight)
        self.assertFalse(list(self.local.glob('.stage-*')))

    def test_links_cannot_escape_blob_roots_or_cycle(self):
        path = self.snapshot / self.weight_name
        digest = self.weights[self.weight_name][1]
        path.write_text('../../../../../blobs/' + digest)
        with self.assertRaisesRegex(cache.InvalidCheckpoint, 'outside'):
            cache.resolve_cached_file(path, self.snapshot, assets.MODEL_ID, assets.MODEL_REVISION)
        path.write_text('../../../blobs/' + digest)
        (self.blobs / digest).write_text('../blobs/' + 'b' * 64)
        (self.blobs / ('b' * 64)).write_text('../blobs/' + digest)
        with self.assertRaisesRegex(cache.InvalidCheckpoint, 'depth|cyclic'):
            cache.resolve_cached_file(path, self.snapshot, assets.MODEL_ID, assets.MODEL_REVISION)

    def test_text_links_are_not_followed_for_an_unrecognized_snapshot_revision(self):
        path = self.snapshot / self.weight_name
        self.assertEqual(cache.resolve_cached_file(path, self.snapshot, assets.MODEL_ID, 'wrong-revision'), path)

    def test_valid_digest_does_not_allow_malformed_json(self):
        invalid = b'not JSON'
        (self.snapshot / self.metadata_name).write_bytes(invalid)
        self.files[self.metadata_name] = expected(invalid)
        with patch('huggingface_hub.hf_hub_download', return_value=str(self.snapshot / self.metadata_name)):
            with self.assertRaisesRegex(RuntimeError, 'asset still invalid'):
                self.stage()


class ReadinessChecks(unittest.TestCase):
    def test_required_tokenizer_scheduler_and_model_failures_never_return_a_ready_backend(self):
        for missing in ('tokenizer', 'scheduler', 'vae', 'text_encoder'):
            transformer = SimpleNamespace(is_loaded_in_8bit=True)
            pipe = SimpleNamespace(tokenizer=Mock(return_value={'input_ids': np.ones((1, 8), dtype=np.int64)}),
                scheduler=object(), vae=SimpleNamespace(clear_cache=Mock()), text_encoder=object(),
                transformer=transformer, load_components=Mock(), update_components=Mock())
            setattr(pipe, missing, None)
            auto = Mock(return_value=transformer)
            diffusers = SimpleNamespace(AutoModel=SimpleNamespace(from_pretrained=auto), BitsAndBytesConfig=Mock(),
                ModularPipeline=SimpleNamespace(from_pretrained=Mock(return_value=pipe)))
            with self.subTest(missing=missing), patch.dict('sys.modules', {'diffusers': diffusers,
                    'diffusers.modular_pipelines': SimpleNamespace(PipelineState=Mock()),
                    'accelerate': SimpleNamespace(cpu_offload=Mock())}), \
                    patch('krea_quant.prepare_int8_linears'), \
                    patch.object(backend, 'gpu_preflight'), patch.object(assets, 'download_assets', return_value=('model', 'pipe')):
                with self.assertRaisesRegex(RuntimeError, missing):
                    backend.KreaBackend.load(self._testMethodName)
                if missing in ('tokenizer', 'scheduler'):
                    auto.assert_not_called()

    def test_bad_tokenizer_call_fails_readiness(self):
        for tokenizer in (Mock(side_effect=ValueError('broken tokenizer')), Mock(return_value={'input_ids': np.zeros((0, 8))})):
            with self.assertRaisesRegex(RuntimeError, 'tokenizer readiness check failed'):
                backend.check_tokenizer(SimpleNamespace(tokenizer=tokenizer, scheduler=object()))

    def test_server_does_not_become_ready_after_component_failure(self):
        with patch.multiple(server, is_krea_backend=True, is_world_backend=False, pipeline_ready=False,
                            krea_backend=None, enable_tensorrt=False), \
                patch.object(backend.KreaBackend, 'load', side_effect=RuntimeError('tokenizer missing')):
            with self.assertRaisesRegex(RuntimeError, 'tokenizer missing'):
                server.load_pipeline()
            self.assertFalse(server.pipeline_ready)

    def test_no_symlink_flags_and_allocator_are_isolated_to_clip_backend_children(self):
        with tempfile.TemporaryDirectory(prefix='colab-sd-krea-launch-') as directory:
            with patch.dict(os.environ, {'HF_HUB_DISABLE_SYMLINKS': '0', 'HF_HUB_DISABLE_SHARED_BLOBS': '0'}, clear=True):
                for selection in ('krea', 'sd-turbo', 'ltx'):
                    session = colab_launcher.BridgeSession(root=directory, backend=selection, tunnel=False)
                    with patch.object(colab_launcher.socket, 'create_connection', side_effect=ConnectionRefusedError), \
                            patch.object(session, '_spawn') as spawn, patch.object(session, 'wait_for_ready'):
                        session.start()
                    environment = spawn.call_args.args[2]
                    self.assertEqual(environment['HF_HUB_DISABLE_SYMLINKS'], '1' if selection in ('krea', 'ltx') else '0')
                    self.assertEqual(environment['HF_HUB_DISABLE_SHARED_BLOBS'], '1' if selection in ('krea', 'ltx') else '0')
                    self.assertEqual(os.environ['HF_HUB_DISABLE_SYMLINKS'], '0')
                    if selection in ('krea', 'ltx'):
                        self.assertEqual(environment['PYTORCH_ALLOC_CONF'], 'expandable_segments:True')
                    else:
                        self.assertNotIn('PYTORCH_ALLOC_CONF', environment)
                    self.assertNotIn('PYTORCH_ALLOC_CONF', os.environ)
