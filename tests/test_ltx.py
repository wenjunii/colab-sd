"""LTX clip lifecycle, protocol, source containment and notebook contracts; CPU only."""
import ast
import asyncio
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
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
import zipfile

import numpy as np
import build_clean_toe as builder
import colab_launcher
import colab_server as server
import generate_ltx_notebook as notebook
import ltx_assets as assets
import ltx_setup
from ltx_backend import LTXBackend, validate_ltx_controls, startup_size
from ltx_runtime import CachedPromptEncoder
from test_streaming import FakeWebSocket, until
from test_waypoint import Par


def fake_backend():
    pipe = Mock(side_effect=lambda **kwargs: SimpleNamespace(
        video=iter([np.zeros((17, 512, 512, 3), dtype=np.uint8)])))
    return LTXBackend(pipe, SimpleNamespace(inference_mode=nullcontext), num_frames=17)


class LTXChecks(unittest.TestCase):
    def test_prompt_cache_reuses_embeddings_across_seed_and_image_changes_and_releases_on_edit(self):
        encoder, release = Mock(side_effect=lambda *a, **k: [object()]), Mock()
        cached = CachedPromptEncoder(encoder, on_miss=release)
        first = cached(['forest'], enhance_first_prompt=False, enhance_prompt_image='first.png', enhance_prompt_seed=42)
        self.assertIs(first, cached(['forest'], enhance_first_prompt=False,
                                    enhance_prompt_image='next.png', enhance_prompt_seed=43))
        self.assertTrue(cached.cache_hit)
        self.assertEqual(encoder.call_count, 1)
        self.assertIsNot(first, cached(['city'], enhance_first_prompt=False))
        self.assertEqual(release.call_count, 2)
        encoder.side_effect = RuntimeError('encode failed')
        with self.assertRaises(RuntimeError):
            cached(['new prompt'], enhance_first_prompt=False)
        self.assertIsNone(cached.value)
        self.assertIsNone(cached.key)

    def test_continuous_generation_requires_start_advances_seeds_and_honors_pause_off_and_disconnect(self):
        backend = fake_backend()
        backend.configure({'prompt_a': 'forest', 'ltx_continuous': True})
        self.assertFalse(backend.can_generate)
        backend.configure({'ltx_generate': True})
        for _ in range(17):
            backend.next_frame()
        self.assertTrue(backend.can_generate)
        self.assertEqual(backend.progress()['world_phase'], 'next_clip')
        backend.configure({'ltx_paused': True})
        self.assertIsNone(backend.next_frame())
        self.assertEqual(backend.pipe.call_count, 1)
        backend.configure({'ltx_paused': False})
        backend.next_frame()
        self.assertEqual(backend.pipe.call_args.kwargs['seed'], 43)
        self.assertEqual(len(backend.frames), 16)
        backend.configure({'ltx_continuous': False})
        for _ in range(16):
            backend.next_frame()
        self.assertFalse(backend.can_generate)
        backend.configure({'ltx_continuous': True})
        backend.release_controls()
        self.assertFalse(backend.can_generate)
        backend.configure({'prompt_a': 'city', 'seed': 99, 'ltx_generate': True})
        backend.next_frame()
        self.assertEqual(backend.pipe.call_args.kwargs['seed'], 99)

    def test_continuation_uses_last_rgb_frame_and_cleans_only_owned_image(self):
        with tempfile.TemporaryDirectory(prefix='colab-sd-ltx-chain-') as directory:
            root = Path(directory).resolve()
            original = root / 'original.txt'
            original.write_text('user input fixture')
            backend = fake_backend()
            pixels = np.zeros((17, 512, 512, 3), dtype=np.uint8)
            pixels[..., 0] = 255
            backend.pipe.side_effect = lambda **_: SimpleNamespace(video=iter([pixels]))
            backend.image, backend.workspace = original, root
            backend.image_factory = Mock(side_effect=lambda path, *_, **__: path)
            backend.configure({'prompt_a': 'forest', 'ltx_continuous': True,
                               'ltx_continue_scene': True, 'ltx_generate': True})
            for _ in range(17):
                backend.next_frame()
            self.assertEqual(backend.pipe.call_args.kwargs['images'], [original])
            backend.next_frame()
            carry = Path(backend.image_factory.call_args.args[0])
            self.assertTrue(carry.is_relative_to(root))
            import cv2
            self.assertEqual(tuple(cv2.imread(str(carry))[0, 0]), (0, 0, 255))
            self.assertEqual(backend.pipe.call_args.kwargs['images'], [str(carry)])
            self.assertEqual(backend.image_factory.call_args.kwargs, {'crf': 0})
            backend.release_controls()
            self.assertFalse(carry.exists())
            self.assertEqual(original.read_text(), 'user input fixture')

    def test_lookahead_is_bounded_and_playback_uses_warm_not_cold_timing(self):
        backend = fake_backend()
        backend.configure({'prompt_a': 'forest', 'ltx_continuous': True, 'ltx_generate': True})
        with patch('ltx_backend.time.perf_counter', side_effect=[0., 100., 100., 102.]):
            backend.render_next_clip()
            self.assertFalse(backend.can_deliver)
            self.assertEqual(backend.progress()['world_phase'], 'buffering')
            backend.render_next_clip()
        self.assertTrue(backend.can_deliver)
        self.assertEqual(backend.progress()['ltx_queued_frames'], 34)
        self.assertAlmostEqual(backend.playback_fps, 17 / 2.4)
        self.assertFalse(backend.needs_render)
        self.assertFalse(backend.render_next_clip())
        for _ in range(17):
            self.assertIsNotNone(backend.pop_frame())
        self.assertEqual(len(backend.frames), 17)
        self.assertTrue(backend.needs_render)
        backend.configure({'ltx_continuous': False})
        for _ in range(17):
            self.assertIsNotNone(backend.pop_frame())
        self.assertFalse(backend.needs_render)
        backend.release_controls()

    def test_prompt_edit_discards_inflight_old_scene_and_does_not_require_another_pulse(self):
        backend = fake_backend()
        entered, resume = threading.Event(), threading.Event()
        original = backend.pipe.side_effect
        def render(**kwargs):
            entered.set()
            if not resume.wait(5):
                raise RuntimeError('test gate timed out')
            return original(**kwargs)
        backend.pipe.side_effect = render
        backend.configure({'prompt_a': 'forest', 'ltx_generate': True})
        with ThreadPoolExecutor(max_workers=1) as worker:
            pending = worker.submit(backend.render_next_clip)
            try:
                self.assertTrue(entered.wait(2))
                self.assertTrue(backend.configure({'prompt_a': 'city'}))
                self.assertTrue(backend.progress()['ltx_prompt_pending'])
                resume.set()
                self.assertFalse(pending.result(timeout=2))
            finally:
                resume.set()
        self.assertEqual(len(backend.frames), 0)
        self.assertIsNone(backend._last_rgb)
        self.assertTrue(backend.needs_render)
        backend.next_frame()
        self.assertEqual(backend.pipe.call_args.kwargs['prompt'], 'city')
        self.assertEqual(backend.pipe.call_args.kwargs['images'], [])
        self.assertFalse(backend.progress()['ltx_prompt_pending'])
        backend.release_controls()

    def test_continuous_off_drops_lookahead_without_carrying_an_unseen_scene(self):
        backend = fake_backend()
        backend.pipe.side_effect = lambda **kw: SimpleNamespace(
            video=iter([np.full((17, 512, 512, 3), kw['seed'], dtype=np.uint8)]))
        backend.configure({'prompt_a': 'forest', 'ltx_continuous': True, 'ltx_generate': True})
        backend.render_next_clip()
        backend.render_next_clip()
        self.assertEqual(int(backend._last_rgb[0, 0, 0]), 43)
        backend.configure({'ltx_continuous': False})
        self.assertEqual(backend.progress()['ltx_queued_frames'], 17)
        self.assertEqual(int(backend._last_rgb[0, 0, 0]), 42)
        backend.release_controls()

    def test_quality_refresh_breaks_recursive_feedback_without_dropping_seed_or_prompt(self):
        with tempfile.TemporaryDirectory(prefix='colab-sd-ltx-refresh-') as directory:
            backend = fake_backend()
            backend.workspace = Path(directory)
            backend.image_factory = Mock(side_effect=lambda path, *_, **__: path)
            backend.configure({'prompt_a': 'forest', 'ltx_continuous': True,
                               'ltx_continue_scene': True, 'ltx_refresh_every': 2, 'ltx_generate': True})
            try:
                for _ in range(17 * 3):
                    backend.next_frame()
                calls = [c.kwargs for c in backend.pipe.call_args_list]
                self.assertEqual([c['seed'] for c in calls], [42, 43, 44])
                self.assertEqual([c['prompt'] for c in calls], ['forest'] * 3)
                self.assertEqual(calls[0]['images'], [])
                self.assertEqual(len(calls[1]['images']), 1)
                self.assertEqual(calls[2]['images'], [])
                self.assertTrue(backend.progress()['ltx_quality_refresh'])
            finally:
                backend.release_controls()

    def test_failed_continuous_clip_never_retries_without_generate(self):
        backend = fake_backend()
        backend.profile = 'resident'
        backend.pipe.side_effect = RuntimeError('CUDA out of memory')
        backend.configure({'prompt_a': 'forest', 'ltx_continuous': True, 'ltx_generate': True})
        with self.assertRaises(ValueError):
            backend.next_frame()
        self.assertFalse(backend.can_generate)
        backend.configure({'ltx_paused': False, 'ltx_continuous': True})
        self.assertIsNone(backend.next_frame())
        self.assertEqual(backend.pipe.call_count, 1)
        backend.pipe.stage.release_transformer.assert_called_once()

    def test_denied_access_stops_before_any_weight_download(self):
        with tempfile.TemporaryDirectory(prefix='colab-sd-ltx-access-') as directory, \
                patch('shutil.disk_usage', return_value=SimpleNamespace(free=100 * 1024**3)), \
                patch('huggingface_hub.hf_hub_download', side_effect=RuntimeError('access denied')) as head, \
                patch('huggingface_hub.snapshot_download') as download:
            with self.assertRaisesRegex(RuntimeError, 'LTX model access failed'):
                assets.download_models(directory)
            self.assertTrue(head.call_args.kwargs['dry_run'])
            download.assert_not_called()

    def test_installer_reuses_its_environment_and_never_records_credentials(self):
        with tempfile.TemporaryDirectory(prefix='colab-sd-ltx-setup-') as directory:
            root = Path(directory).resolve()
            (root / 'cloudflared').write_bytes(b'fixture')
            with patch.object(ltx_setup.platform, 'system', return_value='Linux'), \
                    patch.object(ltx_setup, 'require_unused_port'), \
                    patch.object(ltx_setup, 'ensure_source', return_value=root / 'source'), \
                    patch.object(ltx_setup, 'ensure_environment', return_value=(root / 'venv/bin/python', False)), \
                    patch.object(ltx_setup.subprocess, 'run',
                                 return_value=subprocess.CompletedProcess([], 0, stdout='')) as run, \
                    patch.dict(os.environ, {'HF_TOKEN': 'test-only-placeholder'}):
                ltx_setup.setup(root)
                commands = [call.args[0] for call in run.call_args_list]
                self.assertIn(ltx_setup.TORCH_INDEX, commands[0])
                self.assertIn('--no-deps', commands[2])
                for call in run.call_args_list:
                    self.assertEqual(call.kwargs['env']['PATH'].split(os.pathsep)[0], str(root / 'venv/bin'))
                self.assertNotIn('test-only-placeholder', (root / 'installed.json').read_text())
                run.reset_mock()
                ltx_setup.setup(root)
                self.assertEqual(run.call_count, 2)  # pip check + import/GPU preflight; no reinstall

    def test_child_finds_venv_build_tool_without_changing_parent_path(self):
        with tempfile.TemporaryDirectory(prefix='colab-sd-ltx-path-') as directory:
            root = Path(directory).resolve()
            scripts, other_scripts = root / 'venv/bin', root / 'other-bin'
            scripts.mkdir(parents=True)
            other_scripts.mkdir()
            tool_name = 'ltx-build-tool.cmd' if os.name == 'nt' else 'ltx-build-tool'
            for folder in (scripts, other_scripts):
                tool = folder / tool_name
                tool.write_text('fixture only', encoding='utf-8')
                tool.chmod(0o755)
            with patch.dict(os.environ, {'PATH': str(other_scripts), 'PIP_INDEX_URL': 'untrusted-fixture'}):
                environment = ltx_setup.setup_environment(scripts / 'python')
                result = subprocess.run([sys.executable, '-B', '-c',
                    'import shutil, sys; print(shutil.which(sys.argv[1]))', tool_name],
                    env=environment, capture_output=True, text=True, check=True, timeout=15)
                self.assertEqual(Path(result.stdout.strip()), scripts / tool_name)
                self.assertEqual(os.environ['PATH'], str(other_scripts))
                self.assertNotIn('PIP_INDEX_URL', environment)

    def test_failed_setup_shows_resolver_cause_without_recording_success_or_token(self):
        with tempfile.TemporaryDirectory(prefix='colab-sd-ltx-failure-') as directory:
            root = Path(directory).resolve()
            failure = subprocess.CalledProcessError(1, ['pip'],
                output='ERROR: incompatible safetensors pin\ncredential=test-only-placeholder\n')
            with patch.object(ltx_setup.platform, 'system', return_value='Linux'), \
                    patch.object(ltx_setup, 'require_unused_port'), \
                    patch.object(ltx_setup, 'ensure_source', return_value=root / 'source'), \
                    patch.object(ltx_setup, 'ensure_environment', return_value=(root / 'venv/bin/python', False)), \
                    patch.object(ltx_setup.subprocess, 'run', side_effect=[
                        subprocess.CompletedProcess([], 0, stdout='Torch already installed'), failure]) as run, \
                    patch.dict(os.environ, {'HF_TOKEN': 'test-only-placeholder'}), \
                    patch('sys.stdout', new_callable=io.StringIO) as output:
                with self.assertRaisesRegex(RuntimeError, 'install runtime dependencies') as caught:
                    ltx_setup.setup(root)
                message = str(caught.exception)
                self.assertIn('incompatible safetensors pin', message)
                self.assertNotIn('test-only-placeholder', message + output.getvalue())
                self.assertEqual(run.call_count, 2)
                self.assertFalse((root / 'installed.json').exists())

    def test_control_validation_rejects_bad_patches_and_resolution_changes(self):
        for value in (None, [], {'mode': 'krea'}, {'seed': -1}, {'seed': True},
                      {'width': 768}, {'prompt_a': ''}, {'ltx_generate': 1}, {'ltx_paused': 'true'},
                      {'ltx_continuous': 1}, {'ltx_continue_scene': 'true'},
                      {'ltx_refresh_every': True}, {'ltx_refresh_every': -1}, {'ltx_refresh_every': 33}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_ltx_controls(value, (512, 512))
        self.assertEqual(validate_ltx_controls({'prompt_a': ' forest ', 'seed': 42,
            'ltx_generate': True, 'ltx_paused': False}, (512, 512)),
            {'prompt_a': 'forest', 'seed': 42, 'ltx_generate': True, 'ltx_paused': False})

    def test_manual_generation_is_bounded_and_does_not_repeat(self):
        backend = fake_backend()
        backend.configure({'prompt_a': 'forest'})
        self.assertIsNone(backend.next_frame())
        backend.pipe.assert_not_called()
        backend.configure({'ltx_generate': True})
        for remaining in range(16, -1, -1):
            frame, duration = backend.next_frame()
            self.assertEqual(frame.shape, (512, 512, 3))
            self.assertEqual(len(backend.frames), remaining)
        self.assertFalse(backend.can_generate)
        self.assertIsNone(backend.next_frame())
        self.assertEqual(backend.pipe.call_count, 1)
        self.assertEqual(backend.progress()['ltx_clip'], 1)
        self.assertEqual(backend.pipe.call_args.kwargs['num_frames'], 17)

    def test_bfloat16_decoder_frames_convert_to_valid_bgr_bytes(self):
        import torch
        backend = fake_backend()
        pixels = torch.zeros((17, 512, 512, 3), dtype=torch.bfloat16)
        pixels[..., 0], pixels[..., 1] = 1.0, 0.5
        backend.pipe.side_effect = lambda **_: SimpleNamespace(video=iter([pixels]))
        backend.configure({'prompt_a': 'red and green', 'ltx_generate': True})
        frame, _ = backend.next_frame()
        self.assertEqual(frame.dtype, np.uint8)
        self.assertEqual(tuple(frame[0, 0]), (0, 128, 255))

    def test_pause_next_prompt_seed_and_image_are_applied_at_clip_request(self):
        backend = fake_backend()
        image_condition = object()
        backend.image = image_condition
        backend.configure({'prompt_a': 'forest', 'ltx_generate': True})
        backend.next_frame()
        backend.configure({'ltx_paused': True})
        self.assertIsNone(backend.next_frame())
        self.assertEqual(len(backend.frames), 16)
        backend.configure({'prompt_a': 'city', 'seed': 99, 'ltx_generate': True, 'ltx_paused': False})
        self.assertEqual(len(backend.frames), 0)
        backend.next_frame()
        args = backend.pipe.call_args.kwargs
        self.assertEqual((args['prompt'], args['seed'], args['images']), ('city', 99, [image_condition]))
        backend.release_controls()
        self.assertFalse(backend.can_generate)
        self.assertEqual(len(backend.frames), 0)

    def test_failed_clip_stops_until_explicit_generate_and_invalid_output_is_bounded(self):
        backend = fake_backend()
        for batch in (np.zeros((18, 512, 512, 3), dtype=np.uint8),
                      np.zeros((1, 8, 8, 3), dtype=np.uint8),
                      np.full((1, 512, 512, 3), np.nan, dtype=np.float32),
                      np.zeros((17, 512, 512, 3), dtype=np.uint16)):
            backend.pipe.side_effect = lambda **_: SimpleNamespace(video=iter([batch]))
            backend.configure({'prompt_a': 'test', 'ltx_generate': True})
            with self.assertRaisesRegex(ValueError, 'LTX clip failed'):
                backend.next_frame()
            calls = backend.pipe.call_count
            backend.configure({'prompt_a': 'changed'})
            self.assertIsNone(backend.next_frame())
            self.assertEqual(backend.pipe.call_count, calls)
            self.assertEqual(len(backend.frames), 0)
            self.assertEqual(backend.progress()['world_phase'], 'error')

    def test_startup_size_and_launcher_identity(self):
        with patch.dict(os.environ, clear=True):
            self.assertEqual(startup_size(), (1024, 576))
        with patch.dict(os.environ, {'BRIDGE_INITIAL_WIDTH': '768', 'BRIDGE_INITIAL_HEIGHT': '512'}):
            self.assertEqual(startup_size(), (768, 512))
        for size in ((768, 512), (1024, 576)):
            session = colab_launcher.BridgeSession(backend='ltx', output_size=size)
            self.assertEqual(session.output_size, size)
        with self.assertRaises(ValueError):
            colab_launcher.BridgeSession(backend='ltx', tensorrt=True)
        with self.assertRaises(ValueError):
            colab_launcher.BridgeSession(backend='ltx', output_size=(832, 480))
        result = subprocess.run([sys.executable, '-B', '-c',
            'import colab_server as s; print(s.health().body.decode())'], cwd=notebook.PROJECT_DIR,
            env={**os.environ, 'BRIDGE_BACKEND': 'ltx', 'ENABLE_TENSORRT': '0',
                 'BRIDGE_INITIAL_WIDTH': '1024', 'BRIDGE_INITIAL_HEIGHT': '576'},
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        health = json.loads(result.stdout)
        self.assertEqual((health['backend'], health['mode'], health['width'], health['height']), ('ltx', 'ltx', 1024, 576))
        self.assertFalse(health['ready'])
        self.assertTrue(health['resolution_locked'])

    def test_td_handshake_and_generate_pulse_send_only_ltx_controls(self):
        pars = SimpleNamespace(Mode=Par('krea'), Resolution=Par('512'), Status=Par(''),
                               Prompta=Par('forest'), Ltxpaused=Par(False), Ltxinfo=Par(''), Seed=Par(42),
                               Ltxcontinuous=Par(False), Ltxchain=Par(False), Ltxspeed=Par(''))
        storage, messages = {}, []
        component = SimpleNamespace(par=pars, fetch=lambda key, default=None: storage.get(key, default),
                                    store=lambda key, value: storage.__setitem__(key, value))
        ws = SimpleNamespace(par=SimpleNamespace(active=Par(True)), sendText=lambda value: messages.append(json.loads(value)))
        namespace = {'parent': lambda: component, 'op': lambda name: ws if name == 'websocket1' else None}
        exec(builder.WEBSOCKET_CODE, namespace)
        namespace['onReceiveText'](ws, 0, json.dumps({'type': 'welcome', 'ready': True, 'protocol': 1,
            'backend': 'ltx', 'supported_modes': ['ltx'], 'mode': 'ltx', 'width': 1024, 'height': 576,
            'resolution_locked': True, 'ltx_clip': 0, 'ltx_continuous': True, 'ltx_continue_scene': True}))
        self.assertEqual(pars.Mode.val, 'ltx')
        namespace['send_control'](ltx_generate=True)
        self.assertEqual(set(messages[-1]), {'type', 'mode', 'width', 'height', 'prompt_a', 'seed',
            'ltx_paused', 'ltx_generate', 'ltx_continuous', 'ltx_continue_scene', 'ltx_refresh_every'})
        self.assertTrue(messages[-1]['ltx_continuous'])
        self.assertTrue(messages[-1]['ltx_continue_scene'])
        self.assertTrue(messages[-1]['ltx_generate'])
        self.assertEqual((messages[-1]['width'], messages[-1]['height']), (1024, 576))
        self.assertIn('Clip 0', pars.Ltxinfo.val)

    def test_notebook_embeds_current_sources_and_cells_compile(self):
        book = notebook.make_notebook()
        for cell in book['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
        tree = ast.parse(''.join(book['cells'][2]['source']))
        sources = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'sources')
        for name in notebook.SOURCES:
            self.assertEqual(sources[name], (notebook.PROJECT_DIR / name).read_text(encoding='utf-8'))
        self.assertIn('userdata.get', ''.join(book['cells'][2]['source']))

    def test_source_hash_and_archive_containment_precede_execution(self):
        with tempfile.TemporaryDirectory(prefix='colab-sd-ltx-test-') as directory:
            root = Path(directory).resolve()
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                assets.unpack_source(b'untrusted', root)
            payload = io.BytesIO()
            with zipfile.ZipFile(payload, 'w') as archive:
                archive.writestr(f'LTX-2-{assets.CODE_REVISION}/../escape.py', 'bad')
            data = payload.getvalue()
            with patch.object(assets, 'CODE_SHA256', hashlib.sha256(data).hexdigest()):
                with self.assertRaisesRegex(ValueError, 'invalid path'):
                    assets.unpack_source(data, root)
            self.assertFalse(list(root.iterdir()))


class LTXTransportChecks(unittest.IsolatedAsyncioTestCase):
    async def test_delivery_overlaps_render_and_prompt_edits_do_not_wait_for_gpu(self):
        backend, ws = fake_backend(), FakeWebSocket()
        entered, resume = threading.Event(), threading.Event()

        def render(**kwargs):
            if kwargs['seed'] == 44:
                entered.set()
                if not resume.wait(10):
                    raise RuntimeError('test render gate timed out')
            pixels = np.full((17, 512, 512, 3), 255 if kwargs['prompt'] == 'a red car' else 0, dtype=np.uint8)
            return SimpleNamespace(video=iter([pixels]))

        backend.pipe.side_effect = render
        with patch.multiple(server, is_world_backend=False, is_krea_backend=False, is_ltx_backend=True,
            ltx_backend=backend, pipeline_ready=True, current_mode='ltx', output_width=512, output_height=512,
            active_client=None, access_token='', current_prompt_a='default', current_seed=42), \
                patch.object(server, 'get_gpu_telemetry', return_value=('--', '--')):
            task = asyncio.create_task(server.websocket_stream_endpoint(ws))
            async def send(data):
                await ws.incoming.put({'text': json.dumps(data)})
            try:
                await until(lambda: ws.text)
                await send({'type': 'client_options', 'frame_ack': True, 'frame_window': 1})
                await send({'type': 'control', 'prompt_a': 'forest', 'ltx_continuous': True, 'ltx_generate': True})
                for count in range(1, 18):
                    await until(lambda: len(ws.binary) >= count)
                    self.assertEqual(len(ws.binary), count)
                    await send({'type': 'frame_ack', 'frame_id': ws.binary[-1][0]})
                await until(entered.is_set)
                # Clip 3 is blocked, but clip 2 still delivers from the CPU queue.
                await until(lambda: len(ws.binary) >= 18)
                await send({'type': 'ping', 'sent_at': 123})
                await until(lambda: any(x.get('type') == 'pong' and x.get('sent_at') == 123 for x in ws.text))
                await send({'type': 'control', 'prompt_a': 'a red car', 'ltx_paused': True})
                await until(lambda: backend.prompt == 'a red car' and backend.paused)
                self.assertFalse(resume.is_set())
                self.assertEqual(backend.progress()['ltx_queued_frames'], 0)
                self.assertIsNone(backend._last_rgb)
                await send({'type': 'control', 'ltx_paused': True})
                resume.set()
                await until(lambda: backend.paused)
                self.assertEqual(backend.pipe.call_count, 3)
                await send({'type': 'frame_ack', 'frame_id': ws.binary[-1][0]})
                await send({'type': 'control', 'ltx_paused': False})
                await until(lambda: len(ws.binary) > 18)
                import cv2
                image = cv2.imdecode(np.frombuffer(ws.binary[-1][3:], np.uint8), cv2.IMREAD_COLOR)
                self.assertGreater(image.mean(), 250)
                self.assertEqual(backend.pipe.call_args.kwargs['prompt'], 'a red car')
            finally:
                resume.set()
                await ws.close()
                await asyncio.wait_for(task, 3)
            self.assertFalse(backend.can_generate)
            self.assertFalse(backend.running)
            self.assertEqual(len(backend.frames), 0)

    async def test_configuration_and_ack_gate_manual_clip_delivery(self):
        backend, ws = fake_backend(), FakeWebSocket()
        with patch.multiple(server, is_world_backend=False, is_krea_backend=False, is_ltx_backend=True,
            ltx_backend=backend, pipeline_ready=True, current_mode='ltx', output_width=512, output_height=512,
            active_client=None, access_token='', current_prompt_a='default', current_seed=42), \
                patch.object(server, 'get_gpu_telemetry', return_value=('--', '--')):
            task = asyncio.create_task(server.websocket_stream_endpoint(ws))
            async def send(data):
                await ws.incoming.put({'text': json.dumps(data)})
            try:
                await until(lambda: ws.text)
                self.assertEqual(ws.text[0]['supported_modes'], ['ltx'])
                await send({'type': 'client_options', 'frame_ack': True, 'frame_window': 1})
                await send({'type': 'control', 'prompt_a': 'forest'})
                await send({'type': 'ping', 'sent_at': 1})
                await until(lambda: any(x.get('type') == 'pong' for x in ws.text))
                backend.pipe.assert_not_called()
                await send({'type': 'control', 'ltx_generate': True})
                await until(lambda: ws.binary)
                await asyncio.sleep(.1)
                self.assertEqual(len({p[0] for p in ws.binary}), 1)
                await send({'type': 'control', 'ltx_paused': True})
                await until(lambda: backend.paused)
                await send({'type': 'frame_ack', 'frame_id': ws.binary[0][0]})
                await asyncio.sleep(.1)
                self.assertEqual(len({p[0] for p in ws.binary}), 1)
            finally:
                await ws.close()
                await asyncio.wait_for(task, 3)
            self.assertFalse(backend.can_generate)
            self.assertEqual(len(backend.frames), 0)
            self.assertIsNone(server.active_client)
