"""World-model contract checks using a CPU fake, never downloading GPU packages/weights."""
import ast
import asyncio
import json
import os
import subprocess
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch

import build_clean_toe as builder
import colab_launcher
import colab_server as server
import generate_waypoint_notebook as notebook
from waypoint_backend import (WaypointBackend, WORLD_SIZE, validate_world_controls,
                              MODEL_REVISION, VAE_REVISION, engine_class_for_profile)
from test_streaming import FakeWebSocket, Par, until


def fake_backend():
    batch = torch.full((4, 720, 1280, 3), 100, dtype=torch.uint8)
    engine = SimpleNamespace(reset=Mock(), append_frame=Mock(), gen_frame=Mock(return_value=batch))
    return WaypointBackend(engine, lambda **kw: SimpleNamespace(**kw), torch)


def seed_jpeg():
    return cv2.imencode('.jpg', np.full((90, 160, 3), (0, 0, 255), np.uint8))[1].tobytes()


class WaypointChecks(unittest.TestCase):
    def test_invalid_world_patches_do_not_mutate_state(self):
        backend = fake_backend()
        for invalid in ({'world_buttons': [256]}, {'world_buttons': [True]},
                        {'world_mouse_x': float('nan')}, {'world_mouse_y': 1.01},
                        {'world_paused': 1}, {'world_reset': 'yes'}, {'mode': 'txt2img'},
                        {'width': 1024, 'height': 768}, {'seed': -1}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                backend.configure({'world_buttons': [87], **invalid})
            self.assertEqual(backend.buttons, [])
        self.assertEqual(validate_world_controls({'world_buttons': [87, 87, 65]}),
                         {'world_buttons': [65, 87]})

    def test_seed_and_batch_queue_preserve_native_pixels_and_are_bounded(self):
        backend = fake_backend()
        self.assertIsNone(backend.next_frame())
        backend.set_seed_image(seed_jpeg())
        seed = backend.engine.append_frame.call_args.args[0]
        self.assertEqual(tuple(seed.shape), (4, 720, 1280, 3))
        self.assertEqual(seed.dtype, torch.uint8)
        self.assertGreater(int(seed[0, 0, 0, 0]), 240)
        backend.configure({'world_buttons': [87], 'world_mouse_x': 0.2})
        for remaining in (3, 2, 1, 0):
            image, batch_ms = backend.next_frame()
            self.assertEqual(image.shape, (720, 1280, 3))
            self.assertEqual(len(backend.frames), remaining)
        self.assertEqual(backend.engine.gen_frame.call_count, 1)
        ctrl = backend.engine.gen_frame.call_args.kwargs['ctrl']
        self.assertEqual(ctrl.button, {87})
        self.assertEqual(ctrl.mouse, (0.2, 0.0))
        backend.next_frame()
        self.assertEqual(backend.engine.gen_frame.call_count, 2)

    def test_pause_reset_and_disconnect_reuse_weights_and_clear_pending_frames(self):
        backend = fake_backend()
        backend.set_seed_image(seed_jpeg())
        backend.next_frame()
        backend.configure({'world_paused': True})
        self.assertIsNone(backend.next_frame())
        backend.configure({'world_reset': True, 'seed': 7})
        self.assertEqual(len(backend.frames), 0)
        self.assertEqual(backend.engine.append_frame.call_count, 2)
        backend.configure({'world_paused': False, 'world_buttons': [87]})
        backend.next_frame()
        backend.release_controls()
        self.assertEqual(backend.buttons, [])
        self.assertEqual(len(backend.frames), 0)

    def test_failed_reset_cannot_continue_with_partial_world_state(self):
        backend = fake_backend()
        backend.engine.append_frame.side_effect = RuntimeError('failed')
        with self.assertRaises(RuntimeError):
            backend.set_seed_image(seed_jpeg())
        self.assertFalse(backend.can_generate)

    def test_loader_pins_both_models_and_selects_a100_compatible_quantization(self):
        world = SimpleNamespace(WorldEngine=Mock(), CtrlInput=Mock())
        with patch.dict('sys.modules', {'world_engine': world}), \
                patch('waypoint_backend.engine_class_for_profile', return_value=world.WorldEngine), \
                patch('huggingface_hub.snapshot_download', side_effect=['local/model', 'local/vae']) as download, \
                patch.object(torch.cuda, 'is_available', return_value=True), \
                patch.object(torch.cuda, 'get_device_capability', return_value=(8, 0)):
            WaypointBackend.load('intw8a8')
            self.assertEqual([call.kwargs['revision'] for call in download.call_args_list],
                             [MODEL_REVISION, VAE_REVISION])
            self.assertEqual(world.WorldEngine.call_args.kwargs['quant'], 'intw8a8')
            self.assertEqual(world.WorldEngine.call_args.kwargs['model_config_overrides'], {'ae_uri': 'local/vae'})
            with self.assertRaises(ValueError):
                WaypointBackend.load('fp8w8a8')

    def test_launcher_isolates_backend_and_disallows_sd_tensorrt(self):
        session = colab_launcher.BridgeSession(backend='waypoint')
        self.assertEqual(session.output_size, WORLD_SIZE)
        with self.assertRaises(ValueError):
            colab_launcher.BridgeSession(backend='waypoint', tensorrt=True)

    def test_fresh_server_selects_world_backend_without_sd_resolution_validation(self):
        environment = {**os.environ, 'BRIDGE_BACKEND': 'waypoint', 'ENABLE_TENSORRT': '0',
                       'BRIDGE_INITIAL_WIDTH': '1280', 'BRIDGE_INITIAL_HEIGHT': '720',
                       'PYTHONDONTWRITEBYTECODE': '1', 'WAYPOINT_QUANT': 'intw8a8'}
        result = subprocess.run([sys.executable, '-c',
            'import json, colab_server as s; print(s.health().body.decode())'],
            cwd=notebook.PROJECT_DIR, env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        status = json.loads(result.stdout)
        self.assertEqual((status['backend'], status['mode'], status['width'], status['height']),
                         ('waypoint', 'world', 1280, 720))
        self.assertFalse(status['ready'])  # importing is not a loaded/warmed GPU model

    def test_notebook_sources_match_and_cells_compile(self):
        book = notebook.make_notebook()
        for cell in book['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
        setup = ast.parse(''.join(book['cells'][2]['source']))
        sources = next(ast.literal_eval(node.value) for node in setup.body
                       if isinstance(node, ast.Assign) and node.targets[0].id == 'sources')
        for name in notebook.SOURCES:
            self.assertEqual(sources[name], (notebook.PROJECT_DIR / name).read_text(encoding='utf-8'))
        self.assertIn('backend=\'waypoint\'', ''.join(book['cells'][3]['source']))

    def test_fast_profile_recompiles_original_functions_without_mutating_upstream(self):
        class Engine:
            def _cache_pass(self, value):
                return value + 1
            def _denoise_pass(self, value):
                return value * 2
        original = Engine._cache_pass
        compiler = SimpleNamespace(compile=Mock(side_effect=lambda function, **kw: function))
        fast = engine_class_for_profile(Engine, compiler, 'fast')
        self.assertEqual((fast()._cache_pass(2), fast()._denoise_pass(2)), (3, 4))
        self.assertIs(Engine._cache_pass, original)
        for call in compiler.compile.call_args_list:
            self.assertTrue(call.kwargs['fullgraph'])
            self.assertFalse(call.kwargs['options']['max_autotune'])
            self.assertFalse(call.kwargs['options']['triton.cudagraphs'])
        compiler.compile.reset_mock()
        self.assertIs(engine_class_for_profile(Engine, compiler, 'max-autotune'), Engine)
        compiler.compile.assert_not_called()
        with self.assertRaises(ValueError):
            engine_class_for_profile(Engine, compiler, 'unknown')

    def test_world_progress_distinguishes_seed_preparation_first_batch_and_pause(self):
        backend = fake_backend()
        self.assertEqual(backend.progress()['world_phase'], 'awaiting_seed')
        phases = []
        backend.engine.append_frame.side_effect = lambda image: phases.append(backend.progress()['world_phase'])
        backend.set_seed_image(seed_jpeg())
        self.assertEqual(phases, ['seeding'])
        self.assertEqual(backend.progress()['world_phase'], 'ready')
        batch = backend.engine.gen_frame.return_value
        backend.engine.gen_frame.side_effect = lambda **kw: (phases.append(backend.progress()['world_phase']) or batch)
        backend.next_frame()
        self.assertEqual(phases[-1], 'first_batch')
        self.assertEqual(backend.progress()['world_phase'], 'streaming')
        backend.configure({'world_paused': True})
        self.assertEqual(backend.progress()['world_phase'], 'paused')

    def test_td_idle_display_preserves_world_progress(self):
        storage = {'connected': True, 'server_ready': True, 'last_frame_receive': 0,
                   'world_phase': 'first_batch', 'world_status': 'Generating first world batch (42s)'}
        pars = SimpleNamespace(Mode=Par('world'), Status=Par(''), Fps=Par(''))
        comp = SimpleNamespace(par=pars, fetch=lambda key, default=None: storage.get(key, default),
                               store=lambda key, value: storage.__setitem__(key, value))
        ns = {'parent': lambda: comp, 'op': lambda name: None}
        exec(builder.EXEC_SEND_FRAME_CODE, ns)
        ns['onFrameStart'](1)
        self.assertEqual(pars.Status.val, 'Generating first world batch (42s)')

    def test_td_world_handshake_selects_native_mode_and_sends_supported_controls(self):
        pars = SimpleNamespace(Mode=Par('txt2img'), Resolution=Par('512'), Status=Par(''),
                               Worldbuttons=Par('87,65'), Worldmousex=Par(.2), Worldmousey=Par(0),
                               Worldpaused=Par(False), Seed=Par(42), Connect=Par(True))
        storage = {}
        comp = SimpleNamespace(par=pars, fetch=lambda key, default=None: storage.get(key, default),
                               store=lambda key, value: storage.__setitem__(key, value))
        messages = []
        ws = SimpleNamespace(par=SimpleNamespace(active=Par(True)),
                             sendText=lambda data: messages.append(json.loads(data)))
        ns = {'parent': lambda: comp, 'op': lambda name: ws if name == 'websocket1' else None}
        exec(builder.WEBSOCKET_CODE, ns)
        ns['onReceiveText'](ws, 0, json.dumps({'type': 'welcome', 'protocol': 1, 'ready': True,
             'backend': 'waypoint', 'supported_modes': ['world'], 'mode': 'world',
             'width': 1280, 'height': 720, 'resolution_locked': True}))
        control = messages[-1]
        self.assertEqual(control['mode'], 'world')
        self.assertEqual(control['world_buttons'], [87, 65])
        self.assertNotIn('prompt_a', control)
        self.assertTrue(storage['server_ready'])
        self.assertEqual(storage['requested_configuration'], ('world', 1280, 720))


class WorldTransportChecks(unittest.IsolatedAsyncioTestCase):
    async def test_progress_and_pings_arrive_while_first_seed_blocks_gpu_worker(self):
        backend, ws = fake_backend(), FakeWebSocket()
        started, release = threading.Event(), threading.Event()
        def prepare(image):
            started.set()
            if not release.wait(5):
                raise RuntimeError('test seed worker timed out')
        backend.engine.append_frame.side_effect = prepare
        with patch.multiple(server, is_world_backend=True, world_backend=backend,
                            pipeline_ready=True, current_mode='world', output_width=1280,
                            output_height=720, active_client=None, access_token=''), \
                patch.object(server, 'get_gpu_telemetry', return_value=('--', '--')):
            task = asyncio.create_task(server.websocket_stream_endpoint(ws))
            try:
                await until(lambda: ws.text)
                await ws.incoming.put({'text': json.dumps({'type': 'client_options', 'frame_ack': True, 'frame_window': 1})})
                await ws.incoming.put({'text': json.dumps({'type': 'world_seed'})})
                await ws.incoming.put({'bytes': seed_jpeg()})
                await until(started.is_set)
                await ws.incoming.put({'text': json.dumps({'type': 'ping', 'sent_at': 123})})
                await until(lambda: any(row.get('type') == 'pong' for row in ws.text))
                await until(lambda: any(row.get('type') == 'stats' and row.get('world_phase') == 'seeding'
                                        for row in ws.text), timeout=3)
                self.assertEqual(ws.binary, [])
            finally:
                release.set()
                await ws.close()
                await asyncio.wait_for(task, 3)

    async def test_existing_world_waits_for_credit_negotiation_on_reconnect(self):
        backend = fake_backend()
        backend.set_seed_image(seed_jpeg())
        ws = FakeWebSocket()
        with patch.multiple(server, is_world_backend=True, world_backend=backend,
                            pipeline_ready=True, current_mode='world', output_width=1280,
                            output_height=720, active_client=None, access_token=''), \
                patch.object(server, 'get_gpu_telemetry', return_value=('--', '--')):
            task = asyncio.create_task(server.websocket_stream_endpoint(ws))
            try:
                await until(lambda: ws.text)
                await ws.incoming.put({'text': json.dumps({'type': 'ping', 'sent_at': 1})})
                await until(lambda: any(row.get('type') == 'pong' for row in ws.text))
                await asyncio.sleep(.05)
                self.assertEqual(ws.binary, [])
                backend.engine.gen_frame.assert_not_called()
                await ws.incoming.put({'text': json.dumps({'type': 'client_options', 'frame_ack': True,
                                                          'frame_window': 1})})
                await until(lambda: ws.binary)
            finally:
                await ws.close()
                await asyncio.wait_for(task, 3)

    async def test_seed_gate_credit_window_ping_and_reconnect_cleanup(self):
        backend = fake_backend()
        ws = FakeWebSocket()
        with patch.multiple(server, is_world_backend=True, world_backend=backend,
                            pipeline_ready=True, current_mode='world', output_width=1280,
                            output_height=720, active_client=None, access_token=''), \
                patch.object(server, 'get_gpu_telemetry', return_value=('--', '--')):
            task = asyncio.create_task(server.websocket_stream_endpoint(ws))
            try:
                await until(lambda: ws.text)
                self.assertEqual(ws.text[0]['supported_modes'], ['world'])
                await ws.incoming.put({'bytes': seed_jpeg()})  # unsolicited upload is ignored
                await ws.incoming.put({'text': json.dumps({'type': 'ping', 'sent_at': 1})})
                await until(lambda: any(row.get('type') == 'pong' for row in ws.text))
                self.assertEqual(backend.engine.append_frame.call_count, 0)
                await ws.incoming.put({'text': json.dumps({'type': 'client_options', 'frame_ack': True,
                                                          'frame_window': 1})})
                await ws.incoming.put({'text': json.dumps({'type': 'world_seed'})})
                await ws.incoming.put({'bytes': seed_jpeg()})
                await until(lambda: ws.binary)
                await asyncio.sleep(.1)
                ids = {chunk[0] for chunk in ws.binary}
                self.assertEqual(len(ids), 1)
                self.assertEqual(backend.engine.gen_frame.call_count, 1)
                await ws.incoming.put({'text': json.dumps({'type': 'frame_ack', 'frame_id': next(iter(ids))})})
                await until(lambda: len({chunk[0] for chunk in ws.binary}) == 2)
                self.assertEqual(backend.engine.gen_frame.call_count, 1)  # next cached frame, no extra batch
            finally:
                await ws.close()
                await asyncio.wait_for(task, 3)
            self.assertIsNone(server.active_client)
            self.assertEqual(len(backend.frames), 0)
