import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import build_clean_toe as builder
import colab_server as server
from colab_launcher import BridgeSession
import tensorrt_backend as backend
from test_streaming import FakeWebSocket, until, Par


class LatencyChecks(unittest.IsolatedAsyncioTestCase):
    async def test_render_acknowledgements_bound_the_video_queue_without_blocking_ping(self):
        ws = FakeWebSocket()
        with patch.multiple(server, pipeline_ready=True, current_mode='txt2img', enable_tensorrt=False), \
                patch.object(server, 'process_frame', return_value=(b'jpeg', 1, 2)) as process, \
                patch.object(server, 'get_gpu_telemetry', return_value=('--', '--')):
            task = asyncio.create_task(server.websocket_stream_endpoint(ws))
            try:
                await until(lambda: ws.text)
                await ws.incoming.put({'text': json.dumps({'type': 'client_options', 'frame_ack': True, 'frame_window': 1})})
                await until(lambda: ws.binary)
                # A pre-negotiation frame may already be sent. Find the final
                # outstanding frame, then prove the queue remains bounded.
                await asyncio.sleep(.12)
                count = len(ws.binary)
                await asyncio.sleep(.12)
                self.assertEqual(len(ws.binary), count)
                await ws.incoming.put({'text': '{"type":"ping","sent_at":42}'})
                await until(lambda: any(item.get('type') == 'pong' for item in ws.text))
                await ws.incoming.put({'text': json.dumps({'type': 'frame_ack', 'frame_id': ws.binary[-1][0]})})
                await until(lambda: len(ws.binary) > count)
                self.assertLessEqual(process.call_count, count + 1)
            finally:
                await ws.close()
                await asyncio.wait_for(task, 2)

    async def test_probe_remains_independent_of_a_slow_video_send(self):
        class SlowVideo(FakeWebSocket):
            async def send_bytes(self, data):
                await asyncio.sleep(.02)
                await super().send_bytes(data)
        video, probe = SlowVideo(), FakeWebSocket()
        with patch.multiple(server, pipeline_ready=True, current_mode='txt2img', access_token=''), \
                patch.object(server, 'process_frame', return_value=(b'x' * 48000, 1, 2)), \
                patch.object(server, 'get_gpu_telemetry', return_value=('--', '--')):
            stream = asyncio.create_task(server.websocket_stream_endpoint(video))
            ping = asyncio.create_task(server.websocket_ping_endpoint(probe))
            try:
                await until(lambda: video.binary)
                await probe.incoming.put({'text': '{"type":"ping","sent_at":123}'})
                await until(lambda: any(item.get('type') == 'pong' for item in probe.text), timeout=.2)
                self.assertLess(len(video.binary), 24)
                self.assertIs(server.active_client, video)
                await video.incoming.put({'text': '{"type":"ping","sent_at":456}'})
                await until(lambda: any(item.get('type') == 'pong' for item in video.text), timeout=.2)
                self.assertLess(len(video.binary), 24)
            finally:
                await video.close()
                await probe.close()
                await asyncio.wait_for(asyncio.gather(stream, ping), 2)

    async def test_locked_tensorrt_profile_rejects_resize_without_rebuilding(self):
        ws = FakeWebSocket()
        with patch.multiple(server, pipeline_ready=True, current_mode='img2img', enable_tensorrt=True,
                            allow_tensorrt_resize=False, output_width=768, output_height=576,
                            img2img_engine='TensorRT (UNet)', active_engine='TensorRT (UNet)'), \
                patch.object(server, 'load_pipeline') as load, \
                patch.object(server, 'get_gpu_telemetry', return_value=('--', '--')):
            task = asyncio.create_task(server.websocket_stream_endpoint(ws))
            try:
                await until(lambda: ws.text)
                self.assertTrue(ws.text[0]['resolution_locked'])
                await ws.incoming.put({'text': '{"type":"control","width":512,"height":512}'})
                await until(lambda: any(item.get('code') == 'resolution_locked' for item in ws.text))
                load.assert_not_called()
                error = next(item for item in ws.text if item.get('code') == 'resolution_locked')
                self.assertTrue(error['ready'])
                self.assertEqual(error['engine'], 'TensorRT (UNet)')
                self.assertEqual((error['width'], error['height']), (768, 576))
            finally:
                await ws.close()
                await asyncio.wait_for(task, 2)


class ProfileCacheChecks(unittest.TestCase):
    def test_weight_changes_and_hardware_changes_invalidate_archives(self):
        class Model(torch.nn.Module):
            dtype = torch.float32
            config = {'channels': 4, '_use_default_values': ['a', 'b']}
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(2))
                self.register_buffer('scalar', torch.tensor(1))
        model = Model()
        args = (model, 512, 512, 1, '2.11.0', '13.0', '2.11.0', '10.15.1', 'A100', (8, 0))
        original = backend.engine_metadata(*args)
        model.config['_use_default_values'] = ['b', 'a']
        self.assertEqual(original, backend.engine_metadata(*args))
        with torch.no_grad():
            model.weight.add_(1)
        self.assertNotEqual(original['weights'], backend.engine_metadata(*args)['weights'])
        self.assertNotEqual(original, backend.engine_metadata(*args[:-2], 'T4', (7, 5)))

    def test_bad_checksums_and_invalid_engine_outputs_are_never_used(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            archive = root / 'unet.ts'
            archive.write_bytes(b'fake archive')
            metadata = {'profile': 'test'}
            (root / 'manifest.json').write_text(json.dumps({'metadata': metadata, 'sha256': 'wrong'}))
            inputs, expected = [torch.ones(1)], torch.ones(1)
            with patch.object(torch.jit, 'load') as load, self.assertLogs(backend.__name__, level='WARNING'):
                self.assertIsNone(backend.load_engine_archive(root, metadata, inputs, expected))
                load.assert_not_called()
            (root / 'manifest.json').write_text(json.dumps({'metadata': metadata, 'sha256': backend.file_digest(archive)}))
            module = Mock(inlined_graph='tensorrt::execute_engine', return_value=torch.tensor([float('nan')]))
            module.eval.return_value = module
            with patch.object(torch.jit, 'load', return_value=module), self.assertLogs(backend.__name__, level='WARNING'):
                self.assertIsNone(backend.load_engine_archive(root, metadata, inputs, expected))

    def test_startup_size_and_cache_settings_are_passed_to_the_server(self):
        session = BridgeSession(output_size=(1024, 768), tunnel=False, engine_cache='profiles')
        with patch('colab_launcher.socket.create_connection', side_effect=ConnectionRefusedError), \
                patch.object(session, '_spawn') as spawn, patch.object(session, 'wait_for_ready'):
            session.start()
        environment = spawn.call_args.args[2]
        self.assertEqual(environment['BRIDGE_INITIAL_WIDTH'], '1024')
        self.assertEqual(environment['BRIDGE_INITIAL_HEIGHT'], '768')
        self.assertEqual(environment['BRIDGE_TRT_ALLOW_RESIZE'], '0')
        self.assertEqual(environment['BRIDGE_TRT_ENGINE_REUSE'], '1')


class ClientLatencyChecks(unittest.TestCase):
    def test_server_startup_profile_is_adopted_and_locked_before_controls_are_sent(self):
        storage = {}
        pars = SimpleNamespace(Mode=Par('txt2img'), Resolution=Par('512'), Framewindow=Par('3'), Status=Par(''))
        component = SimpleNamespace(par=pars, fetch=lambda key, default=None: storage.get(key, default),
                                    store=lambda key, value: storage.__setitem__(key, value))
        ws = SimpleNamespace(par=SimpleNamespace(active=Par(True)), sendText=Mock())
        namespace = {'parent': lambda: component, 'op': lambda name: ws if name == 'websocket1' else None}
        exec(builder.WEBSOCKET_CODE, namespace)
        namespace['onReceiveText'](ws, 0, json.dumps({'type': 'welcome', 'protocol': 1, 'ready': True,
                                                    'width': 768, 'height': 576, 'mode': 'txt2img',
                                                    'resolution_locked': True, 'frame_ack': True}))
        self.assertEqual(pars.Resolution.val, '768x576')
        messages = [json.loads(call.args[0]) for call in ws.sendText.call_args_list]
        control = next(message for message in messages if message['type'] == 'control')
        self.assertEqual((control['width'], control['height']), (768, 576))
        exec(builder.PAREXEC_CODE, namespace)
        pars.Resolution.val, pars.Resolution.name = '1024x768', 'Resolution'
        namespace['onValueChange'](pars.Resolution, '768x576')
        self.assertEqual(pars.Resolution.val, '768x576')

    def test_probe_rtt_is_measured_independently_of_stream_rtt(self):
        import time
        sent = time.perf_counter() - .03
        storage = {'probe_sent_at': sent, 'probe_waiting': True}
        pars = SimpleNamespace(Ping='', Streamrtt='2000 ms')
        component = SimpleNamespace(par=pars, fetch=lambda key, default=None: storage.get(key, default),
                                    store=lambda key, value: storage.__setitem__(key, value))
        namespace = {'parent': lambda: component}
        exec(builder.LATENCY_CODE, namespace)
        namespace['onReceiveText'](None, 0, json.dumps({'type': 'pong', 'sent_at': sent}))
        self.assertGreaterEqual(float(pars.Ping.split()[0]), 30)
        self.assertEqual(pars.Streamrtt, '2000 ms')
        self.assertFalse(storage['probe_waiting'])

    def test_received_frames_wait_for_display_before_granting_credit(self):
        storage = {'frame_ack_supported': True, 'handshake_complete': True}
        pars = SimpleNamespace(Mode=Par('txt2img'), Audiorx=Par(False))
        component = SimpleNamespace(par=pars, fetch=lambda key, default=None: storage.get(key, default),
                                    store=lambda key, value: storage.__setitem__(key, value))
        ws = SimpleNamespace(par=SimpleNamespace(active=Par(True)), sendText=Mock())
        receiver = SimpleNamespace(cook=Mock())
        lookup = lambda name: ws if name == 'websocket1' else receiver if name == 'script_receiver' else None
        namespace = {'parent': lambda: component, 'op': lookup}
        exec(builder.WEBSOCKET_CODE, namespace)
        namespace['onReceiveBinary'](ws, bytes([7, 0, 1]) + b'jpeg')
        ws.sendText.assert_not_called()
        exec(builder.EXEC_SEND_FRAME_CODE, namespace)
        namespace['onFrameStart'](1)
        receiver.cook.assert_called_once()
        sent = [json.loads(call.args[0]) for call in ws.sendText.call_args_list]
        self.assertIn({'type': 'frame_ack', 'frame_id': 7}, sent)

    def test_slow_output_frame_keeps_assembly_state_and_ack_credit(self):
        storage = {'frame_ack_supported': True}
        component = SimpleNamespace(par=SimpleNamespace(Mode=Par('txt2img')),
                                    fetch=lambda key, default=None: storage.get(key, default),
                                    store=lambda key, value: storage.__setitem__(key, value))
        namespace = {'parent': lambda: component}
        exec(builder.WEBSOCKET_CODE, namespace)
        with patch('time.perf_counter', return_value=10):
            namespace['onReceiveBinary'](None, bytes([7, 0, 2]) + b'first')
        with patch('time.perf_counter', return_value=12):
            namespace['onReceiveBinary'](None, bytes([7, 1, 2]) + b'last')
        self.assertEqual(storage['last_frame_bytes'], b'firstlast')
        self.assertEqual(storage['pending_frame_acks'], [7])
