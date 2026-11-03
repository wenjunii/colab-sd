import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

import bridge_protocol as protocol
import build_clean_toe as builder
import colab_server as server
from test_streaming import FakeWebSocket, Par, until


class VideoCompressionChecks(unittest.TestCase):
    def test_transport_budgets_are_independent_of_output_dimensions(self):
        for size in protocol.OUTPUT_RESOLUTIONS:
            fast = protocol.video_byte_budget(*size, 'fast')
            balanced = protocol.video_byte_budget(*size, 'balanced')
            detail = protocol.video_byte_budget(*size, 'detail')
            self.assertLessEqual(fast, balanced)
            self.assertLessEqual(balanced, detail)
        self.assertEqual(protocol.video_byte_budget(1024, 768, 'fast'), 48000)
        self.assertEqual(protocol.video_byte_budget(1024, 768, 'balanced'), 64000)
        self.assertEqual(protocol.video_byte_budget(1024, 768, 'detail'), 144000)
        with self.assertRaises(ValueError):
            protocol.video_byte_budget(1024, 768, 'invalid')

    def test_fast_jpeg_reduces_traffic_preserves_resolution_and_reuses_encoding_quality(self):
        image = np.random.default_rng(42).integers(0, 256, (768, 1024, 3), dtype=np.uint8)
        with patch.object(server, 'jpeg_quality_hints', {}):
            detail = server.compress_frame(image, 144000)
            with patch.object(server.cv2, 'imencode', wraps=cv2.imencode) as encode:
                fast = server.compress_frame(image, 48000)
                self.assertGreater(encode.call_count, 1)
                encode.reset_mock()
                again = server.compress_frame(image, 48000)
                self.assertEqual(encode.call_count, 1)
            self.assertEqual(fast, again)
            self.assertLessEqual(len(fast), 48000)
            self.assertLess(len(fast), len(detail))
            decoded = cv2.imdecode(np.frombuffer(fast, np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(decoded.shape, image.shape)
            self.assertGreater(decoded.std(), 2)

    def test_high_entropy_frames_keep_streaming_at_full_size_when_target_is_unreachable(self):
        gray = np.random.default_rng(42).integers(0, 2, (768, 1024), dtype=np.uint8) * 255
        image = np.repeat(gray[:, :, None], 3, axis=2)
        with patch.object(server, 'jpeg_quality_hints', {}), self.assertLogs(server.logger, level='WARNING'):
            encoded = server.compress_frame(image, 48000, allow_oversize=True)
        self.assertGreater(len(encoded), 48000)
        self.assertLessEqual(len(encoded), 255 * protocol.OUTPUT_CHUNK_BYTES)
        self.assertEqual(cv2.imdecode(np.frombuffer(encoded, np.uint8), 1).shape, image.shape)


class DeliveryProtocolChecks(unittest.IsolatedAsyncioTestCase):
    async def test_quality_switches_and_display_acknowledgements_do_not_reload_models(self):
        ws = FakeWebSocket()
        with patch.multiple(server, pipeline_ready=True, current_mode='img2img', enable_tensorrt=True,
                            output_width=1024, output_height=768), \
                patch.object(server, 'process_frame', return_value=(b'jpeg', 61, 67)) as process, \
                patch.object(server, 'load_pipeline') as load, \
                patch.object(server, 'get_gpu_telemetry', return_value=('--', '--')):
            task = asyncio.create_task(server.websocket_stream_endpoint(ws))
            try:
                await until(lambda: ws.text)
                self.assertIn('fast', ws.text[0]['video_quality_profiles'])
                for quality, budget in [('fast', 48000), ('balanced', 64000), ('detail', 144000)]:
                    count = len(ws.binary)
                    await ws.incoming.put({'text': json.dumps({'type': 'client_options', 'frame_ack': True,
                                                              'frame_window': 1, 'video_quality': quality})})
                    await ws.incoming.put({'bytes': b'\xff\xd8test frame'})
                    await until(lambda: len(ws.binary) > count)
                    self.assertEqual(process.call_args.args[-1], budget)
                    await asyncio.sleep(.02)
                    await ws.incoming.put({'text': json.dumps({'type': 'frame_ack', 'frame_id': ws.binary[-1][0],
                                                              'display_ms': 4.2, 'client_frame_ms': 16.7})})
                await until(lambda: any(item.get('delivery_ms') is not None for item in ws.text))
                stats = next(item for item in reversed(ws.text) if item.get('delivery_ms') is not None)
                self.assertEqual(stats['frame_bytes'], 4)
                self.assertGreaterEqual(stats['delivery_ms'], 20)
                self.assertEqual(stats['display_ms'], 4.2)
                self.assertEqual(stats['client_frame_ms'], 16.7)
                self.assertEqual(stats['video_quality'], 'detail')
                await ws.incoming.put({'text': '{"type":"client_options","video_quality":[]}'})
                await until(lambda: any(item.get('code') == 'invalid_message' for item in ws.text))
                load.assert_not_called()
            finally:
                await ws.close()
                await asyncio.wait_for(task, 2)


class VideoClientChecks(unittest.TestCase):
    def test_quality_and_buffer_settings_are_sent_without_generation_controls(self):
        storage = {'frame_ack_supported': True}
        pars = SimpleNamespace(Framewindow=Par('6'), Videoquality=Par('fast'))
        component = SimpleNamespace(par=pars, fetch=lambda key, default=None: storage.get(key, default))
        ws = SimpleNamespace(sendText=Mock())
        namespace = {'parent': lambda: component, 'op': lambda name: ws}
        exec(builder.WEBSOCKET_CODE, namespace)
        namespace['send_client_options']()
        self.assertEqual(json.loads(ws.sendText.call_args.args[0]),
                         {'type': 'client_options', 'frame_ack': True, 'frame_window': 6, 'video_quality': 'fast'})

    def test_delivery_readouts_show_frame_size_rtt_and_client_cook_cost(self):
        pars = SimpleNamespace(Framesize=Par(''), Deliverytime=Par(''), Cooktime=Par(''), Streamwait=Par(''))
        component = SimpleNamespace(par=pars)
        namespace = {'parent': lambda: component}
        exec(builder.WEBSOCKET_CODE, namespace)
        namespace['onReceiveText'](None, 0, json.dumps({'type': 'stats', 'frame_bytes': 42000,
                                                      'delivery_ms': 250, 'display_ms': 3,
                                                      'credit_wait_ms': 40}))
        self.assertEqual(pars.Framesize.val, '42.0 KB')
        self.assertEqual(pars.Deliverytime.val, '250.0 ms')
        self.assertEqual(pars.Cooktime.val, '3.0 ms')
        self.assertEqual(pars.Streamwait.val, '40.0 ms')
