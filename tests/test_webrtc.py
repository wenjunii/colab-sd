"""Real CPU VP8 transport plus bridge negotiation, fallback and TD signaling contracts."""
import asyncio
import io
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
import numpy as np

import build_clean_toe as builder
import colab_server as server
import webrtc_transport as rtc
from test_streaming import FakeWebSocket, Par, until


class IceChecks(unittest.TestCase):
    def test_all_backend_raw_outputs_bypass_jpeg_encoding(self):
        image = np.zeros((768, 1024, 3), dtype=np.uint8)
        fake = SimpleNamespace(next_frame=lambda: (image, 12), pop_frame=lambda: (image, 12))
        for backend, mode in (('sd', 'txt2img'), ('sd', 'img2img'),
                              ('waypoint', 'world'), ('krea', 'krea'), ('ltx', 'ltx')):
            with self.subTest(backend=backend, mode=mode), patch.multiple(server,
                    is_world_backend=backend == 'waypoint', is_krea_backend=backend == 'krea',
                    is_ltx_backend=backend == 'ltx', world_backend=fake, krea_backend=fake, ltx_backend=fake), \
                    patch.object(server, 'run_model_inference', return_value=image), \
                    patch.object(server, 'compress_frame', side_effect=AssertionError('JPEG used in raw path')):
                pixels, _, _ = server.process_frame(mode, raw_output=True)
                self.assertEqual(pixels.shape, (768, 1024, 3))

    def test_explicit_servers_are_validated_and_secrets_are_not_required_for_lan(self):
        servers = [{'urls': ['turn:relay.example:3478?transport=udp'], 'username': 'user', 'credential': 'private'}]
        with patch.dict(os.environ, {'BRIDGE_ICE_SERVERS_JSON': json.dumps(servers)}, clear=True):
            self.assertEqual(rtc.ice_servers(), servers)
        for invalid in ({}, [{'urls': 'https://wrong.example'}], [{'urls': 'turn:user:pass@relay'}],
                        [{'urls': ['stun:bad\nurl']}], [{'urls': 'turn:relay', 'credential': []}]):
            with self.assertRaises(ValueError):
                rtc.validate_ice_servers(invalid)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(rtc.ice_servers(), [{'urls': ['stun:stun.l.google.com:19302']}])

    def test_cloudflare_key_is_used_only_at_the_fixed_provider_endpoint(self):
        response = io.BytesIO(json.dumps({'iceServers': {'urls': ['turn:relay:3478'],
            'username': 'short-user', 'credential': 'short-secret'}}).encode())
        with patch.dict(os.environ, {'CLOUDFLARE_TURN_KEY_ID': 'key-123',
                                     'CLOUDFLARE_TURN_API_TOKEN': 'long-secret'}, clear=True), \
                patch.object(rtc.urllib.request, 'urlopen', return_value=response) as request:
            result = rtc.ice_servers()
        sent = request.call_args.args[0]
        self.assertTrue(sent.full_url.startswith('https://rtc.live.cloudflare.com/'))
        self.assertNotIn('long-secret', json.dumps(result))
        self.assertEqual(result[0]['credential'], 'short-secret')


class RealWebRTCChecks(unittest.IsolatedAsyncioTestCase):
    async def test_latest_frame_queue_replaces_stale_frames_and_has_monotonic_timestamps(self):
        track = rtc.make_track()
        try:
            for value in range(12):
                track.put(np.full((48, 64, 3), value, dtype=np.uint8))
            self.assertEqual(track.queue.qsize(), 1)
            self.assertEqual(track.dropped, 11)
            first = await track.recv()
            self.assertEqual(first.to_ndarray(format='bgr24')[0, 0, 0], 11)
            track.put(np.zeros((48, 64, 3), np.uint8))
            second = await track.recv()
            self.assertGreater(second.pts, first.pts)
            waiter = asyncio.create_task(track.recv())
            await asyncio.sleep(0)
            track.stop()
            with self.assertRaises(Exception):
                await waiter
        finally:
            track.stop()

    async def test_real_bridge_negotiates_vp8_full_resolution_and_falls_back_without_model_reload(self):
        ws = FakeWebSocket()
        receiver = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        received = asyncio.Queue()

        @receiver.on('track')
        def on_track(track):
            received.put_nowait(track)

        image = np.zeros((768, 1024, 3), np.uint8)
        image[:384, :, 2] = 220
        image[384:, :, 1] = 220
        def process(mode, frame=None, budget=None, raw_output=False):
            return (image if raw_output else b'jpeg', 10, 11)

        with patch.object(rtc, 'ice_servers', return_value=[]), \
                patch.multiple(server, pipeline_ready=True, current_mode='txt2img',
                               is_world_backend=False, is_krea_backend=False, is_ltx_backend=False,
                               output_width=1024, output_height=768, active_client=None), \
                patch.object(server, 'process_frame', side_effect=process) as frames, \
                patch.object(server, 'apply_controls') as controls, \
                patch.object(server, 'load_pipeline') as load, \
                patch.object(server, 'get_gpu_telemetry', return_value=('--', '--')):
            task = asyncio.create_task(server.websocket_stream_endpoint(ws))
            try:
                await until(lambda: ws.text)
                self.assertIn('webrtc', ws.text[0]['video_transports'])
                await ws.incoming.put({'text': json.dumps({'type': 'client_options', 'frame_ack': True,
                    'frame_window': 1, 'video_transport': 'webrtc'})})
                await until(lambda: any(d['type'] == 'rtc_offer' for d in ws.text), timeout=10)
                offer = next(d for d in ws.text if d['type'] == 'rtc_offer')
                self.assertIn('VP8/90000', offer['sdp'])
                self.assertNotIn('H264/90000', offer['sdp'])
                await receiver.setRemoteDescription(RTCSessionDescription(offer['sdp'], 'offer'))
                await receiver.setLocalDescription(await receiver.createAnswer())
                await ws.incoming.put({'text': json.dumps({'type': 'rtc_answer', 'session': offer['session'],
                    'sdp': receiver.localDescription.sdp})})
                track = await asyncio.wait_for(received.get(), 10)
                frame = await asyncio.wait_for(track.recv(), 10)
                self.assertEqual((frame.width, frame.height), (1024, 768))
                pixels = frame.to_ndarray(format='bgr24')
                self.assertGreater(pixels[:300, :, 2].mean(), 190)
                self.assertGreater(pixels[450:, :, 1].mean(), 190)
                await ws.incoming.put({'text': '{"type":"control","prompt_a":"new scene","seed":71}'})
                await until(lambda: controls.called)
                self.assertEqual(controls.call_args.args[0]['prompt_a'], 'new scene')
                previous = len(ws.binary)
                await ws.incoming.put({'text': json.dumps({'type': 'rtc_stop', 'session': offer['session']})})
                await until(lambda: any(d['type'] == 'rtc_closed' for d in ws.text))
                await until(lambda: len(ws.binary) > previous)
                self.assertTrue(any(c.kwargs.get('raw_output') for c in frames.call_args_list))
                load.assert_not_called()
            finally:
                await ws.close()
                await asyncio.wait_for(task, 10)
                await receiver.close()

    async def test_failed_setup_redacts_credentials_and_preserves_fallback(self):
        sent = []
        async def send(data):
            sent.append(data)
        output = rtc.WebRTCOutput(send)
        with patch.object(rtc, 'ice_servers', side_effect=ValueError('secret URL/password')):
            output.start()
            await output.task
        self.assertTrue(output.closed)
        self.assertFalse(output.connected)
        self.assertNotIn('secret URL/password', json.dumps(sent))
        self.assertEqual(sent[-1]['type'], 'rtc_closed')


class NativeSignalingChecks(unittest.TestCase):
    def make_client(self):
        storage = {'connected': True}
        component = SimpleNamespace(par=SimpleNamespace(Transportstate=Par(''), Fps=Par(''), Status=Par('')),
            fetch=lambda k, d=None: storage.get(k, d), store=lambda k, v: storage.__setitem__(k, v))
        peer = SimpleNamespace(par=SimpleNamespace(stun=Par(''), active=Par(False)),
            openConnection=Mock(return_value='native-id'), closeConnection=Mock(),
            setRemoteDescription=Mock(), setLocalDescription=Mock(), createAnswer=Mock(), getStats=Mock())
        video = SimpleNamespace(par=SimpleNamespace(active=Par(False)))
        socket = SimpleNamespace(sendText=Mock())
        nodes = {'webrtc1': peer, 'webrtc_video': video, 'websocket1': socket}
        namespace = {'parent': lambda: component, 'op': lambda name: nodes.get(name)}
        exec(builder.WEBRTC_CODE, namespace)
        return storage, namespace, peer, video, socket

    def test_offer_answer_tracks_and_late_callbacks_are_scoped_to_the_current_session(self):
        storage, code, peer, video, socket = self.make_client()
        code['handle']({'type': 'rtc_offer', 'session': 'first', 'sdp': 'offer', 'ice_servers': []})
        peer.setRemoteDescription.assert_called_once_with('native-id', 'offer', 'offer')
        code['onAnswer'](peer, 'native-id', 'answer')
        self.assertEqual(json.loads(socket.sendText.call_args.args[0])['session'], 'first')
        code['onTrack'](peer, 'native-id', 'track-id', 'video')
        self.assertEqual(video.par.webrtctrack, 'track-id')
        code['reset']()
        code['onConnectionStateChange'](peer, 'native-id', 'failed')
        self.assertFalse(storage['rtc_video_active'])
        self.assertIsNone(storage['rtc_connection'])

    def test_only_decoded_frames_activate_native_video_and_stats_use_decoded_fps(self):
        storage, code, peer, video, socket = self.make_client()
        storage.update(rtc_connection='id', rtc_session='session', rtc_fps_time=0, server_ready=True)
        peer.getStats.return_value = json.dumps({'report': {'type': 'inbound-rtp', 'kind': 'video', 'framesDecoded': 8}})
        code['tick'](1)
        peer.getStats.assert_called_once_with(connectionId='id')
        self.assertTrue(storage['rtc_video_active'])
        self.assertEqual(storage['rtc_decoded'], 8)
        self.assertEqual(storage['last_frame_receive'], 1)
        self.assertEqual(code['decoded_count']([{'type': 'outbound-rtp', 'framesDecoded': 100}]), 0)

    def test_late_offer_cannot_override_a_manual_websocket_selection(self):
        storage, code, peer, video, socket = self.make_client()
        component = code['parent']()
        component.par.Videotransport = Par('websocket')
        code['handle']({'type': 'rtc_offer', 'session': 'late', 'sdp': 'offer'})
        peer.openConnection.assert_not_called()
        self.assertEqual(json.loads(socket.sendText.call_args.args[0]), {'type': 'rtc_stop', 'session': 'late'})
