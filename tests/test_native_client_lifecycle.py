import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import build_clean_toe as builder
from test_streaming import Par


class NativeLifecycleChecks(unittest.TestCase):
    def make_client(self):
        storage = {}
        pars = SimpleNamespace(Mode=Par('txt2img'), Resolution=Par('512'), Status=Par(''),
                               Framewindow=Par('3'), Videoquality=Par('balanced'), Engine=Par(''))
        component = SimpleNamespace(par=pars, fetch=lambda key, default=None: storage.get(key, default),
                                    store=lambda key, value: storage.__setitem__(key, value))
        ws = SimpleNamespace(par=SimpleNamespace(active=Par(True)), sendText=Mock())
        namespace = {'parent': lambda: component, 'op': lambda name: ws if name == 'websocket1' else None}
        exec(builder.WEBSOCKET_CODE, namespace)
        return storage, component, ws, namespace

    def welcome(self, namespace, ws):
        namespace['onReceiveText'](ws, 0, json.dumps({'type': 'welcome', 'protocol': 1, 'ready': True,
                                                    'width': 1024, 'height': 768, 'mode': 'txt2img',
                                                    'frame_ack': True, 'resolution_locked': True}))
        namespace['onReceiveText'](ws, 0, json.dumps({'type': 'configured', 'ready': True,
                                                    'width': 1024, 'height': 768, 'mode': 'txt2img'}))

    def test_native_welcome_before_connect_preserves_handshake_and_pending_acknowledgements(self):
        storage, component, ws, namespace = self.make_client()
        self.welcome(namespace, ws)
        namespace['onReceiveBinary'](ws, bytes([9, 0, 1]) + b'jpeg')
        namespace['onConnect'](ws)
        for key in ('connected', 'server_ready', 'handshake_complete', 'frame_ack_supported', 'display_pending'):
            self.assertTrue(storage[key], key)
        self.assertEqual(storage['pending_frame_acks'], [9])
        self.assertEqual(component.par.Resolution.val, '1024x768')

    def test_connect_before_welcome_and_disconnect_reset_remain_supported(self):
        storage, component, ws, namespace = self.make_client()
        namespace['onConnect'](ws)
        self.assertFalse(storage['handshake_complete'])
        self.welcome(namespace, ws)
        self.assertTrue(storage['handshake_complete'])
        namespace['onDisconnect'](ws)
        for key in ('connected', 'server_ready', 'handshake_complete', 'frame_ack_supported', 'display_pending'):
            self.assertFalse(storage[key], key)
        self.assertEqual(storage['pending_frame_acks'], [])

    def test_probe_ready_before_connect_is_not_overwritten(self):
        storage, component, ws, _ = self.make_client()
        component.par.Ping = '-- ms'
        namespace = {'parent': lambda: component}
        exec(builder.LATENCY_CODE, namespace)
        namespace['onReceiveText'](ws, 0, '{"type":"probe_ready"}')
        storage['probe_waiting'] = True
        namespace['onConnect'](ws)
        self.assertTrue(storage['probe_ready'])
        self.assertTrue(storage['probe_waiting'])
        namespace['onDisconnect'](ws)
        self.assertFalse(storage['probe_ready'])
        self.assertFalse(storage['probe_waiting'])

    def test_native_dat_address_separates_port_and_preserves_path_and_query(self):
        _, component, _, _ = self.make_client()
        namespace = {'parent': lambda: component}
        exec(builder.PAREXEC_CODE, namespace)
        cases = [('ws://127.0.0.1:8765/ws/stream', 'ws://127.0.0.1/ws/stream', 8765),
                 ('wss://host.example:8443/prefix/ws/stream?key=one',
                  'wss://host.example/prefix/ws/stream?key=one', 8443),
                 ('ws://[::1]:8765/ws/stream', 'ws://[::1]/ws/stream', 8765)]
        for value, address, port in cases:
            component.par.Serverurl = Par(value)
            self.assertEqual(namespace['server_address'](), address)
            self.assertEqual(namespace['server_port'](), port)
            self.assertEqual(namespace['probe_url'](), address.replace('/ws/stream', '/ws/ping'))

    def test_native_sockets_answer_protocol_keepalive_pings(self):
        for code in (builder.WEBSOCKET_CODE, builder.LATENCY_CODE):
            namespace = {}
            exec(code, namespace)
            dat = SimpleNamespace(sendPong=Mock())
            namespace['onReceivePing'](dat, b'keepalive')
            dat.sendPong.assert_called_once_with(b'keepalive')
