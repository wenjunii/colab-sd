import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

import build_clean_toe as builder
import colab_server as server
from colab_launcher import BridgeSession
import workstation


class WorkstationChecks(unittest.TestCase):
    def test_cloudflare_protocol_cli_reaches_the_launcher(self):
        with tempfile.TemporaryDirectory() as name:
            runtime = Path(name).resolve()
            python = workstation.runtime_python(runtime)
            python.parent.mkdir(parents=True)
            python.touch()
            wrapper = runtime / 'StreamDiffusion/utils/wrapper.py'
            wrapper.parent.mkdir(parents=True)
            wrapper.touch()
            (runtime / 'workstation.json').write_text(json.dumps({
                'managed_by': 'colab-sd', 'revision': workstation.UPSTREAM_REVISION, 'ready': True}))
            for transport, option, expected in (('cloudflare', None, 'http2'),
                                                 ('cloudflare', 'auto', 'auto'),
                                                 ('cloudflare', 'quic', 'quic'),
                                                 ('lan', None, 'http2')):
                with self.subTest(transport=transport, protocol=option), \
                        patch.object(workstation, 'gpu_check'), \
                        patch.object(workstation, 'pairing_token', return_value='test-token'), \
                        patch.object(workstation, 'cloudflared_binary', return_value='/cloudflared') as binary, \
                        patch.object(workstation, 'BridgeSession') as launcher, patch('builtins.print'):
                    arguments = ['start', '--runtime', str(runtime), '--tensorrt', '--resolution', '1024x768']
                    if transport != 'lan':
                        arguments += ['--transport', transport]
                    if option is not None:
                        arguments += ['--tunnel-protocol', option]
                    self.assertEqual(workstation.main(arguments), 0)
                    settings = launcher.call_args.kwargs
                    self.assertEqual(settings['tunnel_protocol'], expected)
                    self.assertEqual(settings['output_size'], (1024, 768))
                    self.assertTrue(settings['tensorrt'])
                    self.assertTrue(settings['reuse_engines'])
                    self.assertEqual(settings['access_token'], 'test-token')
                    self.assertEqual(settings['tunnel'], transport == 'cloudflare')
                    self.assertEqual(settings['host'], '127.0.0.1' if settings['tunnel'] else '0.0.0.0')
                    if settings['tunnel']:
                        binary.assert_called_once_with(runtime)
                    else:
                        binary.assert_not_called()
                    launcher.return_value.start.assert_called_once()
                    launcher.return_value.wait.assert_called_once()
                    launcher.return_value.stop.assert_called_once()

    def test_setup_installs_cuda_into_its_own_environment_and_marks_success(self):
        with tempfile.TemporaryDirectory() as name, patch('workstation.check_platform'), \
                patch('workstation.shutil.which', return_value='git'), patch('workstation.venv.EnvBuilder'), \
                patch('workstation.subprocess.run'), patch('workstation.gpu_check'), \
                patch('workstation.run_pip') as pip, patch('builtins.print'):
            workstation.setup(name)
            for call in pip.call_args_list:
                self.assertEqual(call.args[0], workstation.runtime_python(name))
            self.assertTrue(any('torch==2.13.0+cu130' in call.args for call in pip.call_args_list))
            self.assertTrue(json.loads((Path(name) / 'workstation.json').read_text())['ready'])

    def test_lan_launch_uses_requested_port_interpreter_and_private_token(self):
        with tempfile.TemporaryDirectory() as name:
            session = BridgeSession(root=name, tunnel=False, host='0.0.0.0', port=8765,
                                    advertise_host='192.168.1.50', python_executable='/venv/python',
                                    server_script=Path(name) / 'bridge.py', access_token='private-test-token')
            with patch('colab_launcher.socket.create_connection', side_effect=ConnectionRefusedError), \
                    patch.object(session, '_spawn') as spawn, patch.object(session, 'wait_for_ready'):
                url = session.start()
            self.assertEqual(url, 'ws://192.168.1.50:8765/ws/stream')
            spawn.assert_called_once()
            command, environment = spawn.call_args.args[1:]
            self.assertEqual(command[0], '/venv/python')
            self.assertNotIn('private-test-token', ' '.join(command))
            self.assertEqual(environment['BRIDGE_ACCESS_TOKEN'], 'private-test-token')
            self.assertEqual(environment['BRIDGE_PORT'], '8765')
            self.assertEqual(environment['BRIDGE_SESSION_ID'], session.session_id)

    def test_readiness_rejects_an_unrelated_process_and_handles_ipv6(self):
        session = BridgeSession(tunnel=False, host='::', port=8765)
        self.assertEqual(session.endpoint(path='/health'), 'http://[::1]:8765/health')
        session.processes['server'] = SimpleNamespace(poll=lambda: None)
        with patch('colab_launcher.urllib.request.urlopen', return_value=io.StringIO('{"ready":true,"session_id":"another"}')):
            with self.assertRaisesRegex(RuntimeError, 'different server'):
                session.wait_for_ready()

    def test_pairing_key_persists_and_cloudflared_download_matches_os(self):
        with tempfile.TemporaryDirectory() as name:
            token = workstation.pairing_token(name)
            self.assertGreaterEqual(len(token), 32)
            self.assertEqual(token, workstation.pairing_token(name))
            target = Path(name) / ('cloudflared.exe' if workstation.os.name == 'nt' else 'cloudflared')
            def download(url, path):
                self.assertIn('windows-amd64.exe' if workstation.os.name == 'nt' else 'linux-amd64', url)
                Path(path).write_bytes(b'test executable')
            with patch('workstation.shutil.which', return_value=None), \
                    patch('workstation.urllib.request.urlretrieve', side_effect=download):
                self.assertEqual(workstation.cloudflared_binary(name), target)
            self.assertEqual(target.read_bytes(), b'test executable')

    def test_setup_does_not_write_into_an_unmanaged_folder(self):
        with tempfile.TemporaryDirectory() as name:
            marker = Path(name) / 'user-data'
            marker.write_text('keep this')
            with patch('workstation.check_platform'), patch('workstation.subprocess.run') as run:
                with self.assertRaisesRegex(RuntimeError, 'not empty'):
                    workstation.setup(name)
            run.assert_not_called()
            self.assertEqual(marker.read_text(), 'keep this')


class AuthenticatedStreamChecks(unittest.TestCase):
    def setUp(self):
        self.token = 'test-workstation-access-token-32-characters'
        self.state = patch.multiple(server, access_token=self.token, pipeline_ready=True,
                                    current_mode='img2img', active_client=None,
                                    stream_txt2img=None, stream_img2img=None)
        self.state.start()
        self.addCleanup(self.state.stop)

    def test_unauthorized_clients_never_take_control_or_receive_video(self):
        with patch.object(server, 'load_pipeline'), patch.object(server, 'process_frame') as process, TestClient(server.app) as client:
            with client.websocket_connect('/ws/stream') as ws:
                self.assertEqual(ws.receive_json()['type'], 'auth_required')
                ws.send_json({'type': 'auth', 'token': 'wrong'})
                error = ws.receive_json()
                self.assertEqual(error['code'], 'unauthorized')
                self.assertNotIn(self.token, json.dumps(error))
                self.assertIsNone(server.active_client)
            process.assert_not_called()
            self.assertNotIn(self.token, client.get('/health').text)

    def test_authenticated_video_ping_and_disconnect_with_real_asgi_transport(self):
        pixels = np.full((32, 32, 3), (0, 0, 255), np.uint8)
        jpeg = cv2.imencode('.jpg', pixels)[1].tobytes()
        with patch.object(server, 'load_pipeline'), \
                patch.object(server, 'run_model_inference', side_effect=lambda image, mode: image), \
                patch.object(server, 'get_gpu_telemetry', return_value=('--', '--')), TestClient(server.app) as client:
            with client.websocket_connect('/ws/stream') as ws:
                self.assertEqual(ws.receive_json()['type'], 'auth_required')
                ws.send_json({'type': 'auth', 'token': self.token})
                self.assertEqual(ws.receive_json()['type'], 'welcome')
                ws.send_json({'type': 'ping', 'sent_at': 123})
                self.assertEqual(ws.receive_json(), {'type': 'pong', 'sent_at': 123})
                ws.send_bytes(b'FRM:' + bytes([1, 0, 1]) + jpeg)
                chunks = {}
                for _ in range(100):
                    message = ws.receive()
                    if message.get('bytes'):
                        chunk = message['bytes']
                        chunks[chunk[1]] = chunk[3:]
                        if len(chunks) == chunk[2]:
                            break
                output = cv2.imdecode(np.frombuffer(b''.join(chunks[i] for i in sorted(chunks)), np.uint8), 1)
                self.assertEqual(output.shape[:2], (server.output_height, server.output_width))
                self.assertGreater(int(output[0, 0, 2]), 240)
            self.assertIsNone(server.active_client)


class TouchDesignerWorkstationChecks(unittest.TestCase):
    def test_location_switch_remembers_each_url_and_disconnects(self):
        class Par:
            def __init__(self, value): self.val = value
            def eval(self): return self.val
        class Params:
            def __setattr__(self, name, value):
                if isinstance(getattr(self, name, None), Par) and not isinstance(value, Par):
                    getattr(self, name).val = value
                else:
                    super().__setattr__(name, value)
        pars, storage = Params(), {}
        pars.Serverurl, pars.Connect = Par('wss://colab.example/ws/stream'), Par(True)
        pars.Serverlocation, pars.Status = Par('workstation'), Par('')
        pars.Serverlocation.name = 'Serverlocation'
        component = SimpleNamespace(par=pars, fetch=lambda key, default=None: storage.get(key, default),
                                    store=lambda key, value: storage.__setitem__(key, value))
        namespace = {'parent': lambda: component}
        exec(builder.PAREXEC_CODE, namespace)
        namespace['onValueChange'](pars.Serverlocation, 'colab')
        self.assertFalse(pars.Connect.eval())
        self.assertEqual(pars.Serverurl.eval(), 'ws://127.0.0.1:8000/ws/stream')
        pars.Serverurl.val = 'ws://rtx.example:8765/ws/stream'
        pars.Serverlocation.val = 'colab'
        namespace['onValueChange'](pars.Serverlocation, 'workstation')
        self.assertEqual(pars.Serverurl.eval(), 'wss://colab.example/ws/stream')
        pars.Serverlocation.val = 'workstation'
        namespace['onValueChange'](pars.Serverlocation, 'colab')
        self.assertEqual(pars.Serverurl.eval(), 'ws://rtx.example:8765/ws/stream')

    def test_custom_ports_and_authentication_precede_control_messages(self):
        storage = {}
        class Par:
            def __init__(self, value): self.val = value
            def eval(self): return self.val
        pars = SimpleNamespace(Serverurl=Par('ws://192.168.1.50:8765/ws/stream'),
                               Accesstoken=Par('private-client-token'), Status=Par(''), Connect=Par(True))
        component = SimpleNamespace(par=pars, fetch=lambda key, default=None: storage.get(key, default),
                                    store=lambda key, value: storage.__setitem__(key, value))
        namespace = {'parent': lambda: component, 'op': Mock()}
        exec(builder.PAREXEC_CODE, namespace)
        self.assertEqual(namespace['server_port'](), 8765)
        pars.Serverurl.val = 'wss://example.trycloudflare.com/ws/stream'
        self.assertEqual(namespace['server_port'](), 443)
        exec(builder.WEBSOCKET_CODE, namespace)
        dat = SimpleNamespace(sendText=Mock())
        namespace['onConnect'](dat)
        self.assertFalse(storage['handshake_complete'])
        namespace['send_control']()
        dat.sendText.assert_not_called()
        namespace['onReceiveText'](dat, 0, '{"type":"auth_required"}')
        self.assertEqual(json.loads(dat.sendText.call_args.args[0]), {'type': 'auth', 'token': 'private-client-token'})
        self.assertFalse(storage['server_ready'])
