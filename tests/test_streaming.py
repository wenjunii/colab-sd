"""CPU checks for streaming behavior; no model download or Colab session needed."""
import asyncio
import json
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch
from PIL import Image

import build_clean_toe as builder
import colab_server as server


class ServerChecks(unittest.TestCase):
    def setUp(self):
        self.state = patch.multiple(
            server, pipeline_ready=True, current_mode="txt2img", current_seed=42,
            current_strength=0.6, current_prompt_a="A", current_prompt_b="B",
            current_prompt_blend=0.0, current_neg_prompt="", applied_prompt_state=None,
            prompt_embedding_cache={}, output_width=512, output_height=512,
            stream_txt2img=None, stream_img2img=None, enable_tensorrt=False,
            last_pipeline_error=None,
        )
        self.state.start()
        self.addCleanup(self.state.stop)

    def test_blending_reuses_embeddings_and_strength_restores_blend(self):
        pipe = SimpleNamespace(encode_prompt=Mock(side_effect=lambda **kw:
                               (torch.full((1, 2, 3), 1.0 if kw['prompt'] == 'A' else 3.0),)))
        txt = SimpleNamespace(stream=SimpleNamespace(pipe=pipe, batch_size=1, prompt_embeds=None))
        img = SimpleNamespace(stream=SimpleNamespace(batch_size=2, prompt_embeds=None))
        img.update_strength = lambda strength: setattr(img.stream, 'prompt_embeds', torch.zeros((2, 2, 3)))
        with patch.multiple(server, stream_txt2img=txt, stream_img2img=img), patch.object(server, 'set_seed'):
            for blend in (0.25, 0.5, 0.75, 0.5):
                server.apply_prompts_and_blend("A", "B", blend, "")
            server.apply_controls({'seed': 7, 'strength': 0.8})
            self.assertEqual(pipe.encode_prompt.call_count, 2)
            self.assertTrue(torch.all(img.stream.prompt_embeds == 2.0))
            self.assertEqual(img.stream.prompt_embeds.shape[0], 2)

    def test_editing_prompt_b_keeps_frequently_used_prompt_a_cached(self):
        pipe = SimpleNamespace(encode_prompt=Mock(return_value=(torch.ones((1, 2, 3)),)))
        txt = SimpleNamespace(stream=SimpleNamespace(pipe=pipe, batch_size=1, prompt_embeds=None))
        with patch.multiple(server, stream_txt2img=txt, stream_img2img=None):
            for index in range(10):
                server.apply_prompts_and_blend('A', f'B-{index}', 0.5, '')
        a_calls = [call for call in pipe.encode_prompt.call_args_list if call.kwargs['prompt'] == 'A']
        self.assertEqual(len(a_calls), 1)
        self.assertLessEqual(len(server.prompt_embedding_cache), 4)

    def test_output_resolution_reloads_only_when_size_changes(self):
        with patch.object(server, 'load_pipeline') as load, patch.object(server, 'apply_prompts_and_blend'):
            for width, height in server.output_resolutions:
                server.apply_controls({'width': width, 'height': height})
                self.assertEqual((server.output_width, server.output_height), (width, height))
            self.assertEqual(load.call_count, len(server.output_resolutions))
            server.apply_controls({'width': width, 'height': height})
            self.assertEqual(load.call_count, len(server.output_resolutions))
            for invalid in [(320, 512), (1024, 1024), (1024, 800), (2048, 1536)]:
                with self.assertRaises(ValueError):
                    server.apply_controls({'width': invalid[0], 'height': invalid[1]})
            self.assertEqual((server.output_width, server.output_height), (width, height))

    def test_invalid_controls_do_not_partially_mutate_state(self):
        with self.assertRaises(ValueError):
            server.apply_controls({'mode': 'img2img', 'seed': 123, 'blend': float('nan')})
        self.assertEqual(server.current_mode, 'txt2img')
        self.assertEqual(server.current_seed, 42)

    def test_failed_resize_attempts_to_restore_previous_pipeline(self):
        def load():
            if server.output_width == 1024:
                server.pipeline_ready = False
                raise RuntimeError('out of memory')
            server.pipeline_ready = True

        with patch.object(server, 'load_pipeline', side_effect=load) as loader:
            with self.assertRaisesRegex(RuntimeError, 'out of memory'):
                server.apply_controls({'width': 1024, 'height': 768, 'prompt_a': 'new prompt'})
            self.assertEqual(loader.call_count, 2)
            self.assertEqual((server.output_width, server.output_height), (512, 512))
            self.assertEqual(server.current_prompt_a, 'A')
            self.assertTrue(server.pipeline_ready)

    def test_health_reports_unavailable_when_resize_and_recovery_both_fail(self):
        with patch.object(server, 'load_pipeline', side_effect=RuntimeError('allocation failed')), \
                self.assertLogs(server.logger, level='ERROR'):
            with self.assertRaises(RuntimeError):
                server.apply_controls({'width': 1024, 'height': 768})
        self.assertFalse(server.pipeline_ready)
        self.assertEqual(server.health().status_code, 503)
        self.assertEqual(json.loads(server.health().body)['last_error'], 'Configuration failed; see Colab server logs')

    def test_engine_readout_follows_the_current_mode(self):
        with patch.multiple(server, active_engine='TensorRT (UNet)', img2img_engine='PyTorch (FP16)'):
            self.assertEqual(server.engine_for_mode('txt2img'), 'TensorRT (UNet)')
            self.assertEqual(server.engine_for_mode('img2img'), 'PyTorch (FP16)')

    def test_second_tensorrt_failure_never_reports_ready_or_activates_first_engine(self):
        source_unet = object()
        txt = Mock(stream=SimpleNamespace(pipe=SimpleNamespace(dtype=torch.float16),
                                         vae=None, unet=source_unet))
        core = Mock(unet=source_unet)
        with patch.multiple(server, enable_tensorrt=True, StreamDiffusionWrapper=Mock()), \
                patch.object(server, 'create_txt2img_pipeline', return_value=txt), \
                patch.dict('sys.modules', {'streamdiffusion': SimpleNamespace(StreamDiffusion=Mock(return_value=core))}), \
                patch.object(server, 'compile_tensorrt_unet', side_effect=[object(), RuntimeError('compile failed')]) as compile_unet, \
                patch.object(server, 'apply_prompts_and_blend'), patch('builtins.print'):
            with self.assertRaisesRegex(RuntimeError, 'compile failed'):
                server.load_pipeline()
            self.assertEqual([call.args[3] for call in compile_unet.call_args_list], [1, 2])
            self.assertIs(txt.stream.unet, source_unet)
            self.assertFalse(server.pipeline_ready)
            status = json.loads(server.health().body)
            self.assertEqual(status['requested_engine'], 'TensorRT')
            self.assertEqual(status['engine'], 'PyTorch (FP16)')
            self.assertEqual(server.health().status_code, 503)

    def test_resize_and_jpeg_keep_requested_dimensions_and_color(self):
        source = cv2.imencode('.jpg', np.full((256, 256, 3), (0, 0, 255), np.uint8))[1].tobytes()
        for width, height in server.output_resolutions:
            with self.subTest(size=(width, height)), patch.multiple(server, output_width=width, output_height=height), \
                    patch.object(server, 'run_model_inference', side_effect=lambda img, mode: img):
                data, inf_ms, process_ms = server.process_frame('img2img', source)
                image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(image.shape[:2], (height, width))
                self.assertGreater(int(image[0, 0, 2]), 240)
                self.assertGreaterEqual(process_ms, inf_ms)
        noise = np.random.default_rng(42).integers(0, 256, (768, 768, 3), dtype=np.uint8)
        encoded = server.compress_frame(noise, max_bytes=48000)
        self.assertLessEqual(len(encoded), 48000)
        self.assertEqual(cv2.imdecode(np.frombuffer(encoded, np.uint8), 1).shape[:2], (768, 768))
        largest = np.random.default_rng(1).integers(0, 256, (768, 1024, 3), dtype=np.uint8)
        encoded = server.compress_frame(largest)
        self.assertLessEqual(len(encoded), 144000)
        self.assertEqual(cv2.imdecode(np.frombuffer(encoded, np.uint8), 1).shape[:2], (768, 1024))

    def test_pipeline_initializes_single_text_image_and_resolution_specific_cache(self):
        calls = []

        class Wrapper:
            def __init__(self, **kwargs):
                calls.append(kwargs)
                self.stream = SimpleNamespace(pipe=SimpleNamespace(dtype=torch.float32), vae=None)

            def prepare(self, **kwargs):
                pass

            def __call__(self, *args):
                return Image.new('RGB', (1024, 768))

        class Core:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def prepare(self, *args, **kwargs):
                pass

            def __call__(self, image):
                return image

        with patch.multiple(server, StreamDiffusionWrapper=Wrapper, output_width=1024, output_height=768,
                            postprocess_image=lambda img, **kw: [img]), \
                patch.dict('sys.modules', {'streamdiffusion': SimpleNamespace(StreamDiffusion=Core)}), \
                patch.object(server.torch.cuda, 'is_available', return_value=False), \
                patch.object(server, 'apply_prompts_and_blend'):
            server.load_pipeline()
        self.assertEqual(calls[0]['t_index_list'], [0])
        self.assertEqual((calls[0]['width'], calls[0]['height']), (1024, 768))
        self.assertEqual(calls[0]['acceleration'], 'none')
        self.assertEqual((server.stream_img2img.stream.kwargs['width'],
                          server.stream_img2img.stream.kwargs['height']), (1024, 768))



class FakeWebSocket:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.text = []
        self.binary = []

    async def accept(self):
        pass

    async def receive(self):
        return await self.incoming.get()

    async def send_text(self, text):
        self.text.append(json.loads(text))

    async def send_bytes(self, data):
        self.binary.append(data)

    async def close(self, **kwargs):
        await self.incoming.put({'type': 'websocket.disconnect'})


async def until(predicate, timeout=2.0):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.005)
    await asyncio.wait_for(wait(), timeout)


class WebSocketChecks(unittest.IsolatedAsyncioTestCase):
    async def test_backend_exception_details_stay_out_of_public_status(self):
        ws = FakeWebSocket()
        private_detail = 'internal/private/server/path: backend allocation failed'
        with patch.multiple(server, pipeline_ready=True, current_mode='img2img'), \
                patch.object(server, 'apply_controls', side_effect=RuntimeError(private_detail)), \
                patch.object(server, 'get_gpu_telemetry', return_value=('--', '--')):
            task = asyncio.create_task(server.websocket_stream_endpoint(ws))
            try:
                await until(lambda: ws.text)
                with self.assertLogs(server.logger, level='ERROR') as logs:
                    await ws.incoming.put({'text': json.dumps({'type': 'control', 'seed': 42})})
                    await until(lambda: any(message.get('code') == 'configuration_failed' for message in ws.text))
                error = next(message for message in ws.text if message.get('code') == 'configuration_failed')
                self.assertNotIn(private_detail, json.dumps(error))
                self.assertIn('Colab logs', error['status'])
                self.assertIn(private_detail, '\n'.join(logs.output))
            finally:
                await ws.close()
                await asyncio.wait_for(task, 2)

    async def test_additional_client_is_rejected_and_owner_is_released(self):
        owner, extra = FakeWebSocket(), FakeWebSocket()
        with patch.multiple(server, pipeline_ready=True, current_mode='img2img'), \
                patch.object(server, 'get_gpu_telemetry', return_value=('--', '--')):
            task = asyncio.create_task(server.websocket_stream_endpoint(owner))
            try:
                await until(lambda: owner.text)
                await server.websocket_stream_endpoint(extra)
                self.assertEqual(extra.text[0]['code'], 'server_busy')
                self.assertIs(server.active_client, owner)
            finally:
                await owner.close()
                await asyncio.wait_for(task, 2)
            self.assertIsNone(server.active_client)
    async def test_ping_stays_responsive_and_pending_video_uses_latest_frame(self):
        ws = FakeWebSocket()
        started, release = threading.Event(), threading.Event()
        processed = []
        payload = b'x' * 7000

        def process(mode, data):
            processed.append(data)
            if len(processed) == 1:
                started.set()
                release.wait(3)
            return payload, 10.0, 12.0

        with patch.multiple(server, pipeline_ready=True, current_mode='img2img'), \
                patch.object(server, 'process_frame', side_effect=process), \
                patch.object(server, 'get_gpu_telemetry', return_value=('--', '--')):
            task = asyncio.create_task(server.websocket_stream_endpoint(ws))
            try:
                await until(lambda: ws.text)
                await ws.incoming.put({'bytes': b'FRM:' + bytes([1, 0, 1]) + b'first'})
                await until(started.is_set)
                # Recovery frames arrive while GPU work is blocked. Only newest survives.
                for fid, data in [(2, b'stale'), (3, b'latest')]:
                    await ws.incoming.put({'bytes': b'FRM:' + bytes([fid, 0, 1]) + data})
                await ws.incoming.put({'text': json.dumps({'type': 'ping', 'sent_at': 123.0})})
                await until(lambda: any(msg.get('type') == 'pong' for msg in ws.text))
                self.assertFalse(release.is_set())
                self.assertEqual(next(msg for msg in ws.text if msg.get('type') == 'pong')['sent_at'], 123.0)
                release.set()
                await until(lambda: len(ws.binary) == 8)
                self.assertEqual(processed, [b'first', b'latest'])
                self.assertTrue(all(len(chunk) <= 2051 for chunk in ws.binary))
                self.assertEqual(b''.join(chunk[3:] for chunk in ws.binary[:4]), payload)
            finally:
                release.set()
                await ws.close()
                await asyncio.wait_for(task, 2)

    async def test_resolution_reload_does_not_block_ping_or_queue_controls(self):
        ws = FakeWebSocket()
        started, release = threading.Event(), threading.Event()
        applied = []

        def control(data):
            applied.append(data.copy())
            if len(applied) == 1:
                started.set()
                release.wait(3)

        with patch.multiple(server, pipeline_ready=True, current_mode='img2img'), \
                patch.object(server, 'apply_controls', side_effect=control), \
                patch.object(server, 'get_gpu_telemetry', return_value=('--', '--')):
            task = asyncio.create_task(server.websocket_stream_endpoint(ws))
            try:
                await until(lambda: ws.text)
                await ws.incoming.put({'text': json.dumps({'type': 'control', 'width': 384, 'height': 384})})
                await until(started.is_set)
                for strength in (0.2, 0.4, 0.8):
                    await ws.incoming.put({'text': json.dumps({'type': 'control', 'strength': strength})})
                await ws.incoming.put({'text': json.dumps({'type': 'ping', 'sent_at': 456.0})})
                await until(lambda: any(msg.get('type') == 'pong' for msg in ws.text))
                release.set()
                await until(lambda: len(applied) == 2)
                self.assertEqual(applied[-1]['strength'], 0.8)
            finally:
                release.set()
                await ws.close()
                await asyncio.wait_for(task, 2)


class Par:
    def __init__(self, value):
        self.val = value

    def eval(self):
        return self.val

    def __bool__(self):
        return True


class TouchDesignerChecks(unittest.TestCase):
    def test_embedded_callbacks_compile(self):
        for name in ('SCRIPT_RECEIVER_CODE', 'WEBSOCKET_CODE', 'EXEC_SEND_FRAME_CODE', 'PAREXEC_CODE'):
            compile(getattr(builder, name), name, 'exec')

    def test_input_preview_works_before_receiving_any_ai_frame(self):
        component = SimpleNamespace(par=SimpleNamespace(Displaymode=Par('input')), fetch=lambda key, default=None: default)
        pixels = np.zeros((4, 8, 4), np.float32)
        pixels[:, :, 1] = 1
        cam = SimpleNamespace(numpyArray=lambda: pixels)
        script = SimpleNamespace(par=SimpleNamespace(resolutionw=Par(8), resolutionh=Par(4)), copyNumpyArray=Mock())
        namespace = {'parent': lambda: component, 'op': lambda name: cam if name == 'null_cam' else None}
        exec(builder.SCRIPT_RECEIVER_CODE, namespace)
        namespace['onCook'](script)
        result = script.copyNumpyArray.call_args.args[0]
        self.assertEqual(result.shape, (4, 8, 4))
        self.assertTrue(np.all(result[:, :, 1] == 255))

    def test_upload_is_paused_until_server_is_ready(self):
        storage = {'connected': True, 'server_ready': False}
        component = SimpleNamespace(par=SimpleNamespace(Mode=Par('img2img'), Audiorx=Par(False)),
                                    fetch=lambda key, default=None: storage.get(key, default),
                                    store=lambda key, value: storage.__setitem__(key, value))
        cam = Mock()
        ws = SimpleNamespace(par=SimpleNamespace(active=Par(True)), sendText=Mock(), sendBinary=Mock())
        namespace = {'parent': lambda: component, 'op': lambda name: ws if name == 'websocket1' else cam}
        exec(builder.EXEC_SEND_FRAME_CODE, namespace)
        namespace['onFrameStart'](1)
        ws.sendBinary.assert_not_called()
        cam.numpyArray.assert_not_called()

    def test_audio_seed_triggers_on_crossings_and_text_mode_skips_denoise_updates(self):
        amplitude = [0.9]
        class Audio:
            numChans = 1

            def __getitem__(self, index):
                return SimpleNamespace(eval=lambda: amplitude[0])

        pars = SimpleNamespace(Audiorx=Par(True), Mode=Par('txt2img'), Seed=Par(42), Audseedthresh=0.8)
        component = SimpleNamespace(par=pars)
        lookup = Mock(return_value=Audio())
        namespace = {'parent': lambda: component, 'op': lookup}
        exec(builder.EXEC_SEND_FRAME_CODE, namespace)
        with patch.object(namespace['random'], 'randint', return_value=123) as random_seed, patch('builtins.print'):
            namespace['handle_audio_reactivity'](1)
            namespace['handle_audio_reactivity'](2)
            self.assertEqual(random_seed.call_count, 1)
            amplitude[0] = 0.1
            namespace['handle_audio_reactivity'](2.1)
            amplitude[0] = 0.9
            namespace['handle_audio_reactivity'](2.5)
            self.assertEqual(random_seed.call_count, 2)
        self.assertTrue(all(call.args[0] == 'in_audio' for call in lookup.call_args_list))

    def test_pasted_https_tunnel_url_is_normalized(self):
        url = Par('https://example.trycloudflare.com/')
        url.name = 'Serverurl'
        component = SimpleNamespace(par=SimpleNamespace())
        namespace = {'parent': lambda: component}
        exec(builder.PAREXEC_CODE, namespace)
        namespace['onValueChange'](url, '')
        self.assertEqual(url.val, 'wss://example.trycloudflare.com/ws/stream')

    def test_incompatible_server_version_has_a_clear_status(self):
        storage = {}
        pars = SimpleNamespace(Status=Par(''), Connect=Par(True))
        component = SimpleNamespace(par=pars, fetch=lambda key, default=None: storage.get(key, default),
                                    store=lambda key, value: storage.__setitem__(key, value))
        namespace = {'parent': lambda: component}
        exec(builder.WEBSOCKET_CODE, namespace)
        with patch('builtins.print'):
            namespace['onReceiveText'](None, 0, json.dumps({'type': 'welcome'}))
        self.assertFalse(pars.Connect.val)
        self.assertFalse(storage['server_ready'])
        self.assertIn('versions differ', pars.Status.val)

    def test_output_presets_send_separate_dimensions(self):
        import shlex
        row = next(line for line in builder.CPARM_CONTENT.splitlines() if ' Resolution ' in line)
        fields = shlex.split(row)
        menu_start = fields.index('4097')
        menu_count = int(fields[menu_start + 1])
        values = fields[menu_start + 2:-1:2]
        self.assertEqual(len(values), menu_count)
        self.assertEqual(menu_count, len(server.output_resolutions))
        self.assertIn('1024x768', values)
        self.assertIn('768x1024', values)
        messages = []
        pars = SimpleNamespace(Mode=Par('txt2img'), Resolution=Par('512'))
        storage = {}
        component = SimpleNamespace(par=pars, fetch=lambda key, default=None: storage.get(key, default),
                                    store=lambda key, value: storage.__setitem__(key, value))
        ws = SimpleNamespace(par=SimpleNamespace(active=Par(True)),
                             sendText=lambda data: messages.append(json.loads(data)))
        namespace = {'parent': lambda: component, 'op': lambda name: ws}
        exec(builder.WEBSOCKET_CODE, namespace)
        for value, expected in zip(values, server.output_resolutions):
            with self.subTest(value=value), patch('builtins.print'):
                pars.Resolution.val = value
                namespace['send_control']()
                self.assertEqual((messages[-1]['width'], messages[-1]['height']), expected)
                self.assertLessEqual(expected[0] * expected[1], 1024 * 768)
                self.assertEqual(expected[0] % 8, 0)
                self.assertEqual(expected[1] % 8, 0)

    def test_upload_preserves_red_and_uses_gpu_downsample_branch(self):
        storage = {'connected': True, 'server_ready': True}
        component = SimpleNamespace(par=SimpleNamespace(Mode=Par('img2img'), Audiorx=Par(False)),
                                    fetch=lambda key, default=None: storage.get(key, default),
                                    store=lambda key, value: storage.__setitem__(key, value))
        array = np.zeros((256, 256, 4), dtype=np.float32)
        array[:, :, 0] = 1
        cam = SimpleNamespace(cook=Mock(), numpyArray=lambda: array)
        chunks = []
        ws = SimpleNamespace(par=SimpleNamespace(active=Par(True)), sendText=Mock(), sendBinary=chunks.append)
        ops = {'fit_send': cam, 'websocket1': ws}
        namespace = {'parent': lambda: component, 'op': lambda name: ops.get(name)}
        exec(builder.EXEC_SEND_FRAME_CODE, namespace)
        namespace['onFrameStart'](1)
        jpeg = b''.join(chunk[7:] for chunk in chunks)
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        self.assertGreater(int(image[0, 0, 2]), 240)
        self.assertLess(int(image[0, 0, 0]), 10)
        self.assertFalse(storage['can_send'])

    def test_rectangular_upload_preserves_aspect_ratio(self):
        for width, height in [(256, 192), (192, 256), (256, 144), (144, 256)]:
            storage = {'connected': True, 'server_ready': True}
            component = SimpleNamespace(par=SimpleNamespace(Mode=Par('img2img'), Audiorx=Par(False)),
                                        fetch=lambda key, default=None: storage.get(key, default),
                                        store=lambda key, value: storage.__setitem__(key, value))
            cam = SimpleNamespace(cook=Mock(), numpyArray=lambda: np.zeros((height, width, 4), np.float32))
            chunks = []
            ws = SimpleNamespace(par=SimpleNamespace(active=Par(True)), sendText=Mock(), sendBinary=chunks.append)
            ops = {'fit_send': cam, 'websocket1': ws}
            namespace = {'parent': lambda: component, 'op': lambda name: ops.get(name)}
            exec(builder.EXEC_SEND_FRAME_CODE, namespace)
            with self.subTest(size=(width, height)), patch('builtins.print'):
                namespace['onFrameStart'](1)
                jpeg = b''.join(chunk[7:] for chunk in chunks)
                self.assertEqual(cv2.imdecode(np.frombuffer(jpeg, np.uint8), 1).shape[:2], (height, width))

    def test_received_frames_defer_display_and_measure_ping_separately(self):
        storage = {'last_send_time': time.perf_counter() - 0.1}
        pars = SimpleNamespace(Mode=Par('img2img'), Frametime=Par(''), Ping=Par(''))
        component = SimpleNamespace(par=pars, fetch=lambda key, default=None: storage.get(key, default),
                                    store=lambda key, value: storage.__setitem__(key, value))
        op = Mock()
        namespace = {'parent': lambda: component, 'op': op}
        exec(builder.WEBSOCKET_CODE, namespace)
        namespace['onReceiveBinary'](None, bytes([255, 0, 2]) + b'abc')
        namespace['onReceiveBinary'](None, bytes([255, 1, 2]) + b'def')
        self.assertEqual(storage['last_frame_bytes'], b'abcdef')
        self.assertTrue(storage['display_pending'])
        self.assertTrue(storage['can_send'])
        op.assert_not_called()
        self.assertEqual(pars.Ping.val, '')
        namespace['onReceiveText'](None, 0, json.dumps({'type': 'pong', 'sent_at': time.perf_counter() - 0.02}))
        self.assertGreaterEqual(float(pars.Ping.val.split()[0]), 20)


if __name__ == '__main__':
    unittest.main()
