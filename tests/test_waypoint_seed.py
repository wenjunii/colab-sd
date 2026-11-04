"""Prompt-to-world scheduling and controls without GPU packages or model downloads."""
from types import SimpleNamespace
import asyncio
import json
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

import build_clean_toe as builder
import colab_server as server
from waypoint_backend import WaypointBackend, validate_world_controls
from waypoint_seed import PromptSeedGenerator, MODEL_REVISION, SEED_SIZE
from test_waypoint import fake_backend, seed_jpeg
from test_streaming import Par, FakeWebSocket, until


def hybrid():
    backend = fake_backend()
    backend.seed_generator = SimpleNamespace(generate=Mock(
        side_effect=lambda *args: np.full((90, 160, 3), args[-1] % 256, np.uint8)))
    backend.configure({'world_seed_source': 'sdturbo', 'prompt_a': 'mountain valley'})
    return backend


def consume_batch(backend):
    for _ in range(4):
        assert backend.next_frame() is not None


class PromptWorldChecks(unittest.TestCase):
    def test_reseed_timer_starts_after_compilation_and_waits_for_batch_boundary(self):
        backend = hybrid()
        batch = backend.engine.gen_frame.return_value
        with patch('waypoint_backend.time.monotonic', return_value=100) as clock:
            def slow_first_batch(**kwargs):
                clock.return_value = 200
                return batch
            backend.engine.gen_frame.side_effect = slow_first_batch
            backend.next_frame()
            self.assertEqual(backend._seed_deadline, 205)
            clock.return_value = 206
            backend.engine.gen_frame.side_effect = None
            for _ in range(3):
                backend.next_frame()
            self.assertEqual(backend.seed_count, 1)
            consume_batch(backend)
            self.assertEqual(backend.seed_count, 2)
            self.assertEqual([c.args[-1] for c in backend.seed_generator.generate.call_args_list], [42, 43])
            self.assertEqual(backend.engine.append_frame.call_count, 2)

    def test_auto_switch_off_keeps_world_playing_and_manual_reseed_still_works(self):
        backend = hybrid()
        consume_batch(backend)
        backend.configure({'world_auto_seed': False})
        backend._seed_deadline = 0
        for _ in range(3):
            consume_batch(backend)
        self.assertEqual(backend.seed_count, 1)
        self.assertGreater(backend.engine.gen_frame.call_count, 1)
        backend.configure({'prompt_a': 'desert', 'world_seed_interval': 5})
        consume_batch(backend)
        self.assertEqual(backend.seed_count, 1)  # editing alone does not reset with Auto off
        backend.configure({'world_reseed': True})
        consume_batch(backend)
        self.assertEqual(backend.seed_count, 2)
        self.assertFalse(backend.auto_seed)
        backend.configure({'world_auto_seed': True, 'world_seed_interval': 10})
        backend._seed_deadline = 0
        consume_batch(backend)
        self.assertEqual(backend.seed_count, 3)

    def test_held_movement_controls_do_not_postpone_the_reseed_timer(self):
        backend = hybrid()
        consume_batch(backend)
        deadline = backend._seed_deadline
        for _ in range(10):
            backend.configure({'world_buttons': [87], 'world_seed_interval': 5,
                               'world_auto_seed': True, 'prompt_a': 'mountain valley'})
        self.assertEqual(backend._seed_deadline, deadline)

    def test_pause_stops_both_models_and_input_seed_disables_prompt_source(self):
        backend = hybrid()
        backend.configure({'world_paused': True})
        self.assertIsNone(backend.next_frame())
        backend.seed_generator.generate.assert_not_called()
        backend.configure({'world_paused': False})
        consume_batch(backend)
        backend.set_seed_image(seed_jpeg())
        backend._seed_deadline = 0
        consume_batch(backend)
        self.assertEqual(backend.seed_source, 'input')
        self.assertEqual(backend.seed_count, 1)

    def test_invalid_or_unavailable_prompt_controls_cannot_partially_mutate_world(self):
        backend = fake_backend()
        for invalid in ({'world_seed_source': 'sdturbo'}, {'world_seed_interval': .1},
                        {'world_seed_interval': float('nan')}, {'world_auto_seed': 'on'},
                        {'blend': 2}, {'prompt_a': ''}, {'world_reseed': 1}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                backend.configure({'world_buttons': [87], **invalid})
            self.assertEqual(backend.buttons, [])
            self.assertEqual(backend.seed_source, 'input')
        self.assertEqual(validate_world_controls({'world_seed_interval': 5})['world_seed_interval'], 5)

    def test_seed_failure_stops_auto_retry_and_preserves_an_existing_world(self):
        backend = hybrid()
        backend.seed_generator.generate.side_effect = ValueError('test generation failure')
        with self.assertLogs('waypoint_backend', level='ERROR'):
            self.assertIsNone(backend.next_frame())
        self.assertFalse(backend.auto_seed)
        self.assertFalse(backend.can_generate)
        self.assertIsNone(backend.next_frame())
        self.assertEqual(backend.seed_generator.generate.call_count, 1)
        backend.seed_generator.generate.side_effect = lambda *args: np.zeros((90, 160, 3), np.uint8)
        backend.configure({'world_reseed': True})
        consume_batch(backend)
        backend.seed_generator.generate.side_effect = ValueError('later failure')
        backend.configure({'world_reseed': True})
        with self.assertLogs('waypoint_backend', level='ERROR'):
            consume_batch(backend)
        self.assertTrue(backend.can_generate)
        self.assertEqual(backend.seed_count, 1)
        self.assertIsNotNone(backend.seed_error)

    def test_seed_generator_blends_cached_embeddings_with_single_step_and_new_noise(self):
        pipe = Mock(return_value=SimpleNamespace(images=np.ones((1, SEED_SIZE[1], SEED_SIZE[0], 3))))
        pipe.encode_prompt.side_effect = lambda text, **kwargs: (torch.full((1, 2, 3), 1.0 if text == 'A' else 3.0),)
        generator = Mock()
        fake_torch = SimpleNamespace(inference_mode=torch.inference_mode, Generator=Mock(return_value=generator))
        seeds = PromptSeedGenerator(pipe, fake_torch)
        for number in (42, 43):
            rgb = seeds.generate('A', 'B', .25, number)
            self.assertEqual(rgb.dtype, np.uint8)
            self.assertTrue(torch.all(pipe.call_args.kwargs['prompt_embeds'] == 1.5))
        self.assertEqual(pipe.encode_prompt.call_count, 2)
        self.assertEqual(pipe.call_args.kwargs['guidance_scale'], 0)
        self.assertEqual(pipe.call_args.kwargs['num_inference_steps'], 1)
        self.assertEqual([c.args[0] for c in generator.manual_seed.call_args_list], [42, 43])

    def test_seed_loader_pins_official_safetensors_without_legacy_streamdiffusion(self):
        pipeline_class = Mock()
        with patch.dict('sys.modules', {'diffusers': SimpleNamespace(StableDiffusionPipeline=pipeline_class)}):
            PromptSeedGenerator.load()
        self.assertEqual(pipeline_class.from_pretrained.call_args.kwargs['revision'], MODEL_REVISION)
        self.assertTrue(pipeline_class.from_pretrained.call_args.kwargs['use_safetensors'])
        pipeline_class.from_pretrained.return_value.to.assert_called_once_with('cuda')

    def test_world_loader_advertises_prompt_seed_capability_only_when_enabled(self):
        world = SimpleNamespace(WorldEngine=Mock(), CtrlInput=Mock())
        with patch.dict('sys.modules', {'world_engine': world, 'gemlite': Mock()}), \
                patch('waypoint_backend.engine_class_for_profile', return_value=world.WorldEngine), \
                patch('huggingface_hub.snapshot_download', return_value='local/model'), \
                patch.object(torch.cuda, 'is_available', return_value=True), \
                patch.object(torch.cuda, 'get_device_capability', return_value=(8, 0)), \
                patch.object(PromptSeedGenerator, 'load') as load_seed:
            original = WaypointBackend.load(prompt_seeds=False)
            load_seed.assert_not_called()
            self.assertFalse(original.progress()['world_prompt_seeding'])
            combined = WaypointBackend.load(prompt_seeds=True)
            load_seed.assert_called_once()
            self.assertIs(combined.seed_generator, load_seed.return_value)
            self.assertTrue(combined.progress()['world_prompt_seeding'])

    def test_td_sends_prompt_seed_settings_only_when_supported(self):
        pars = SimpleNamespace(Mode=Par('world'), Resolution=Par('1280x720'), Status=Par(''),
                               Worldbuttons=Par(''), Worldmousex=Par(0), Worldmousey=Par(0),
                               Worldpaused=Par(False), Worldseedsource=Par('sdturbo'),
                               Worldautoseed=Par(False), Worldseedinterval=Par(9),
                               Prompta=Par('forest'), Promptb=Par('city'), Promptblend=Par(.3), Seed=Par(42))
        storage = {'supported_modes': ['world'], 'world_prompt_seeding': True}
        comp = SimpleNamespace(par=pars, fetch=lambda k, d=None: storage.get(k, d),
                               store=lambda k, v: storage.__setitem__(k, v))
        ws = SimpleNamespace(par=SimpleNamespace(active=Par(True)), sendText=Mock())
        ns = {'parent': lambda: comp, 'op': lambda name: ws}
        exec(builder.WEBSOCKET_CODE, ns)
        ns['send_control'](world_reseed=True)
        import json
        control = json.loads(ws.sendText.call_args.args[0])
        self.assertEqual(control['prompt_a'], 'forest')
        self.assertFalse(control['world_auto_seed'])
        self.assertEqual(control['world_seed_interval'], 9)
        self.assertTrue(control['world_reseed'])
        ns['send_control'](force=True, world_reset=True)
        reset = json.loads(ws.sendText.call_args.args[0])
        self.assertTrue(reset['world_reset'])
        self.assertFalse(reset['world_reseed'])
        ws.sendText.reset_mock()
        storage['world_prompt_seeding'] = False
        ns['send_control']()
        ws.sendText.assert_not_called()
        self.assertIn('Enable prompt seeds', str(pars.Status))


class PromptWorldTransportChecks(unittest.IsolatedAsyncioTestCase):
    async def test_manual_seed_pulse_survives_slider_coalescing_behind_busy_worker(self):
        backend, ws = hybrid(), FakeWebSocket()
        backend.paused = True
        started, release = threading.Event(), threading.Event()
        applied = []
        real_apply = server.apply_controls
        def apply(data):
            applied.append(data.copy())
            if len(applied) == 1:
                started.set()
                if not release.wait(3):
                    raise ValueError('test worker timed out')
            real_apply(data)
        with patch.multiple(server, is_world_backend=True, world_backend=backend,
                            pipeline_ready=True, current_mode='world', output_width=1280,
                            output_height=720, active_client=None, access_token=''), \
                patch.object(server, 'get_gpu_telemetry', return_value=('--', '--')), \
                patch.object(server, 'apply_controls', side_effect=apply):
            task = asyncio.create_task(server.websocket_stream_endpoint(ws))
            try:
                await until(lambda: ws.text)
                await ws.incoming.put({'text': json.dumps({'type': 'control', 'world_mouse_x': .1})})
                await until(started.is_set)
                await ws.incoming.put({'text': json.dumps({'type': 'control', 'world_reseed': True})})
                await ws.incoming.put({'text': json.dumps({'type': 'control', 'world_reseed': False,
                                                          'world_mouse_x': .2})})
                await until(ws.incoming.empty)
                release.set()
                await until(lambda: len(applied) == 2)
                self.assertTrue(applied[-1]['world_reseed'])
                self.assertEqual(applied[-1]['world_mouse_x'], .2)
            finally:
                release.set()
                await ws.close()
                await asyncio.wait_for(task, 3)

    async def test_no_background_reseed_without_credit_and_auto_off_control_stays_responsive(self):
        backend, ws = hybrid(), FakeWebSocket()
        with patch.multiple(server, is_world_backend=True, world_backend=backend,
                            pipeline_ready=True, current_mode='world', output_width=1280,
                            output_height=720, active_client=None, access_token=''), \
                patch.object(server, 'get_gpu_telemetry', return_value=('--', '--')):
            task = asyncio.create_task(server.websocket_stream_endpoint(ws))
            try:
                await until(lambda: ws.text)
                self.assertTrue(ws.text[0]['world_prompt_seeding'])
                await asyncio.sleep(.03)
                self.assertEqual(backend.seed_count, 0)  # wait for protocol negotiation
                await ws.incoming.put({'text': json.dumps({'type': 'client_options',
                                                          'frame_ack': True, 'frame_window': 1})})
                for count in range(1, 5):
                    await until(lambda: len({p[0] for p in ws.binary}) == count)
                    if count < 4:
                        await ws.incoming.put({'text': json.dumps({'type': 'frame_ack', 'frame_id': count % 256})})
                self.assertEqual(len(backend.frames), 0)
                backend._seed_deadline = 0
                await asyncio.sleep(.05)
                self.assertEqual(backend.seed_count, 1)
                await ws.incoming.put({'text': json.dumps({'type': 'control', 'world_auto_seed': False})})
                await until(lambda: not backend.auto_seed)
                await ws.incoming.put({'text': json.dumps({'type': 'frame_ack', 'frame_id': 4})})
                await until(lambda: len({p[0] for p in ws.binary}) == 5)
                self.assertEqual(backend.seed_count, 1)
            finally:
                await ws.close()
                await asyncio.wait_for(task, 3)
