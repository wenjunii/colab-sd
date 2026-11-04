"""Waypoint public-API adapter; the separately installed world_engine owns inference."""
from collections import deque
import math
import time

import cv2
import numpy as np

from bridge_protocol import integer

WORLD_SIZE = (1280, 720)
RUNTIME_REVISION = '68b316390d576dd9f671f552803526bbf8eb2ed5'
MODEL_ID = 'Overworld/Waypoint-1.5-1B'
MODEL_REVISION = '391f92827075edcf4a8b3c8a2ddae010698f8636'
VAE_ID = 'Overworld-Models/taehv1_5'
VAE_REVISION = 'a0253886b13b9c4c3bd224bd479be03f5988a3df'


def validate_world_controls(data, current=WORLD_SIZE):
    """Accept only this checkpoint's native dimensions and supported controls."""
    if not isinstance(data, dict):
        raise ValueError('Control must be an object')
    if (integer(data.get('width', current[0]), 'width'),
            integer(data.get('height', current[1]), 'height')) != WORLD_SIZE:
        raise ValueError('Waypoint uses native 1280 x 720 output')
    if data.get('mode', 'world') != 'world':
        raise ValueError('Select World Model for this server')
    clean = {}
    if 'world_buttons' in data:
        buttons = data['world_buttons']
        if not isinstance(buttons, list) or len(buttons) > 32:
            raise ValueError('Use a list of at most 32 Owl-Control button codes')
        buttons = [integer(value, 'button') for value in buttons]
        if any(not 0 <= value < 256 for value in buttons):
            raise ValueError('Button codes must be between 0 and 255')
        clean['world_buttons'] = sorted(set(buttons))
    for name in ('world_mouse_x', 'world_mouse_y'):
        if name in data:
            value = data[name]
            if (isinstance(value, bool) or not isinstance(value, (float, int))
                    or not math.isfinite(value) or not -1 <= value <= 1):
                raise ValueError('Camera motion must be between -1 and 1')
            clean[name] = float(value)
    for name in ('world_reset', 'world_paused'):
        if name in data:
            if not isinstance(data[name], bool):
                raise ValueError(f'{name} must be a boolean')
            clean[name] = data[name]
    if 'seed' in data:
        seed = integer(data['seed'], 'seed')
        if not 0 <= seed <= 4294967295:
            raise ValueError('seed must be between 0 and 4294967295')
        clean['seed'] = seed
    return clean


class WaypointBackend:
    """Single-worker state: at most one four-frame batch, no unbounded prefetch."""
    def __init__(self, engine, ctrl_type, torch_module, seed=42):
        self.engine, self.ctrl_type, self.torch = engine, ctrl_type, torch_module
        self.frames = deque()
        self.seed_image = None
        self.seed = seed
        self.buttons, self.mouse = [], [0.0, 0.0]
        self.paused = False
        self.batch_ms = 0.0

    @classmethod
    def load(cls, quant='intw8a8'):
        import torch
        from huggingface_hub import snapshot_download
        from world_engine import WorldEngine, CtrlInput
        if quant not in ('intw8a8', 'bf16'):
            raise ValueError('A100 supports intw8a8 or bf16 here; FP8/FP4 are not selected')
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
            raise RuntimeError('Waypoint requires an NVIDIA Ampere-or-newer CUDA GPU')
        model = snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                                  allow_patterns=['config.yaml', 'model.safetensors'])
        vae = snapshot_download(VAE_ID, revision=VAE_REVISION,
                                allow_patterns=['taehv1_5.pth'])
        with torch.no_grad():
            engine = WorldEngine(model, quant=None if quant == 'bf16' else quant,
                                 model_config_overrides={'ae_uri': vae},
                                 device='cuda', dtype=torch.bfloat16)
        return cls(engine, CtrlInput, torch)

    @property
    def can_generate(self):
        return self.seed_image is not None and not self.paused

    def release_controls(self):
        self.buttons, self.mouse = [], [0.0, 0.0]
        self.frames.clear()

    def configure(self, data):
        clean = validate_world_controls(data)
        if 'world_buttons' in clean:
            self.buttons = clean['world_buttons']
        for index, name in enumerate(('world_mouse_x', 'world_mouse_y')):
            if name in clean:
                self.mouse[index] = clean[name]
        if 'world_paused' in clean:
            self.paused = clean['world_paused']
        if 'seed' in clean:
            self.seed = clean['seed']
        if clean.get('world_reset'):
            self.reset()

    def reset(self):
        self.frames.clear()
        self.torch.manual_seed(self.seed)
        try:
            self.engine.reset()
            if self.seed_image is not None:
                self.engine.append_frame(self.seed_image)
        except Exception:
            # Partial cache/encoder state must not be used for further inference.
            self.seed_image = None
            raise

    def set_seed_image(self, jpeg):
        bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError('Seed must be a JPEG image')
        # Fit without stretching: center crop to native 16:9, then resize.
        height, width = bgr.shape[:2]
        crop_w, crop_h = min(width, round(height * 16 / 9)), min(height, round(width * 9 / 16))
        x, y = (width - crop_w) // 2, (height - crop_h) // 2
        rgb = cv2.cvtColor(cv2.resize(bgr[y:y+crop_h, x:x+crop_w], WORLD_SIZE), cv2.COLOR_BGR2RGB)
        image = self.torch.from_numpy(np.repeat(rgb[None], 4, axis=0))
        self.seed_image = image
        self.reset()

    def next_frame(self):
        if not self.can_generate:
            return None
        if not self.frames:
            started = time.perf_counter()
            # Values are explicitly held camera velocity, sampled once per four-frame step.
            ctrl = self.ctrl_type(button=set(self.buttons), mouse=tuple(self.mouse))
            batch = self.engine.gen_frame(ctrl=ctrl).detach().cpu().numpy()
            if batch.shape != (4, 720, 1280, 3) or batch.dtype != np.uint8:
                raise RuntimeError('Waypoint returned an unexpected frame batch')
            self.batch_ms = (time.perf_counter() - started) * 1000
            self.frames.extend(batch)
        return cv2.cvtColor(self.frames.popleft(), cv2.COLOR_RGB2BGR), self.batch_ms
