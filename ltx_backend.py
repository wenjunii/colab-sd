"""Short, explicitly requested LTX-2.5 clips with bounded CPU frame storage."""
from collections import deque
import logging
import os
from pathlib import Path
import threading
import time

import cv2
import numpy as np
from bridge_protocol import integer

logger = logging.getLogger(__name__)
SIZES = ((512, 512), (768, 512), (1024, 768))
FRAME_COUNTS = (17, 33, 65, 121)


def startup_size():
    size = tuple(integer(os.environ.get('BRIDGE_INITIAL_' + key, '512'), key) for key in ('WIDTH', 'HEIGHT'))
    if size not in SIZES:
        raise ValueError('LTX output must be 512x512, 768x512 or 1024x768')
    return size


def validate_ltx_controls(data, size):
    if not isinstance(data, dict) or data.get('mode', 'ltx') != 'ltx':
        raise ValueError('Select LTX-2.5 for this server')
    if tuple(integer(data.get(key, n), key) for key, n in zip(('width', 'height'), size)) != size:
        raise ValueError('LTX output size is locked to the startup preset')
    clean = {}
    if 'prompt_a' in data:
        prompt = data['prompt_a']
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4096:
            raise ValueError('Prompt A must contain 1 to 4096 characters')
        clean['prompt_a'] = prompt.strip()
    if 'seed' in data:
        seed = integer(data['seed'], 'seed')
        if not 0 <= seed <= 4294967295:
            raise ValueError('Seed must be between 0 and 4294967295')
        clean['seed'] = seed
    for name in ('ltx_generate', 'ltx_paused'):
        if name in data:
            if not isinstance(data[name], bool):
                raise ValueError(name + ' must be a boolean')
            clean[name] = data[name]
    return clean


def gpu_preflight():
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError('LTX trial requires an Ampere-or-newer NVIDIA CUDA GPU')
    free, total = torch.cuda.mem_get_info()
    if total < 35 * 1024**3 or free < 30 * 1024**3:
        raise RuntimeError('Use an A100 40 GB+ with at least 30 GiB free VRAM for this LTX trial')
    return {'gpu': torch.cuda.get_device_name(), 'vram_gib': round(total / 1024**3, 1)}


class LTXBackend:
    engine = 'LTX-2.5 distilled (FP8-cast / CPU offload)'

    def __init__(self, pipe, torch_module, size=(512, 512), num_frames=33, image=None):
        if size not in SIZES or isinstance(num_frames, bool) or num_frames not in FRAME_COUNTS:
            raise ValueError('Choose an LTX size and clip length preset')
        self.pipe, self.torch, self.size, self.num_frames = pipe, torch_module, size, num_frames
        self.image = image
        self.prompt, self.seed = '', 42
        self.frames = deque()
        self.requested, self.paused, self.failed = False, False, False
        self.clip_index, self.batch_ms = 0, 0.0
        self.phase, self.phase_started = 'awaiting_prompt', time.monotonic()
        self._lock = threading.Lock()

    @classmethod
    def load(cls, root):
        import torch
        from ltx_assets import download_models, TRANSFORMER, TEXT_ENCODER, VIDEO_VAE, AUDIO_VAE, UPSCALER
        from ltx_pipelines import DistilledPipeline
        from ltx_pipelines.utils.model_paths import ModelPaths
        from ltx_pipelines.utils.types import OffloadMode, ImageConditioningInput
        from ltx_core.quantization.fp8_cast import build_policy
        gpu_preflight()
        size = startup_size()
        frames = integer(os.environ.get('LTX_CLIP_FRAMES', '33'), 'LTX_CLIP_FRAMES')
        if frames not in FRAME_COUNTS:
            raise ValueError('LTX_CLIP_FRAMES must be 17, 33, 65 or 121')
        image = os.environ.get('LTX_INPUT_IMAGE', '').strip() or None
        if image:
            image = str(Path(image).expanduser().resolve(strict=True))
            if cv2.imread(image) is None:
                raise ValueError('LTX_INPUT_IMAGE must be a readable image file')
            image = ImageConditioningInput.at(image, 0, 0.8)
        models = download_models(root)
        paths = ModelPaths.from_split(transformer_path=str(models / TRANSFORMER),
            text_encoder_path=str(models / TEXT_ENCODER), video_vae_path=str(models / VIDEO_VAE),
            audio_vae_path=str(models / AUDIO_VAE))
        pipe = DistilledPipeline(model_paths=paths, spatial_upsampler_path=str(models / UPSCALER),
            loras=[], device=torch.device('cuda'), quantization=build_policy(str(models / TRANSFORMER)),
            offload_mode=OffloadMode.CPU)
        logger.info('LTX pipeline constructed; GPU weights load lazily on Generate Clip')
        return cls(pipe, torch, size, frames, image)

    def _phase(self, name):
        with self._lock:
            self.phase, self.phase_started = name, time.monotonic()

    @property
    def can_generate(self):
        return not self.paused and not self.failed and (bool(self.frames) or self.requested)

    def configure(self, data):
        clean = validate_ltx_controls(data, self.size)
        self.prompt, self.seed = clean.get('prompt_a', self.prompt), clean.get('seed', self.seed)
        self.paused = clean.get('ltx_paused', self.paused)
        if clean.get('ltx_generate'):
            if not self.prompt:
                raise ValueError('Submit Prompt A before generating a clip')
            self.frames.clear()
            self.requested, self.failed = True, False
        if self.phase != 'generating' and not self.failed:
            self._phase('paused' if self.paused else 'streaming' if self.frames else 'awaiting_prompt')

    def release_controls(self):
        self.frames.clear()
        self.requested = False
        self._phase('awaiting_prompt')

    def progress(self):
        with self._lock:
            phase, elapsed = self.phase, time.monotonic() - self.phase_started
        labels = {'awaiting_prompt': 'LTX ready; Submit Prompt A, then Generate Clip on the LTX page',
                  'generating': 'Generating LTX clip (first use loads weights)',
                  'streaming': 'Playing generated LTX clip', 'paused': 'LTX paused',
                  'error': 'LTX clip failed; inspect logs, then Generate Clip to retry'}
        return {'world_phase': phase, 'world_status': labels[phase] + (f' ({elapsed:.0f}s)' if phase == 'generating' else ''),
                'ltx_clip': self.clip_index, 'ltx_clip_frames': self.num_frames, 'ltx_queued_frames': len(self.frames),
                'ltx_paused': self.paused, 'ltx_input_image': bool(self.image),
                'ltx_generation_ms': round(self.batch_ms, 1)}

    def next_frame(self):
        if not self.can_generate:
            return None
        if not self.frames:
            self.requested = False
            self._phase('generating')
            started = time.perf_counter()
            try:
                with self.torch.inference_mode():
                    result = self.pipe(prompt=self.prompt, seed=self.seed, height=self.size[1], width=self.size[0],
                        frame_rate=24.0, num_frames=self.num_frames, images=[self.image] if self.image else [],
                        enhance_prompt=False)
                    for chunk in result.video:
                        if hasattr(chunk, 'detach'):
                            pixels = chunk.detach().cpu()
                            # NumPy cannot represent BF16 decoder output directly.
                            batch = (pixels.float() if pixels.is_floating_point() else pixels).numpy()
                        else:
                            batch = np.asarray(chunk)
                        if (batch.ndim != 4 or batch.shape[1:] != (self.size[1], self.size[0], 3)
                                or not len(batch) or len(self.frames) + len(batch) > self.num_frames
                                or not np.isfinite(batch).all()):
                            raise ValueError('LTX returned invalid or excessive video frames')
                        if np.issubdtype(batch.dtype, np.floating):
                            batch = (np.clip(batch, 0, 1) * 255).round().astype(np.uint8)
                        elif batch.dtype != np.uint8:
                            raise ValueError('LTX returned an unsupported pixel dtype')
                        self.frames.extend(np.ascontiguousarray(batch))
                    if len(self.frames) != self.num_frames:
                        raise ValueError('LTX returned an unexpected clip length')
                    del result
                self.batch_ms = (time.perf_counter() - started) * 1000
                self.clip_index += 1
                self._phase('streaming')
                logger.info('LTX clip %d: %d native frames in %.2fs (%.2f generated FPS)', self.clip_index,
                            self.num_frames, self.batch_ms / 1000, self.num_frames * 1000 / self.batch_ms)
            except Exception as exc:
                self.frames.clear()
                self.failed = True
                self._phase('error')
                raise ValueError('LTX clip failed; inspect server logs before retrying') from exc
        frame = cv2.cvtColor(self.frames.popleft(), cv2.COLOR_RGB2BGR)
        if not self.frames:
            self._phase('awaiting_prompt')
        return frame, self.batch_ms
