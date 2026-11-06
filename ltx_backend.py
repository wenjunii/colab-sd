"""LTX-2.5 clips and opt-in continuous generation with bounded frame storage."""
from collections import deque
import logging
import os
from pathlib import Path
import threading
import tempfile
import time

import cv2
import numpy as np
from bridge_protocol import integer
from ltx_runtime import CachedPromptEncoder

logger = logging.getLogger(__name__)
SIZES = ((512, 512), (768, 512), (1024, 576), (1024, 768))
DEFAULT_SIZE = (1024, 576)
FRAME_COUNTS = (17, 33, 65, 121)


def startup_size():
    size = tuple(integer(os.environ.get('BRIDGE_INITIAL_' + key, str(default)), key)
                 for key, default in zip(('WIDTH', 'HEIGHT'), DEFAULT_SIZE))
    if size not in SIZES:
        raise ValueError('LTX output must be 512x512, 768x512, 1024x576 or 1024x768')
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
    for name in ('ltx_generate', 'ltx_paused', 'ltx_continuous', 'ltx_continue_scene'):
        if name in data:
            if not isinstance(data[name], bool):
                raise ValueError(name + ' must be a boolean')
            clean[name] = data[name]
    if 'ltx_refresh_every' in data:
        value = integer(data['ltx_refresh_every'], 'ltx_refresh_every')
        if not 0 <= value <= 32:
            raise ValueError('LTX refresh interval must be 0 to 32 clips')
        clean['ltx_refresh_every'] = value
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

    def __init__(self, pipe, torch_module, size=(512, 512), num_frames=33, image=None,
                 continuous=False, continue_scene=False, image_factory=None, workspace=None, profile='low_memory',
                 refresh_every=0):
        if size not in SIZES or isinstance(num_frames, bool) or num_frames not in FRAME_COUNTS:
            raise ValueError('Choose an LTX size and clip length preset')
        self.pipe, self.torch, self.size, self.num_frames = pipe, torch_module, size, num_frames
        self.image = image
        self.continuous, self.continue_scene = continuous, continue_scene
        self.refresh_every = validate_ltx_controls({'ltx_refresh_every': refresh_every}, size)['ltx_refresh_every']
        self._last_refresh = False
        self.image_factory, self.workspace, self.profile = image_factory, workspace, profile
        self.running, self.sequence_index, self.last_seed = False, 0, 42
        self._last_rgb, self._carry_directory = None, None
        self.engine = ('LTX-2.5 distilled (FP8 resident / cached prompt)' if profile == 'resident'
                       else 'LTX-2.5 distilled (FP8-cast / CPU offload / cached prompt)')
        self.prompt, self.seed = '', 42
        self.frames = deque()
        self._prefetched = deque()
        self._rendering, self._primed = False, False
        self.revision, self._applied_revision = 0, -1
        self._warm_seconds = deque(maxlen=4)
        self.playback_fps = 24.0
        self.requested, self.paused, self.failed = False, False, False
        self.clip_index, self.batch_ms = 0, 0.0
        self.phase, self.phase_started = 'awaiting_prompt', time.monotonic()
        self._lock = threading.RLock()

    @classmethod
    def load(cls, root):
        import torch
        from ltx_assets import download_models, TRANSFORMER, TEXT_ENCODER, VIDEO_VAE, AUDIO_VAE, UPSCALER
        from ltx_pipelines import DistilledPipeline
        from ltx_pipelines.utils.model_paths import ModelPaths
        from ltx_pipelines.utils.types import OffloadMode, ImageConditioningInput
        from ltx_core.quantization.fp8_cast import build_policy
        from ltx_runtime import optimize_pipeline, PROFILES
        gpu_preflight()
        size = startup_size()
        frames = integer(os.environ.get('LTX_CLIP_FRAMES', '33'), 'LTX_CLIP_FRAMES')
        if frames not in FRAME_COUNTS:
            raise ValueError('LTX_CLIP_FRAMES must be 17, 33, 65 or 121')
        profile = os.environ.get('LTX_PERFORMANCE_PROFILE', 'resident')
        if profile not in PROFILES:
            raise ValueError('LTX_PERFORMANCE_PROFILE must be resident or low_memory')
        image = os.environ.get('LTX_INPUT_IMAGE', '').strip() or None
        if image:
            image = str(Path(image).expanduser().resolve(strict=True))
            if cv2.imread(image) is None:
                raise ValueError('LTX_INPUT_IMAGE must be a readable image file')
            image = ImageConditioningInput.at(image, 0, 0.8, crf=0)
        models = download_models(root)
        paths = ModelPaths.from_split(transformer_path=str(models / TRANSFORMER),
            text_encoder_path=str(models / TEXT_ENCODER), video_vae_path=str(models / VIDEO_VAE),
            audio_vae_path=str(models / AUDIO_VAE))
        quantization = build_policy(str(models / TRANSFORMER))
        pipe = DistilledPipeline(model_paths=paths, spatial_upsampler_path=str(models / UPSCALER),
            loras=[], device=torch.device('cuda'), quantization=quantization,
            offload_mode=OffloadMode.CPU)
        optimize_pipeline(pipe, paths, quantization, profile)
        logger.info('LTX pipeline constructed; GPU weights load lazily on Generate Clip')
        return cls(pipe, torch, size, frames, image,
            continuous=os.environ.get('LTX_CONTINUOUS', '1') == '1',
            continue_scene=os.environ.get('LTX_CONTINUE_SCENE', '1') == '1',
            image_factory=ImageConditioningInput.at, workspace=Path(root), profile=profile,
            refresh_every=integer(os.environ.get('LTX_REFRESH_EVERY', '4'), 'LTX_REFRESH_EVERY'))

    def _phase(self, name):
        with self._lock:
            self.phase, self.phase_started = name, time.monotonic()

    @property
    def can_generate(self):
        with self._lock:
            return not self.paused and not self.failed and (bool(self.frames) or self.requested
                                                           or (self.continuous and self.running))

    @property
    def needs_render(self):
        with self._lock:
            return (not self.paused and not self.failed and not self._rendering
                    and not self._prefetched and (self.requested or (self.continuous and self.running)))

    @property
    def can_deliver(self):
        with self._lock:
            return (not self.paused and not self.failed and bool(self.frames)
                    and (self._primed or not self.continuous))

    def configure(self, data):
        clean = validate_ltx_controls(data, self.size)
        with self._lock:
            changed = (clean.get('prompt_a', self.prompt), clean.get('seed', self.seed)) != (self.prompt, self.seed)
            self.prompt, self.seed = clean.get('prompt_a', self.prompt), clean.get('seed', self.seed)
            self.paused = clean.get('ltx_paused', self.paused)
            if self.continuous and clean.get('ltx_continuous') is False:
                self._prefetched.clear()
            self.continuous = clean.get('ltx_continuous', self.continuous)
            self.continue_scene = clean.get('ltx_continue_scene', self.continue_scene)
            self.refresh_every = clean.get('ltx_refresh_every', self.refresh_every)
            restart = clean.get('ltx_generate', False)
            if restart and not self.prompt:
                raise ValueError('Submit Prompt A before generating a clip')
            if changed or restart:
                # An in-flight render owns its snapshot. Never let it refill a
                # new prompt's queues or restore its old continuation image.
                self.revision += 1
                self.frames.clear()
                self._prefetched.clear()
                self.sequence_index, self._last_rgb, self._primed = 0, None, False
                self._warm_seconds.clear()
                if restart:
                    self.running, self.failed = True, False
                self.requested = self.running and not self.failed
            self._refresh_phase()
            return changed or restart

    def _refresh_phase(self):
        name = ('error' if self.failed else 'paused' if self.paused else
                'buffering' if self.frames and self.continuous and not self._primed else
                'streaming' if self.frames else 'generating' if self._rendering else
                'next_clip' if self.requested or (self.continuous and self.running) else 'awaiting_prompt')
        if self.phase != name:
            self._phase(name)

    def release_controls(self):
        with self._lock:
            self.revision += 1
            self.frames.clear()
            self._prefetched.clear()
            self.requested = False
            self.running, self._last_rgb, self._primed = False, None, False
            if not self._rendering:
                self._cleanup_carry()
            self._phase('awaiting_prompt')

    def _cleanup_carry(self):
        if self._carry_directory is not None:
            self._carry_directory.cleanup()
            self._carry_directory = None

    def close(self):
        self.release_controls()
        if hasattr(self.pipe, 'prompt_encoder'):
            self.pipe.prompt_encoder.clear()
        if self.profile == 'resident':
            self.pipe.stage.release_transformer()

    def _clip_image(self):
        self._last_refresh = bool(self.sequence_index and self.refresh_every
                                  and self.sequence_index % self.refresh_every == 0)
        if self._last_refresh:
            return self.image
        if not (self.continue_scene and self._last_rgb is not None and self.image_factory):
            return self.image
        if self._carry_directory is None:
            self._carry_directory = tempfile.TemporaryDirectory(prefix='ltx-continuation-', dir=self.workspace)
        path = Path(self._carry_directory.name) / 'last-frame.png'
        if not cv2.imwrite(str(path), cv2.cvtColor(self._last_rgb, cv2.COLOR_RGB2BGR)):
            raise ValueError('Could not write the owned LTX continuation image')
        # Feedback frames already passed through the VAE. Avoid introducing
        # upstream's default lossy H.264 re-compression on every continuation.
        return self.image_factory(str(path), 0, 0.8, crf=0)

    def progress(self):
        with self._lock:
            return self._progress()

    def _progress(self):
        phase, elapsed = self.phase, time.monotonic() - self.phase_started
        labels = {'awaiting_prompt': 'LTX ready; Submit Prompt A, then Generate Clip on the LTX page',
                  'generating': 'Generating LTX clip; holding last frame',
                  'buffering': 'Preparing LTX lookahead and measuring sustainable playback; holding last frame',
                  'next_clip': 'Continuous LTX: preparing the next clip; holding last frame',
                  'streaming': 'Playing generated LTX clip', 'paused': 'LTX paused',
                  'error': 'LTX clip failed; inspect logs, then Generate Clip to retry'}
        encoder = getattr(self.pipe, 'prompt_encoder', None)
        cached = isinstance(encoder, CachedPromptEncoder)
        return {'world_phase': phase, 'world_status': labels[phase] + (f' ({elapsed:.0f}s)' if phase == 'generating' else ''),
                'ltx_clip': self.clip_index, 'ltx_clip_frames': self.num_frames,
                'ltx_queued_frames': len(self.frames) + len(self._prefetched),
                'ltx_paused': self.paused, 'ltx_input_image': bool(self.image),
                'ltx_continuous': self.continuous, 'ltx_continue_scene': self.continue_scene,
                'ltx_generation_ms': round(self.batch_ms, 1), 'ltx_profile': self.profile,
                'ltx_generated_fps': round(self.num_frames * 1000 / self.batch_ms, 3) if self.batch_ms else 0,
                'ltx_seed': self.last_seed,
                'ltx_playback_fps': round(self.playback_fps, 3), 'ltx_rendering': self._rendering,
                'ltx_prompt_pending': self.running and self._applied_revision != self.revision,
                'ltx_prompt_revision': self.revision,
                'ltx_refresh_every': self.refresh_every, 'ltx_quality_refresh': self._last_refresh,
                'ltx_prompt_cache_hit': encoder.cache_hit if cached else False,
                'ltx_prompt_ms': round(encoder.elapsed_ms, 1) if cached else 0}

    def render_next_clip(self):
        """GPU-worker operation; delivery and control updates never wait on it."""
        with self._lock:
            if not self.needs_render:
                return False
            automatic = not self.requested
            self.requested, self._rendering = False, True
            revision, prompt = self.revision, self.prompt
            sequence = self.sequence_index
            seed = (self.seed + sequence) % 4294967296
            self._refresh_phase()
        started = time.perf_counter()
        frames = deque()
        try:
            with self._lock:
                image = self._clip_image()
            with self.torch.inference_mode():
                result = self.pipe(prompt=prompt, seed=seed, height=self.size[1], width=self.size[0],
                    frame_rate=24.0, num_frames=self.num_frames, images=[image] if image else [],
                    enhance_prompt=False)
                for chunk in result.video:
                    if hasattr(chunk, 'detach'):
                        pixels = chunk.detach().cpu()
                        # NumPy cannot represent BF16 decoder output directly.
                        batch = (pixels.float() if pixels.is_floating_point() else pixels).numpy()
                    else:
                        batch = np.asarray(chunk)
                    if (batch.ndim != 4 or batch.shape[1:] != (self.size[1], self.size[0], 3)
                            or not len(batch) or len(frames) + len(batch) > self.num_frames
                            or not np.isfinite(batch).all()):
                        raise ValueError('LTX returned invalid or excessive video frames')
                    if np.issubdtype(batch.dtype, np.floating):
                        batch = (np.clip(batch, 0, 1) * 255).round().astype(np.uint8)
                    elif batch.dtype != np.uint8:
                        raise ValueError('LTX returned an unsupported pixel dtype')
                    frames.extend(np.ascontiguousarray(batch))
                if len(frames) != self.num_frames:
                    raise ValueError('LTX returned an unexpected clip length')
                del result
            with self._lock:
                if revision != self.revision or not self.running or (automatic and not self.continuous):
                    return False
                if self._prefetched:
                    raise RuntimeError('LTX lookahead queue is already full')
                if self.frames:
                    self._prefetched = frames
                else:
                    self.frames = frames
                self._last_rgb = frames[-1].copy()
                self.batch_ms = (time.perf_counter() - started) * 1000
                self.clip_index += 1
                self.sequence_index += 1
                self.last_seed, self._applied_revision = seed, revision
                # First render includes cold loading and text encoding. Use a
                # separate warm lookahead sample to avoid pacing from that cost.
                if sequence > 0:
                    self._warm_seconds.append(self.batch_ms / 1000)
                    self.playback_fps = min(24.0, self.num_frames / (max(max(self._warm_seconds), 0.001) * 1.2))
                    self._primed = True
                logger.info('LTX clip %d: %d native frames in %.2fs (%.2f generated FPS)', self.clip_index,
                            self.num_frames, self.batch_ms / 1000, self.num_frames * 1000 / self.batch_ms)
                return True
        except Exception as exc:
            if self.profile == 'resident':
                self.pipe.stage.release_transformer()
            with self._lock:
                if revision != self.revision:
                    logger.warning('Discarded failed obsolete LTX render: %s', type(exc).__name__)
                    return False
                self.frames.clear()
                self._prefetched.clear()
                self.failed = True
            raise ValueError('LTX clip failed; inspect server logs before retrying') from exc
        finally:
            with self._lock:
                self._rendering = False
                if not self.running:
                    self._cleanup_carry()
                self._refresh_phase()

    def pop_frame(self, require_primed=True):
        """CPU-only, nonblocking delivery while another clip renders on GPU."""
        with self._lock:
            if not self.can_generate or not self.frames or (require_primed and not self.can_deliver):
                return None
            rgb, duration = self.frames.popleft(), self.batch_ms
            if not self.frames and self._prefetched:
                self.frames, self._prefetched = self._prefetched, deque()
            self._refresh_phase()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), duration

    def next_frame(self):
        """Blocking single-thread benchmark path; server uses render/pop separately."""
        if not self.can_generate:
            return None
        if not self.frames:
            self.render_next_clip()
        return self.pop_frame(require_primed=False)
