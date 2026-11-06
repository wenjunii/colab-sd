"""Bounded Krea T2V chunks through the separately downloaded HF modular runtime."""
from collections import deque
import logging
import threading
import time

import cv2
import numpy as np

from bridge_protocol import integer

KREA_SIZE = (832, 480)
MAX_CHUNK_FRAMES = 12
logger = logging.getLogger(__name__)


def validate_krea_controls(data):
    if not isinstance(data, dict):
        raise ValueError('Krea control must be an object')
    if data.get('mode', 'krea') != 'krea':
        raise ValueError('Select Krea Realtime for this server')
    if (integer(data.get('width', 832), 'width'), integer(data.get('height', 480), 'height')) != KREA_SIZE:
        raise ValueError('This Krea trial uses fixed 832 x 480 output')
    clean = {}
    if 'prompt_a' in data:
        text = data['prompt_a']
        if not isinstance(text, str) or not text.strip() or len(text) > 4096:
            raise ValueError('Prompt A must contain between 1 and 4096 characters')
        clean['prompt_a'] = text.strip()
    if 'seed' in data:
        seed = integer(data['seed'], 'seed')
        if not 0 <= seed <= 4294967295:
            raise ValueError('Seed must be between 0 and 4294967295')
        clean['seed'] = seed
    for name in ('krea_paused', 'krea_reset'):
        if name in data:
            if not isinstance(data[name], bool):
                raise ValueError(f'{name} must be a boolean')
            clean[name] = data[name]
    return clean


def gpu_preflight(torch_module):
    if not torch_module.cuda.is_available() or torch_module.cuda.get_device_capability()[0] < 8:
        raise RuntimeError('Krea requires an Ampere-or-newer NVIDIA CUDA GPU')
    free, total = torch_module.cuda.mem_get_info()
    if total < 35 * 1024 ** 3 or free < 30 * 1024 ** 3:
        raise RuntimeError('This Krea 14B trial requires at least 35 GiB total / 30 GiB free VRAM. '
                           'Use a fresh A100 40 GB or larger runtime; L4/T4 are not enabled here.')
    return {'gpu': torch_module.cuda.get_device_name(), 'vram_gib': round(total / 1024 ** 3, 1)}


def require_components(pipe, names):
    # Modular Diffusers logs some component failures and leaves None behind.
    # Never advertise readiness after a required component failed to load.
    missing = [name for name in names if getattr(pipe, name, None) is None]
    if missing:
        raise RuntimeError('Krea required component(s) failed to load: ' + ', '.join(missing))


def check_tokenizer(pipe):
    require_components(pipe, ('tokenizer', 'scheduler'))
    try:
        tokens = pipe.tokenizer(['a cinematic landscape'], padding='max_length', truncation=True,
                                max_length=8, return_tensors='pt')
        if tuple(tokens['input_ids'].shape) != (1, 8):
            raise ValueError('tokenizer returned an invalid token shape')
    except Exception as exc:
        raise RuntimeError('Krea tokenizer readiness check failed; inspect tokenizer cache files') from exc


class KreaBackend:
    def __init__(self, pipe, state_factory, torch_module, clip_blocks=64, int8_profile='balanced'):
        if isinstance(clip_blocks, bool) or not isinstance(clip_blocks, int) or not 1 <= clip_blocks <= 64:
            raise ValueError('Krea clip blocks must be between 1 and 64')
        self.pipe, self.state_factory, self.torch = pipe, state_factory, torch_module
        from krea_quant import validate_int8_profile
        self.int8_profile = validate_int8_profile(int8_profile)
        self.clip_blocks = clip_blocks
        self.frames = deque()
        self.prompt, self.seed = '', 42
        self.paused, self.started, self.failed = False, False, False
        self.block_index, self.clip_index, self.batch_ms = 0, 0, 0.0
        self._prompt_changed = True
        self._clear_vae_cache = pipe.vae.clear_cache
        self.state = state_factory()
        self.generator = None
        self._lock = threading.Lock()
        self._phase, self._phase_started = 'awaiting_prompt', time.monotonic()

    def _set_phase(self, value):
        with self._lock:
            if value != self._phase:
                self._phase, self._phase_started = value, time.monotonic()

    def progress(self):
        with self._lock:
            phase, elapsed = self._phase, time.monotonic() - self._phase_started
        labels = {'awaiting_prompt': 'Connected; submit Prompt A to start Krea',
                  'generating': 'Generating Krea chunk; first use may compile kernels',
                  'streaming': 'Krea frames ready', 'paused': 'Krea paused',
                  'error': 'Krea generation failed; inspect server log, then Reset Krea Clip'}
        status = labels[phase] + (f' ({elapsed:.0f}s)' if phase == 'generating' else '')
        # Shared stateful-generation progress fields understood by the TD bridge.
        return {'world_phase': phase, 'world_status': status,
                'krea_block': self.block_index, 'krea_clip': self.clip_index,
                'krea_clip_blocks': self.clip_blocks, 'krea_paused': self.paused,
                'krea_int8_profile': self.int8_profile}

    @property
    def engine(self):
        from krea_quant import int8_engine_label
        return int8_engine_label(self.int8_profile)

    @classmethod
    def load(cls, root, clip_blocks=64):
        import os
        import torch
        from krea_quant import INT8_THRESHOLDS, validate_int8_profile, limit_cast_warnings
        profile = validate_int8_profile(os.environ.get('KREA_INT8_PROFILE', 'fast'))
        limit_cast_warnings()
        gpu_preflight(torch)
        # This server uses only PyTorch attention and the pinned HF source.
        os.environ['DISABLE_SAGEATTENTION'] = '1'
        os.environ['DIFFUSERS_ENABLE_HUB_KERNELS'] = '0'
        from accelerate import cpu_offload
        from diffusers import AutoModel, BitsAndBytesConfig, ModularPipeline
        from diffusers.modular_pipelines import PipelineState
        from krea_assets import download_assets
        snapshot, pipeline_folder = download_assets(root)
        pipe = ModularPipeline.from_pretrained(str(pipeline_folder), trust_remote_code=True)
        pipe.load_components(names=['tokenizer', 'scheduler'], trust_remote_code=False, local_files_only=True)
        check_tokenizer(pipe)
        transformer = AutoModel.from_pretrained(
            str(snapshot), subfolder='transformer', trust_remote_code=True,
            local_files_only=True, dtype=torch.bfloat16, use_safetensors=True,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True,
                llm_int8_threshold=INT8_THRESHOLDS[profile]), device_map='cuda')
        if not getattr(transformer, 'is_loaded_in_8bit', False):
            raise RuntimeError('Requested Krea INT8 transformer did not activate')
        from krea_quant import prepare_int8_linears
        count = prepare_int8_linears(transformer)
        logger.info('Prepared %d INT8 linears for Krea n-dimensional inputs', count)
        pipe.load_components(
            names=['text_encoder', 'vae'],
            trust_remote_code=False, local_files_only=True,
            dtype={'text_encoder': torch.bfloat16, 'vae': torch.float16},
            device_map={'text_encoder': 'cpu', 'vae': 'cuda'},
            use_safetensors={'text_encoder': True, 'vae': True})
        pipe.update_components(transformer=transformer)
        require_components(pipe, ('transformer', 'text_encoder', 'tokenizer', 'vae', 'scheduler'))
        from krea_quant import prepare_vae_layout
        prepare_vae_layout(pipe.vae, profile)
        cpu_offload(pipe.text_encoder, execution_device=torch.device('cuda'), offload_buffers=True)
        # Projection fusion is intentionally absent: it can replace INT8 linears.
        logger.info('Krea INT8 profile: %s (outlier threshold %s)', profile, INT8_THRESHOLDS[profile])
        return cls(pipe, PipelineState, torch, clip_blocks, int8_profile=profile)

    @property
    def can_generate(self):
        return self.started and not self.paused and not self.failed

    def configure(self, data):
        clean = validate_krea_controls(data)
        prompt = clean.get('prompt_a', self.prompt)
        if prompt != self.prompt:
            self.prompt, self._prompt_changed = prompt, True
        seed_changed = clean.get('seed', self.seed) != self.seed
        self.seed = clean.get('seed', self.seed)
        self.paused = clean.get('krea_paused', self.paused)
        if seed_changed or clean.get('krea_reset'):
            self.reset()
        self.started = bool(self.prompt)
        if not self.failed:
            self._set_phase('paused' if self.paused else 'streaming' if self.block_index else 'awaiting_prompt')

    def reset(self, advance=False):
        self.frames.clear()
        # The pinned decoder replaces clear_cache with a no-op after block zero.
        # Restore the saved bound method before starting another clip.
        self.pipe.vae.clear_cache = self._clear_vae_cache
        self._clear_vae_cache()
        self.state = self.state_factory()
        self.block_index, self.clip_index = 0, self.clip_index + 1 if advance else 0
        self.generator, self._prompt_changed, self.failed = None, True, False
        self._set_phase('paused' if self.paused else 'awaiting_prompt')

    def release_controls(self):
        self.frames.clear()
        self.started = False  # wait for the next controller's prompt/configuration
        self._set_phase('awaiting_prompt')

    def next_frame(self):
        if not self.can_generate:
            return None
        if not self.frames:
            if self.block_index >= self.clip_blocks:
                self.reset(advance=True)
            self._set_phase('generating')
            started = time.perf_counter()
            try:
                if self.generator is None:
                    self.generator = self.torch.Generator(device='cuda').manual_seed(
                        (self.seed + self.clip_index) % 4294967296)
                with self.torch.inference_mode():
                    self.state = self.pipe(
                        self.state, prompt=[self.prompt] if self._prompt_changed else None,
                        prompt_embeds=None if self._prompt_changed else self.state.values.get('prompt_embeds'),
                        update_prompt_embeds=self._prompt_changed, num_inference_steps=4,
                        num_blocks=self.clip_blocks, block_idx=self.block_index,
                        width=832, height=480, output_type='np', generator=self.generator)
                batch = np.asarray(self.state.values['videos'][0])
                if (batch.ndim != 4 or batch.shape[1:] != (480, 832, 3)
                        or not 1 <= len(batch) <= MAX_CHUNK_FRAMES or not np.isfinite(batch).all()):
                    raise ValueError('Krea returned an invalid video chunk')
                if np.issubdtype(batch.dtype, np.floating):
                    batch = (np.clip(batch, 0, 1) * 255).round().astype(np.uint8)
                elif batch.dtype != np.uint8:
                    raise ValueError('Krea returned an invalid pixel type')
                self.frames.extend(batch)
                self.state.values['videos'] = None
                self.block_index += 1
                self._prompt_changed = False
                self.batch_ms = (time.perf_counter() - started) * 1000
                self._set_phase('streaming')
            except Exception as exc:
                self.frames.clear()
                self.failed = True
                self._set_phase('error')
                raise ValueError('Krea chunk failed; inspect log and reset the clip') from exc
        return cv2.cvtColor(self.frames.popleft(), cv2.COLOR_RGB2BGR), self.batch_ms
