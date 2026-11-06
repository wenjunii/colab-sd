"""Bounded prompt caching and the pinned LTX transformer's warm-weight API."""
import time

PROFILES = ('resident', 'low_memory')


class CachedPromptEncoder:
    """Retain one encoded prompt, without retaining the large text encoder."""
    def __init__(self, encoder, on_miss=None):
        self.encoder, self.on_miss = encoder, on_miss
        self.key, self.value = None, None
        self.cache_hit, self.elapsed_ms = False, 0.0

    def __getattr__(self, name):
        return getattr(self.encoder, name)

    def clear(self):
        self.key, self.value = None, None

    def __call__(self, prompts, **kwargs):
        options = dict(kwargs)
        if not options.get('enhance_first_prompt', False):
            # Image conditioning and diffusion seed do not affect plain text encoding.
            options.pop('enhance_prompt_image', None)
            options.pop('enhance_prompt_seed', None)
        key = (tuple(prompts), tuple(sorted(options.items())))
        started = time.perf_counter()
        self.cache_hit = self.key == key and self.value is not None
        if not self.cache_hit:
            self.clear()
            if self.on_miss:
                self.on_miss()
            value = self.encoder(prompts, **kwargs)
            self.key, self.value = key, value
        self.elapsed_ms = (time.perf_counter() - started) * 1000
        return self.value


def optimize_pipeline(pipe, paths, quantization, profile):
    if profile not in PROFILES:
        raise ValueError('LTX performance profile must be resident or low_memory')
    release = None
    if profile == 'resident':
        from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
        from ltx_core.loader.registry import ModelRegistry
        from ltx_pipelines.utils.blocks import DiffusionStage
        from ltx_pipelines.utils.types import OffloadMode
        # Upstream's shared allocation reuses the quantized transformer between
        # both denoising stages and subsequent clips. Gemma stays CPU-offloaded.
        pipe.stage = DiffusionStage.from_checkpoint(paths.transformer(), pipe.dtype, pipe.device,
            quantization=quantization, registry=ModelRegistry(), active_weights={},
            offload_mode=OffloadMode.NONE, alloc_trim_strategy=AllocatorTrimStrategy.DEFER)
        # A changed prompt can load a large encoder: release DiT first on 40 GB.
        release = pipe.stage.release_transformer
    pipe.prompt_encoder = CachedPromptEncoder(pipe.prompt_encoder, on_miss=release)
    return pipe
