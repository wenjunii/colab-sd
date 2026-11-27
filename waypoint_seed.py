"""Optional Turbo seed images, using the isolated Waypoint Diffusers stack."""
from collections import OrderedDict

import numpy as np
from colab_setup import SD_TURBO, SDXL_TURBO, model_load_source, turbo_model_files

MODEL_ID, MODEL_REVISION = SD_TURBO
XL_MODEL_ID, XL_MODEL_REVISION = SDXL_TURBO
SEED_MODELS = {'sd-turbo': (MODEL_ID, MODEL_REVISION, 'SD-Turbo'),
               'sdxl-turbo': (XL_MODEL_ID, XL_MODEL_REVISION, 'SDXL-Turbo')}
SEED_SIZE = (768, 432)


class PromptSeedGenerator:
    def __init__(self, pipe, torch_module, model='sd-turbo', device='cuda'):
        if model not in SEED_MODELS:
            raise ValueError('Prompt seed model must be sd-turbo or sdxl-turbo')
        self.pipe, self.torch = pipe, torch_module
        self.model, self.label, self.device = model, SEED_MODELS[model][2], device
        self.embeddings = OrderedDict()

    @classmethod
    def load(cls, model='sd-turbo'):
        if model not in SEED_MODELS:
            raise ValueError('Prompt seed model must be sd-turbo or sdxl-turbo')
        import torch
        if model == 'sdxl-turbo':
            from diffusers import StableDiffusionXLPipeline as pipeline_type
            options = {'add_watermarker': False}
        else:
            from diffusers import StableDiffusionPipeline as pipeline_type
            options = {'safety_checker': None}
        model_id, revision, _ = SEED_MODELS[model]
        with model_load_source(model_id, revision, turbo_model_files(xl=model == 'sdxl-turbo', fp16=True)) as source:
            pipe = pipeline_type.from_pretrained(
                source, revision=revision, variant='fp16',
                torch_dtype=torch.float16, use_safetensors=True, **options)
        pipe.to('cuda')
        pipe.set_progress_bar_config(disable=True)
        return cls(pipe, torch, model)

    def _embedding(self, prompt):
        value = self.embeddings.pop(prompt, None)
        if value is None:
            encoded = self.pipe.encode_prompt(
                prompt, device=self.device, num_images_per_prompt=1,
                do_classifier_free_guidance=False)
            value = (encoded[0], encoded[2]) if self.model == 'sdxl-turbo' else encoded[0]
        self.embeddings[prompt] = value
        while len(self.embeddings) > 2:
            self.embeddings.popitem(last=False)
        return value

    def generate(self, prompt_a, prompt_b, blend, seed):
        with self.torch.inference_mode():
            embedding = self._embedding(prompt_a)
            if prompt_b and blend > 0:
                other = self._embedding(prompt_b)
                embedding = (tuple((1 - blend) * a + blend * b for a, b in zip(embedding, other))
                             if self.model == 'sdxl-turbo' else (1 - blend) * embedding + blend * other)
            conditioning = ({'prompt_embeds': embedding[0], 'pooled_prompt_embeds': embedding[1]}
                            if self.model == 'sdxl-turbo' else {'prompt_embeds': embedding})
            output = self.pipe(
                **conditioning, num_inference_steps=1, guidance_scale=0.0,
                width=SEED_SIZE[0], height=SEED_SIZE[1], output_type='np',
                generator=self.torch.Generator(device=self.device).manual_seed(seed)).images[0]
        if output.shape != (SEED_SIZE[1], SEED_SIZE[0], 3) or not np.isfinite(output).all():
            raise ValueError(f'{self.label} returned an invalid seed image')
        return (np.clip(output, 0, 1) * 255).round().astype(np.uint8)
