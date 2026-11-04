"""Optional SD-Turbo seed images, using the isolated Waypoint Diffusers stack."""
from collections import OrderedDict

import numpy as np

MODEL_ID = 'stabilityai/sd-turbo'
MODEL_REVISION = 'b261bac6fd2cf515557d5d0707481eafa0485ec2'
SEED_SIZE = (768, 432)


class PromptSeedGenerator:
    def __init__(self, pipe, torch_module):
        self.pipe, self.torch = pipe, torch_module
        self.embeddings = OrderedDict()

    @classmethod
    def load(cls):
        import torch
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, variant='fp16',
            torch_dtype=torch.float16, use_safetensors=True, safety_checker=None)
        pipe.to('cuda')
        pipe.set_progress_bar_config(disable=True)
        return cls(pipe, torch)

    def _embedding(self, prompt):
        value = self.embeddings.pop(prompt, None)
        if value is None:
            value = self.pipe.encode_prompt(
                prompt, device='cuda', num_images_per_prompt=1,
                do_classifier_free_guidance=False)[0]
        self.embeddings[prompt] = value
        while len(self.embeddings) > 2:
            self.embeddings.popitem(last=False)
        return value

    def generate(self, prompt_a, prompt_b, blend, seed):
        with self.torch.inference_mode():
            embedding = self._embedding(prompt_a)
            if prompt_b and blend > 0:
                embedding = (1 - blend) * embedding + blend * self._embedding(prompt_b)
            output = self.pipe(
                prompt_embeds=embedding, num_inference_steps=1, guidance_scale=0.0,
                width=SEED_SIZE[0], height=SEED_SIZE[1], output_type='np',
                generator=self.torch.Generator(device='cuda').manual_seed(seed)).images[0]
        if output.shape != (SEED_SIZE[1], SEED_SIZE[0], 3) or not np.isfinite(output).all():
            raise ValueError('SD-Turbo returned an invalid seed image')
        return (np.clip(output, 0, 1) * 255).round().astype(np.uint8)
