"""SDXL conditioning adapter for the pinned, unmodified StreamDiffusion core."""
from types import SimpleNamespace

import torch

MODEL_ID = 'stabilityai/sdxl-turbo'
MODEL_REVISION = '71153311d3dbb46851df1931d3ca6e939de83304'
VAE_ID = 'madebyollin/taesdxl'
VAE_REVISION = 'b20258aaef75ef61e659c1e0f14f251cf0ad153e'


class XLConditionedUNet:
    """Supply live pooled text and size inputs to native or compiled SDXL UNets."""
    def __init__(self, model, width, height, state=None):
        self.model, self.width, self.height = model, width, height
        self.state = state if state is not None else SimpleNamespace(pooled=None, time_ids=None)

    def __getattr__(self, name):
        return getattr(self.model, name)

    def set_conditioning(self, pooled):
        if not torch.is_tensor(pooled) or pooled.ndim != 2 or pooled.shape[0] != 1:
            raise ValueError('SDXL requires one pooled prompt embedding before batch expansion')
        self.state.pooled = pooled
        self.state.time_ids = pooled.new_tensor([[self.height, self.width, 0, 0, self.height, self.width]])

    def with_model(self, model):
        return type(self)(model, self.width, self.height, self.state)

    def __call__(self, sample, timestep, encoder_hidden_states, **kwargs):
        if self.state.pooled is None:
            raise RuntimeError('Prepare SDXL prompt conditioning before inference')
        batch = sample.shape[0]
        kwargs['added_cond_kwargs'] = {
            'text_embeds': self.state.pooled.repeat(batch, 1),
            'time_ids': self.state.time_ids.repeat(batch, 1),
        }
        return self.model(sample, timestep, encoder_hidden_states=encoder_hidden_states, **kwargs)


class XLStreamPipe:
    """Keep StreamDiffusion preparation/update_prompt in sync with XL pooled text."""
    def __init__(self, pipe, width, height):
        self.base = pipe
        self.unet = XLConditionedUNet(pipe.unet, width, height)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def encode_prompt(self, *args, **kwargs):
        if kwargs.get('do_classifier_free_guidance', False):
            raise ValueError('SDXL-Turbo streaming uses no classifier-free guidance')
        output = self.base.encode_prompt(*args, **kwargs)
        self.unet.set_conditioning(output[2])
        return output


class SDXLStreamWrapper:
    def __init__(self, stream):
        self.stream = stream

    @classmethod
    def load(cls, width, height):
        from diffusers import StableDiffusionXLPipeline, AutoencoderTiny
        from streamdiffusion import StreamDiffusion
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float16 if device == 'cuda' else torch.float32
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, variant='fp16', torch_dtype=dtype,
            use_safetensors=True, add_watermarker=False)
        # SDXL needs its own TinyVAE. Replace the original before moving to GPU.
        pipe.vae = AutoencoderTiny.from_pretrained(VAE_ID, revision=VAE_REVISION,
                                                  torch_dtype=dtype, use_safetensors=True)
        pipe.to(device=device, dtype=dtype)
        pipe.set_progress_bar_config(disable=True)
        stream = StreamDiffusion(pipe=XLStreamPipe(pipe, width, height), t_index_list=[0],
                                 torch_dtype=dtype, width=width, height=height,
                                 frame_buffer_size=1, use_denoising_batch=True, cfg_type='none')
        return cls(stream)

    def prepare(self, prompt, **kwargs):
        self.stream.prepare(prompt=prompt, **kwargs)

    def __call__(self):
        output = self.stream.txt2img_sd_turbo(1)
        return self.stream.image_processor.postprocess(output, output_type='pil')[0]
