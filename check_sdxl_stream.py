"""Execute XL streaming or Waypoint seed contracts with tiny CPU components."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request

from colab_setup import UPSTREAM_REVISION

PROJECT = Path(__file__).resolve().parent
PACKAGES = ('diffusers==0.24.0', 'transformers==4.39.3', 'huggingface-hub==0.25.2',
            'accelerate==0.30.1', 'tokenizers==0.15.2', 'scipy==1.17.1',
            'psutil==7.2.2', 'importlib-metadata==10.1.0', 'zipp==3.23.1', 'requests==2.34.2')
WAYPOINT_PACKAGES = ('diffusers==0.39.0', 'huggingface-hub==1.7.2', 'accelerate==1.12.0',
                     'peft==0.21.2', 'scipy==1.17.1', 'psutil==7.2.2',
                     'importlib-metadata==10.1.0', 'zipp==3.23.1', 'requests==2.34.2')


def tiny_xl_pipeline():
    import torch
    from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, LCMScheduler
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    from types import SimpleNamespace
    torch.set_num_threads(1)
    torch.manual_seed(7)
    Path('vocab.json').write_text(json.dumps({'<|startoftext|>': 0, '<|endoftext|>': 1, 'a</w>': 2, 'b</w>': 3}))
    Path('merges.txt').write_text('#version: 0.2\n')
    tokenizer = CLIPTokenizer('vocab.json', 'merges.txt', model_max_length=77)
    config = CLIPTextConfig(vocab_size=4, hidden_size=32, intermediate_size=40, num_hidden_layers=2,
                            num_attention_heads=4, max_position_embeddings=77, projection_dim=16,
                            bos_token_id=0, eos_token_id=1, pad_token_id=1)
    encoder, encoder2 = CLIPTextModel(config), CLIPTextModelWithProjection(config)

    class ToyVAE(torch.nn.Module):
        config = SimpleNamespace(block_out_channels=[1, 2, 3, 4], scaling_factor=1., force_upcast=False)
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
        @property
        def dtype(self):
            return self.dummy.dtype
        @property
        def device(self):
            return self.dummy.device
        def encode(self, image):
            value = torch.nn.functional.interpolate(image, scale_factor=1/8)
            return SimpleNamespace(latents=torch.cat([value, value[:, :1]], dim=1))
        def decode(self, latent, **kwargs):
            return (torch.nn.functional.interpolate(latent[:, :3].tanh(), scale_factor=8),)

    unet = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
                block_out_channels=(16, 32), down_block_types=('CrossAttnDownBlock2D', 'DownBlock2D'),
                up_block_types=('UpBlock2D', 'CrossAttnUpBlock2D'), cross_attention_dim=64,
                attention_head_dim=4, norm_num_groups=8, addition_embed_type='text_time',
                addition_time_embed_dim=4, projection_class_embeddings_input_dim=40)
    pipe = StableDiffusionXLPipeline(vae=ToyVAE(), text_encoder=encoder, text_encoder_2=encoder2,
            tokenizer=tokenizer, tokenizer_2=tokenizer, unet=unet, scheduler=LCMScheduler(), add_watermarker=False)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def probe():
    import numpy as np
    import torch
    from PIL import Image
    from streamdiffusion import StreamDiffusion
    from sdxl_stream import XLStreamPipe, SDXLStreamWrapper
    pipe = tiny_xl_pipeline()
    proxy = XLStreamPipe(pipe, 64, 64)
    streams = [StreamDiffusion(pipe=proxy, t_index_list=indices, torch_dtype=torch.float32,
                width=64, height=64, cfg_type='none', frame_buffer_size=1, use_denoising_batch=True)
               for indices in ([0], [21, 45])]
    with torch.no_grad():
        for stream in streams:
            stream.prepare(prompt='a', num_inference_steps=50)
            assert stream.prompt_embeds.shape == (stream.batch_size, 77, 64)
            assert stream.unet.state.pooled.shape == (1, 16)
            old_pool = stream.unet.state.pooled.clone()
            stream.update_prompt('b')
            assert not torch.allclose(old_pool, stream.unet.state.pooled)
        image = SDXLStreamWrapper(streams[0])()
        assert image.size == (64, 64) and np.asarray(image).std() > 1
        # Execute the img2img path without the core's CUDA-only timing events.
        pixels = streams[1].image_processor.preprocess(Image.new('RGB', (64, 64), (64, 128, 192)))
        for _ in range(3):
            latent = streams[1].encode_image(pixels)
            output = streams[1].decode_image(streams[1].predict_x0_batch(latent))
            assert output.shape == (1, 3, 64, 64) and torch.isfinite(output).all()
    print('Real pinned StreamDiffusion + legacy XL pipeline/dual CLIP: txt2img and batched img2img pass on CPU.')
    print('Random tiny UNet/text encoders and toy VAE only; no pretrained weights or GPU performance tested.')


def probe_waypoint():
    import numpy as np
    import torch
    from unittest.mock import patch
    from diffusers import StableDiffusionPipeline
    import waypoint_seed
    pipe = tiny_xl_pipeline()
    assert hasattr(StableDiffusionPipeline, 'encode_prompt')
    seed = waypoint_seed.PromptSeedGenerator(pipe, torch, model='sdxl-turbo', device='cpu')
    with patch.object(waypoint_seed, 'SEED_SIZE', (64, 64)), \
            patch.object(pipe.text_encoder, 'forward', wraps=pipe.text_encoder.forward) as encoder, \
            patch.object(pipe.text_encoder_2, 'forward', wraps=pipe.text_encoder_2.forward) as encoder2:
        first = seed.generate('a', 'b', .25, 7)
        second = seed.generate('a', 'b', .75, 7)
        # The pipeline can call encode_prompt again to expand supplied tensors;
        # neither CLIP model should recompute the cached prompt pair.
        assert encoder.call_count == encoder2.call_count == 2
        assert first.shape == second.shape == (64, 64, 3) and first.dtype == np.uint8
        assert np.isfinite(first).all() and first.std() > 1 and np.any(first != second)
    print('Waypoint Diffusers 0.39 / Transformers 5: real tiny XL generation with cached token/pooled blending passed.')
    print('Random CPU components only; no pretrained weights, WorldEngine GPU execution or performance tested.')


def main(waypoint=False):
    with tempfile.TemporaryDirectory(prefix='colab-sd-xl-probe-') as directory:
        scratch = Path(directory).resolve()
        assert scratch.parent == Path(tempfile.gettempdir()).resolve()
        packages, source = scratch / 'packages', scratch / 'source'
        if not waypoint:
            package = source / 'streamdiffusion'
            package.mkdir(parents=True)
            for name in ('__init__.py', 'pipeline.py', 'image_filter.py'):
                url = f'https://raw.githubusercontent.com/cumulo-autumn/StreamDiffusion/{UPSTREAM_REVISION}/src/streamdiffusion/{name}'
                with urllib.request.urlopen(url, timeout=60) as response:
                    (package / name).write_bytes(response.read())
        env = {key: value for key, value in os.environ.items() if not key.startswith('PIP_')}
        env.update(PYTHONPATH=os.pathsep.join(map(str, (packages, PROJECT, source))), PYTHONDONTWRITEBYTECODE='1',
                   PIP_CONFIG_FILE=os.devnull, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   HF_HUB_CACHE=str(scratch / 'hub'), HF_MODULES_CACHE=str(scratch / 'modules'),
                   TMP=str(scratch), TEMP=str(scratch), TMPDIR=str(scratch))
        subprocess.run([sys.executable, '-B', '-m', 'pip', 'install', '--no-deps', '--no-cache-dir',
                        '--only-binary=:all:', '--index-url', 'https://pypi.org/simple', '--target', str(packages),
                        *(WAYPOINT_PACKAGES if waypoint else PACKAGES)], env=env, check=True, timeout=180)
        subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--probe-waypoint' if waypoint else '--probe'],
                        cwd=scratch, env=env, check=True, timeout=120)
    print('Permanently removed owned CPU probe scratch; reusable test environment preserved. No model weights downloaded.')


if __name__ == '__main__':
    if sys.argv[1:] == ['--probe']:
        probe()
    elif sys.argv[1:] == ['--probe-waypoint']:
        probe_waypoint()
    elif sys.argv[1:] in ([], ['--waypoint-seeds']):
        main(waypoint=bool(sys.argv[1:]))
    else:
        raise SystemExit('Usage: python check_sdxl_stream.py [--waypoint-seeds]')
