"""Check pinned HF code imports on CPU, without downloading model weights."""
from concurrent.futures import ThreadPoolExecutor
import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request

from krea_assets import MODEL_ID, MODEL_REVISION, WAN_ID, WAN_REVISION, RUNTIME_HASHES, prepare_pipeline
from krea_weights import KREA_WEIGHTS, WAN_WEIGHTS, stage_snapshot

PACKAGES = ('diffusers==0.42.1', 'accelerate==1.12.0', 'peft==0.21.2',
            'einops==0.8.2', 'ftfy==6.3.1', 'wcwidth==0.9.4', 'imageio==2.38.1',
            'huggingface-hub==1.33.0', 'psutil==7.2.2', 'importlib-metadata==10.1.0',
            'zipp==3.23.1', 'requests==2.34.2', 'safetensors==0.8.0')

SMOKE = '''import sys
from diffusers import ModularPipeline, BitsAndBytesConfig, AutoencoderKLWan
from diffusers.modular_pipelines import PipelineState
from diffusers.utils.dynamic_modules_utils import get_class_from_dynamic_module
pipe = ModularPipeline.from_pretrained(sys.argv[1], trust_remote_code=True, local_files_only=True)
assert type(pipe).__name__ == 'WanModularPipeline'
assert 'transformer' in pipe.component_names
model = get_class_from_dynamic_module(sys.argv[2], subfolder='transformer',
    module_file='causal_model.py', class_name='CausalWanModel', trust_remote_code=True, local_files_only=True)
assert model.__name__ == 'CausalWanModel'
assert hasattr(AutoencoderKLWan, 'clear_cache')
assert BitsAndBytesConfig(load_in_8bit=True).load_in_8bit
print('Pinned Krea blocks, transformer class, VAE cache API and INT8 configuration import on CPU.')
from diffusers.models.model_loading_utils import load_state_dict
import torch
weights = load_state_dict(sys.argv[3], map_location='cpu')
assert torch.equal(weights['fixture'], torch.arange(8, dtype=torch.float32))
print('Real Diffusers deserialization of a SHA-256-validated, staged CPU checkpoint passed.')
'''


def main():
    with tempfile.TemporaryDirectory(prefix='colab-sd-krea-import-') as directory:
        scratch = Path(directory).resolve()
        assert scratch.parent == Path(tempfile.gettempdir()).resolve()
        packages, snapshot = scratch / 'packages', scratch / 'snapshot'
        env = {**os.environ, 'PYTHONPATH': str(packages), 'PYTHONDONTWRITEBYTECODE': '1',
               'HF_MODULES_CACHE': str(scratch / 'modules'), 'HF_HUB_CACHE': str(scratch / 'hub'),
               'HF_HUB_OFFLINE': '1', 'DIFFUSERS_ENABLE_HUB_KERNELS': '0', 'DISABLE_SAGEATTENTION': '1',
               'TMPDIR': str(scratch), 'TMP': str(scratch), 'TEMP': str(scratch),
               'PIP_CONFIG_FILE': os.devnull, 'PIP_INDEX_URL': 'https://pypi.org/simple', 'PIP_EXTRA_INDEX_URL': ''}
        subprocess.run([sys.executable, '-B', '-m', 'pip', 'install', '--no-cache-dir', '--no-deps', '--only-binary=:all:',
                        '--target', str(packages), *PACKAGES], env=env, check=True, timeout=180)

        def fetch(name):
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(f'https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}',
                                        timeout=60) as response:
                target.write_bytes(response.read())
        with ThreadPoolExecutor(max_workers=5) as pool:
            list(pool.map(fetch, RUNTIME_HASHES))
        for repo, revision, weights in ((MODEL_ID, MODEL_REVISION, KREA_WEIGHTS),
                                         (WAN_ID, WAN_REVISION, WAN_WEIGHTS)):
            with urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true',
                                        timeout=60) as response:
                metadata = {item['rfilename']: item for item in json.load(response)['siblings']}
            for name, (size, digest) in weights.items():
                assert (metadata[name]['size'], metadata[name]['lfs']['sha256']) == (size, digest), name
        print('Nine checkpoint sizes/SHA-256 values match pinned Hub metadata; no weight payloads fetched.')
        import hashlib
        import numpy as np
        from safetensors.numpy import save_file
        fixture = scratch / 'fixture'
        fixture.mkdir()
        checkpoint = fixture / 'tiny.safetensors'
        save_file({'fixture': np.arange(8, dtype=np.float32)}, checkpoint)
        expected = {'tiny.safetensors': (checkpoint.stat().st_size, hashlib.sha256(checkpoint.read_bytes()).hexdigest())}
        staged = stage_snapshot(fixture, scratch / 'staged', MODEL_ID, MODEL_REVISION, expected, [])
        # Exercise dynamic imports from the same revision-named local folder
        # layout used by the GPU loader, including its custom transformer code.
        runtime = stage_snapshot(snapshot, scratch / 'model-files' / ('krea-' + MODEL_REVISION),
                                 MODEL_ID, MODEL_REVISION, {}, list(RUNTIME_HASHES))
        pipeline = prepare_pipeline(runtime, scratch / 'wan', scratch)
        subprocess.run([sys.executable, '-B', '-c', SMOKE, str(pipeline), str(runtime), str(staged / checkpoint.name)],
                        cwd=scratch, env=env, check=True, timeout=120)
    print('Disposable import scratch permanently removed; CPU environment preserved. No model weights downloaded.')


if __name__ == '__main__':
    main()
