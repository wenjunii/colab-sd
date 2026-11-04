"""Check pinned HF code imports on CPU, without downloading model weights."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request

from krea_assets import MODEL_ID, MODEL_REVISION, RUNTIME_HASHES, prepare_pipeline

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
        pipeline = prepare_pipeline(snapshot, scratch / 'wan', scratch)
        subprocess.run([sys.executable, '-B', '-c', SMOKE, str(pipeline), str(snapshot)],
                        cwd=scratch, env=env, check=True, timeout=120)
    print('Disposable import scratch permanently removed; CPU environment preserved. No model weights downloaded.')


if __name__ == '__main__':
    main()
