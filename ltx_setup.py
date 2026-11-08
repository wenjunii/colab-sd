"""Separate Windows/Linux CUDA environment for the pinned Apache LTX packages."""
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import urllib.request

from waypoint_setup import ensure_environment
from krea_setup import require_unused_port
from ltx_assets import CODE_REVISION, CODE_SHA256, ensure_source
from colab_setup import pip_cache_arguments

TORCH_INDEX = 'https://download.pytorch.org/whl/cu130'
TORCH_PACKAGES = ('torch==2.11.0', 'torchvision==0.26.0', 'torchaudio==2.11.0')
DEPENDENCIES = ('transformers==5.14.1', 'huggingface-hub==1.33.0', 'accelerate==1.12.0',
                'safetensors==0.8.0', 'einops==0.8.2', 'colour-science==0.4.7',
                'scipy==1.17.1', 'av==17.1.0', 'aiortc==1.15.0', 'openimageio==3.2.2.0',
                'cloudpickle==3.1.2', 'uv-build==0.9.30', 'pillow==13.0.0',
                'fastapi==0.162.0', 'uvicorn==0.49.0', 'websockets==16.0',
                'opencv-python-headless==4.13.0.92')


def setup_environment(python):
    environment = {k: v for k, v in os.environ.items() if not k.startswith('PIP_')}
    environment.update(PIP_CONFIG_FILE=os.devnull, PYTHONDONTWRITEBYTECODE='1')
    # Using the venv interpreter alone does not expose its installed build tools.
    # Keep the venv's bin path even when its Python executable is a symlink.
    scripts = str(Path(python).absolute().parent)
    paths = [scripts]
    if platform.system() == 'Windows' and Path(scripts).name.lower() != 'scripts':
        paths.append(str(Path(scripts) / 'Scripts'))
    environment['PATH'] = os.pathsep.join(filter(None, (*paths, environment.get('PATH', os.defpath))))
    return environment


def installation_signature():
    return {'python': list(sys.version_info[:2]), 'code': CODE_REVISION, 'sha256': CODE_SHA256,
            'torch': list(TORCH_PACKAGES), 'dependencies': list(DEPENDENCIES)}


def _run_stage(stage, command, environment, cwd=None):
    print(f'LTX setup: {stage}...', flush=True)

    def redact(output):
        for name in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN'):
            secret = environment.get(name)
            if secret:
                output = output.replace(secret, '[redacted]')
        return output

    try:
        result = subprocess.run(command, env=environment, cwd=cwd, check=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding='utf-8', errors='replace')
    except subprocess.CalledProcessError as error:
        output = redact(error.stdout or '')
        if output:
            print(output, flush=True)
        tail = '\n'.join(output.splitlines()[-40:])[-12000:]
        raise RuntimeError(f'LTX setup failed during {stage} (exit {error.returncode}).\n'
                           + (tail or 'The subprocess returned no output.')) from None
    if result.stdout:
        print(redact(result.stdout), flush=True)


def setup(root, install_tunnel=True, project_dir=None, check_port=True):
    if platform.system() not in ('Linux', 'Windows') or platform.machine().lower() not in ('x86_64', 'amd64'):
        raise RuntimeError('This LTX installer supports x86-64 Linux and Windows CUDA hosts')
    if check_port:
        require_unused_port()
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = ensure_source(root)
    python, repaired = ensure_environment(root / 'venv', label='LTX-2.5')
    signature = installation_signature()
    marker = root / 'installed.json'
    reusable = not repaired and marker.is_file() and json.loads(marker.read_text()) == signature
    environment = setup_environment(python)
    if not reusable:
        _run_stage('install CUDA PyTorch packages', [str(python), '-m', 'pip', 'install', *pip_cache_arguments(),
                        '--index-url', TORCH_INDEX, *TORCH_PACKAGES], environment)
        _run_stage('install runtime dependencies', [str(python), '-m', 'pip', 'install', *pip_cache_arguments(),
                        '--index-url', 'https://pypi.org/simple', *DEPENDENCIES], environment)
        _run_stage('install pinned LTX source', [str(python), '-m', 'pip', 'install', '--no-deps', '--no-build-isolation',
                        str(source / 'packages/ltx-core'), str(source / 'packages/ltx-pipelines')],
                       environment)
    else:
        print('LTX setup: reusing the installed environment on this VM.', flush=True)
    _run_stage('check installed dependencies', [str(python), '-m', 'pip', 'check'], environment)
    _run_stage('check GPU and LTX imports', [str(python), '-c', 'from ltx_backend import gpu_preflight; '
                    'from ltx_pipelines import DistilledPipeline; '
                    'from ltx_core.quantization.fp8_cast import build_policy; '
                    'print(gpu_preflight(), flush=True)'], environment, cwd=project_dir or root)
    marker.write_text(json.dumps(signature, indent=2) + '\n', encoding='utf-8')
    if not install_tunnel:
        return python, None
    binary = root / ('cloudflared.exe' if platform.system() == 'Windows' else 'cloudflared')
    if not binary.exists():
        partial = root / 'cloudflared.download'
        try:
            asset = 'cloudflared-windows-amd64.exe' if platform.system() == 'Windows' else 'cloudflared-linux-amd64'
            urllib.request.urlretrieve('https://github.com/cloudflare/cloudflared/releases/latest/download/' + asset, partial)
            partial.replace(binary)
        finally:
            partial.unlink(missing_ok=True)
        binary.chmod(0o755)
    return python, binary
