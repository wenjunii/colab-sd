"""Separate Windows/Linux environment for the Apache-declared Krea HF runtime."""
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import urllib.request

from waypoint_setup import ensure_environment

DEPENDENCIES = (
    'diffusers==0.42.1', 'transformers==5.10.1', 'huggingface-hub==1.33.0',
    'accelerate==1.12.0', 'bitsandbytes==0.50.2', 'peft==0.21.2',
    'einops==0.8.2', 'ftfy==6.3.1', 'sentencepiece==0.2.1', 'imageio==2.38.1',
    'fastapi==0.162.0', 'uvicorn==0.49.0', 'websockets==16.0',
    'opencv-python-headless==4.13.0.92', 'pillow==13.0.0',
    'aiortc==1.15.0', 'av==17.1.0',
)
TORCH = 'torch==2.13.0'
TORCH_INDEX = 'https://download.pytorch.org/whl/cu130'


def require_unused_port(port=8000):
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=.5):
            pass
    except OSError:
        return
    raise RuntimeError(f'A server already uses port {port}. Use a fresh GPU runtime for Krea; '
                       'the existing server was left running.')


def installation_signature():
    return {'python': list(sys.version_info[:2]), 'torch': TORCH, 'index': TORCH_INDEX,
            'dependencies': list(DEPENDENCIES)}


def setup(root, install_tunnel=True, project_dir=None, check_port=True):
    if platform.system() not in ('Linux', 'Windows') or platform.machine().lower() not in ('x86_64', 'amd64'):
        raise RuntimeError('This Krea installer supports x86-64 Linux and Windows CUDA hosts')
    if check_port:
        require_unused_port()
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    python, repaired = ensure_environment(root / 'venv', label='Krea')
    signature = installation_signature()
    marker = root / 'installed.json'
    reusable = not repaired and marker.is_file() and json.loads(marker.read_text()) == signature
    pip_env = {key: value for key, value in os.environ.items() if not key.startswith('PIP_')}
    pip_env['PIP_CONFIG_FILE'] = os.devnull
    if not reusable:
        subprocess.run([str(python), '-m', 'pip', 'install', '--no-cache-dir', TORCH,
                        '--index-url', TORCH_INDEX], env=pip_env, check=True)
        subprocess.run([str(python), '-m', 'pip', 'install', '--no-cache-dir',
                        '--index-url', 'https://pypi.org/simple', *DEPENDENCIES], env=pip_env, check=True)
    subprocess.run([str(python), '-m', 'pip', 'check'], env=pip_env, check=True)
    # GPU/import checks precede model downloads. No attention extension or
    # noncommercial Krea GitHub application is installed.
    subprocess.run([str(python), '-c',
                    'import torch; from diffusers import ModularPipeline, BitsAndBytesConfig; '
                    'import bitsandbytes; from krea_backend import gpu_preflight; '
                    'print(gpu_preflight(torch), flush=True)'], cwd=project_dir or root, env=pip_env, check=True)
    marker.write_text(json.dumps(signature, indent=2) + '\n')
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
