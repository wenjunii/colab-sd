"""Install the optional Waypoint runtime in its own environment, never over SD-Turbo."""
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import urllib.request
import venv

RUNTIME_REVISION = '68b316390d576dd9f671f552803526bbf8eb2ed5'
DEPENDENCIES = (
    f'world_engine @ git+https://github.com/Overworldai/world_engine.git@{RUNTIME_REVISION}',
    'fastapi==0.162.0', 'uvicorn==0.49.0', 'websockets==16.0',
    'opencv-python-headless==4.13.0.92', 'pillow==13.0.0',
    'transformers==5.10.1', 'huggingface-hub==1.7.2',
)


def setup(root):
    if platform.system() not in ('Linux', 'Windows') or platform.machine().lower() not in ('x86_64', 'amd64'):
        raise RuntimeError('This installer supports x86-64 Linux and Windows CUDA hosts')
    root = Path(root).resolve()
    environment = root / 'venv'
    python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    marker = root / 'installed.json'
    signature = {'python': list(sys.version_info[:2]), 'dependencies': list(DEPENDENCIES)}
    root.mkdir(parents=True, exist_ok=True)
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    reusable = marker.exists() and json.loads(marker.read_text()) == signature
    if not reusable:
        # Windows PyPI Torch may be CPU-only; select the CUDA build explicitly.
        subprocess.run([str(python), '-m', 'pip', 'install', '--no-cache-dir',
                        'torch==2.11.0', '--index-url', 'https://download.pytorch.org/whl/cu130'], check=True)
        subprocess.run([str(python), '-m', 'pip', 'install', '--no-cache-dir', *DEPENDENCIES], check=True)
    subprocess.run([str(python), '-m', 'pip', 'check'], check=True)
    # Fail before model downloads when the isolated runtime cannot use CUDA.
    subprocess.run([str(python), '-c',
                    'import torch; from world_engine import WorldEngine; '
                    'assert torch.cuda.is_available(), "CUDA unavailable in Waypoint environment"; '
                    'assert torch.cuda.get_device_capability()[0] >= 8, "Ampere+ GPU required"; '
                    'print(torch.__version__, torch.cuda.get_device_name(), flush=True)'], check=True)
    marker.write_text(json.dumps(signature, indent=2) + '\n', encoding='utf-8')
    binary = root / ('cloudflared.exe' if os.name == 'nt' else 'cloudflared')
    if not binary.exists():
        asset = 'cloudflared-windows-amd64.exe' if os.name == 'nt' else 'cloudflared-linux-amd64'
        partial = binary.with_suffix('.download')
        try:
            urllib.request.urlretrieve(f'https://github.com/cloudflare/cloudflared/releases/latest/download/{asset}', partial)
            partial.replace(binary)
        finally:
            partial.unlink(missing_ok=True)
        binary.chmod(0o755)
    return python, binary
