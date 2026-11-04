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
BOOTSTRAP_PIP = 'pip==26.3'
DEPENDENCIES = (
    f'world_engine @ git+https://github.com/Overworldai/world_engine.git@{RUNTIME_REVISION}',
    'fastapi==0.162.0', 'uvicorn==0.49.0', 'websockets==16.0',
    'opencv-python-headless==4.13.0.92', 'pillow==13.0.0',
    'transformers==5.10.1', 'huggingface-hub==1.7.2',
)
# Compatible with the existing Hub 1.7 / Transformers 5 pins. Newer Diffusers
# require a newer Hub; do not silently upgrade the working world-model stack.
SEED_DEPENDENCIES = ('diffusers==0.39.0',)


def _capture(command, stage):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding='utf-8', errors='replace')
    if result.returncode:
        raise RuntimeError(f'{stage} failed (exit {result.returncode}):\n{result.stdout[-6000:]}')
    return result.stdout.strip()


def ensure_environment(environment):
    """Create or repair pip without relying on the host's ensurepip package."""
    environment = Path(environment).resolve()
    python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    created = not python.exists() or not (environment / 'pyvenv.cfg').is_file()
    if created:
        venv.EnvBuilder(with_pip=False).create(environment)
    # Check the target before allowing host pip to install into it. A venv's
    # interpreter may be a symlink, so inspect sys.prefix rather than resolving it.
    state = json.loads(_capture([str(python), '-I', '-c',
        'import json, sys; print(json.dumps({"prefix": sys.prefix, '
        '"base": sys.base_prefix, "version": list(sys.version_info[:2])}))'],
        'Virtual environment check'))
    if (Path(state['prefix']).resolve() != environment
            or state['prefix'] == state['base']
            or state['version'] != list(sys.version_info[:2])):
        raise RuntimeError('Waypoint Python is not an isolated environment for this Python version; '
                           'use a fresh runtime or a different Waypoint runtime folder')
    probe = subprocess.run([str(python), '-I', '-m', 'pip', '--version'],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if probe.returncode == 0:
        return python, created
    if not created:
        # An ensurepip failure leaves python/pyvenv.cfg but may omit activation
        # scripts. Complete these files in place; preserve installed packages.
        venv.EnvBuilder(with_pip=False).create(environment)
    print('Bootstrapping pip in the isolated Waypoint environment (without ensurepip)...', flush=True)
    output = _capture([sys.executable, '-m', 'pip', '--python', str(python),
                       'install', '--no-cache-dir', '--force-reinstall', BOOTSTRAP_PIP], 'Isolated pip bootstrap')
    if output:
        print(output, flush=True)
    _capture([str(python), '-I', '-m', 'pip', '--version'], 'Isolated pip verification')
    return python, True


def setup(root, prompt_seeds=False):
    if platform.system() not in ('Linux', 'Windows') or platform.machine().lower() not in ('x86_64', 'amd64'):
        raise RuntimeError('This installer supports x86-64 Linux and Windows CUDA hosts')
    root = Path(root).resolve()
    environment = root / 'venv'
    marker = root / 'installed.json'
    dependencies = (*DEPENDENCIES, *(SEED_DEPENDENCIES if prompt_seeds else ()))
    signature = {'python': list(sys.version_info[:2]), 'dependencies': list(dependencies)}
    root.mkdir(parents=True, exist_ok=True)
    python, repaired = ensure_environment(environment)
    reusable = not repaired and marker.exists() and json.loads(marker.read_text()) == signature
    if not reusable:
        # Windows PyPI Torch may be CPU-only; select the CUDA build explicitly.
        subprocess.run([str(python), '-m', 'pip', 'install', '--no-cache-dir',
                        'torch==2.11.0', '--index-url', 'https://download.pytorch.org/whl/cu130'], check=True)
        subprocess.run([str(python), '-m', 'pip', 'install', '--no-cache-dir', *dependencies], check=True)
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
