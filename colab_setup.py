"""Colab setup source embedded into the generated notebook."""
import os
from pathlib import Path
import platform
import subprocess
import sys
import urllib.request

UPSTREAM_REVISION = 'b623251dc055e1fd858d53509aa43e09dfc5cdc0'
CORE_DEPENDENCIES = (
    'diffusers==0.24.0', 'transformers==4.39.3',
    'huggingface-hub==0.25.2', 'accelerate==0.30.1',
    'fastapi', 'uvicorn', 'websockets', 'python-multipart',
    'opencv-python-headless', 'pillow',
)


def setup_environment(root='/content/StreamDiffusion'):
    if platform.system() != 'Linux':
        raise RuntimeError('This setup targets Google Colab Linux GPU runtimes.')
    root = Path(root).resolve()
    if not root.exists():
        subprocess.run(['git', 'clone', 'https://github.com/cumulo-autumn/StreamDiffusion.git', str(root)], check=True)
    subprocess.run(['git', '-C', str(root), 'fetch', '--depth', '1', 'origin', UPSTREAM_REVISION], check=True)
    subprocess.run(['git', '-C', str(root), 'checkout', '--detach', UPSTREAM_REVISION], check=True)
    os.chdir(root)
    # Resolve the editable package and compatibility pins together so a later
    # install cannot silently undo earlier transformer/hub constraints.
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'peft'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.', *CORE_DEPENDENCIES], check=True)
    target = root / 'cloudflared'
    if not target.exists():
        temporary = root / 'cloudflared.download'
        try:
            urllib.request.urlretrieve('https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64', temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    target.chmod(0o755)
    print('Setup complete. StreamDiffusion revision:', UPSTREAM_REVISION)
    print('Run the next cell to write the bridge server and helpers.')
    return root


if __name__ == '__main__':
    setup_environment()
