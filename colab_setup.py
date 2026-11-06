"""Colab setup source embedded into the generated notebook."""
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import urllib.request

UPSTREAM_REVISION = 'b623251dc055e1fd858d53509aa43e09dfc5cdc0'
CORE_DEPENDENCIES = (
    'diffusers==0.24.0', 'transformers==4.39.3',
    'huggingface-hub==0.25.2', 'accelerate==0.30.1',
    'fastapi', 'uvicorn', 'websockets', 'python-multipart',
    'opencv-python-headless', 'pillow',
    'aiortc==1.15.0', 'av==17.1.0',
)
DEFAULT_DRIVE_MODEL_FOLDER = 'StreamDiffusion/models'
MODEL_CACHE_VARIABLES = ('HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'TRANSFORMERS_CACHE')


def configure_model_cache(use_google_drive=False, drive_folder=DEFAULT_DRIVE_MODEL_FOLDER,
                          mount_point='/content/drive', local_cache=None):
    """Select model storage before launching the server's fresh Python process."""
    if use_google_drive:
        folder = Path(drive_folder)
        if not str(drive_folder).strip() or not folder.parts or folder.is_absolute() or '..' in folder.parts:
            raise ValueError('Drive model folder must be a relative folder inside MyDrive, without ..')
        from google.colab import drive
        drive.mount(str(mount_point))
        my_drive = (Path(mount_point) / 'MyDrive').resolve()
        if not my_drive.is_dir():
            raise RuntimeError('Google Drive is not mounted. Authorize Drive access and rerun setup.')
        cache = (my_drive / folder).resolve()
        if not cache.is_relative_to(my_drive) or cache == my_drive:
            raise ValueError('Choose a dedicated model cache folder inside MyDrive.')
    else:
        cache = Path(local_cache or Path.home() / '.cache' / 'huggingface' / 'hub').expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    # Fail visibly on a disconnected/read-only/full Drive instead of claiming
    # persistence while downloading models into the temporary VM filesystem.
    with tempfile.NamedTemporaryFile(prefix='.bridge-cache-check-', dir=cache) as probe:
        probe.write(b'model-cache-write-check')
        probe.flush()
    for variable in MODEL_CACHE_VARIABLES:
        os.environ[variable] = str(cache)
    # Leave HF_HOME and HF_TOKEN_PATH alone: model persistence does not require
    # moving authentication tokens into the user's Drive model folder.
    location = 'Google Drive (persistent)' if use_google_drive else 'runtime disk (temporary)'
    print(f'Model cache: {cache} [{location}]', flush=True)
    if use_google_drive:
        print('Existing cached weights are reused. Missing or updated model files download here on first use.', flush=True)
    return cache


def setup_environment(root='/content/StreamDiffusion', use_google_drive=False,
                      drive_model_folder=DEFAULT_DRIVE_MODEL_FOLDER):
    if platform.system() != 'Linux':
        raise RuntimeError('This setup targets Google Colab Linux GPU runtimes.')
    configure_model_cache(use_google_drive, drive_model_folder)
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
    # Colab's generated setup cell defines these form values above this source.
    # Importing this module for workstation setup never mounts Drive.
    setup_environment(use_google_drive=globals().get('USE_GOOGLE_DRIVE', False),
                      drive_model_folder=globals().get('DRIVE_MODEL_FOLDER', DEFAULT_DRIVE_MODEL_FOLDER))
