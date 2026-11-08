"""Colab setup source embedded into the generated notebook."""
import os
import json
import importlib.metadata
import importlib.util
from pathlib import Path
import platform
import re
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


def pip_cache_arguments(default_no_cache=True):
    """Reuse package downloads only when this project selected a persistent cache."""
    cache = os.environ.get('BRIDGE_PIP_CACHE_DIR')
    if cache:
        return ['--cache-dir', str(Path(cache).expanduser().resolve())]
    return ['--no-cache-dir'] if default_no_cache else []


def model_cache_arguments():
    cache = os.environ.get('HF_HUB_CACHE')
    return {'cache_dir': cache} if cache else {}


def cached_model_snapshot(repo, revision, files):
    """Check every required pinned file locally before making any Hub request.

    Size checks select an existing snapshot; staging still validates its hashes.
    This does not change or delete existing models, snapshots or credentials.
    """
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo) or not re.fullmatch(r'[a-f0-9]{40}', revision):
        raise ValueError('Local model reuse requires a pinned repository/revision')
    cache = Path(os.environ.get('HF_HUB_CACHE', Path.home() / '.cache/huggingface/hub')).expanduser().resolve()
    snapshot = cache / ('models--' + repo.replace('/', '--')) / 'snapshots' / revision
    if not snapshot.is_dir():
        print(f'[models] {repo}: pinned snapshot not cached at {snapshot}; missing files will be fetched.', flush=True)
        return None
    from krea_weights import InvalidCheckpoint, resolve_cached_file
    total = 0
    missing = []
    entries = files.items() if isinstance(files, dict) else ((name, None) for name in files)
    for name, expected in entries:
        candidate = snapshot / name
        if not candidate.absolute().is_relative_to(snapshot) or '..' in Path(name).parts or Path(name).is_absolute():
            raise ValueError('Model filename escaped its snapshot')
        try:
            candidate = resolve_cached_file(candidate, snapshot, repo, revision)
            size = candidate.stat().st_size
            if (not candidate.is_file() or (expected is None and size == 0)
                    or (expected is not None and size != expected[0])):
                raise InvalidCheckpoint('missing or wrong-sized cached file')
            total += size
        except (OSError, InvalidCheckpoint):
            missing.append(name)
    if missing:
        print(f'[models] {repo}: {len(missing)} required file(s) missing/incomplete; existing cached files are kept.', flush=True)
        return None
    print(f'[models] Reusing {repo} from {snapshot} ({total / 1024**3:.2f} GiB); no Hub download requested.', flush=True)
    return str(snapshot)


def core_packages_ready(root):
    """Verify installed versions and the editable source before skipping pip."""
    try:
        for requirement in CORE_DEPENDENCIES:
            name, _, expected = requirement.partition('==')
            if expected and importlib.metadata.version(name) != expected:
                return False
            if not expected:
                importlib.metadata.version(name)
            modules = {'huggingface-hub': ('huggingface_hub',), 'opencv-python-headless': ('cv2',),
                       'pillow': ('PIL',), 'python-multipart': ('python_multipart', 'multipart')}.get(
                           name, (name.replace('-', '_'),))
            if not any((spec := importlib.util.find_spec(module)) is not None
                       and spec.origin and Path(spec.origin).is_file() for module in modules):
                return False
        direct = json.loads(importlib.metadata.distribution('streamdiffusion').read_text('direct_url.json') or '{}')
        if direct.get('url') != Path(root).as_uri() or direct.get('dir_info', {}).get('editable') is not True:
            return False
        try:
            importlib.metadata.version('peft')
            return False  # Legacy Diffusers is incompatible with this extra.
        except importlib.metadata.PackageNotFoundError:
            return True
    except (importlib.metadata.PackageNotFoundError, ValueError, TypeError):
        return False


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
    package_cache = None
    if use_google_drive:
        package_cache = (cache.parent / '.bridge-packages' / 'pip-cache').resolve()
        if not package_cache.is_relative_to(my_drive):
            raise ValueError('Package cache must remain inside MyDrive.')
        package_cache.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix='.bridge-cache-check-', dir=package_cache) as probe:
            probe.write(b'package-cache-write-check')
            probe.flush()
    for variable in MODEL_CACHE_VARIABLES:
        os.environ[variable] = str(cache)
    if package_cache:
        os.environ['BRIDGE_PIP_CACHE_DIR'] = str(package_cache)
    else:
        os.environ.pop('BRIDGE_PIP_CACHE_DIR', None)
    # Leave HF_HOME and HF_TOKEN_PATH alone: model persistence does not require
    # moving authentication tokens into the user's Drive model folder.
    location = 'Google Drive (persistent)' if use_google_drive else 'runtime disk (temporary)'
    print(f'Model cache: {cache} [{location}]', flush=True)
    repositories = sum(path.is_dir() for path in cache.glob('models--*'))
    print(f'Saved model repositories in this folder: {repositories} (required files are checked when loaded).', flush=True)
    if use_google_drive:
        print('Existing cached weights are reused. Missing or updated model files download here on first use.', flush=True)
        print(f'Package download cache: {package_cache} [Google Drive (persistent)]', flush=True)
        print('Fresh VMs still install packages locally, copy staged models from Drive, and load models into GPU memory.', flush=True)
        if not repositories:
            print('This cache folder has no saved Hub repositories yet. If you cached models before, keep that same DRIVE_MODEL_FOLDER.', flush=True)
    else:
        print('Drive persistence is OFF: model/package files on this VM are lost when the runtime is deleted.', flush=True)
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
    marker = root / '.bridge-installed.json'
    signature = {'managed_by': 'colab-sd', 'revision': UPSTREAM_REVISION,
                 'python': list(sys.version_info[:2]), 'dependencies': list(CORE_DEPENDENCIES)}
    try:
        reusable = json.loads(marker.read_text(encoding='utf-8')) == signature and core_packages_ready(root)
    except (OSError, ValueError):
        reusable = False
    if reusable:
        print('StreamDiffusion setup: reusing the installed packages on this VM.', flush=True)
    else:
        print('StreamDiffusion setup: installing packages on this VM; compatible cached downloads are reused.', flush=True)
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'peft'], check=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', *pip_cache_arguments(False),
                        '-q', '-e', '.', *CORE_DEPENDENCIES], check=True)
        marker.write_text(json.dumps(signature, indent=2) + '\n', encoding='utf-8')
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
