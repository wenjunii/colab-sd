"""Colab setup source embedded into the generated notebook."""
from contextlib import contextmanager
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
SD_TURBO = ('stabilityai/sd-turbo', 'b261bac6fd2cf515557d5d0707481eafa0485ec2')
SDXL_TURBO = ('stabilityai/sdxl-turbo', '71153311d3dbb46851df1931d3ca6e939de83304')
TAESD = ('madebyollin/taesd', '614f76814bbe30edbe2e627ace1c2234c81a2c0e')
TAESDXL = ('madebyollin/taesdxl', 'b20258aaef75ef61e659c1e0f14f251cf0ad153e')
TINY_VAE_FILES = ('config.json', 'diffusion_pytorch_model.safetensors')


def pip_cache_arguments(default_no_cache=True):
    """Reuse package downloads only when this project selected a persistent cache."""
    cache = os.environ.get('BRIDGE_PIP_CACHE_DIR')
    if cache:
        return ['--cache-dir', str(Path(cache).expanduser().resolve())]
    return ['--no-cache-dir'] if default_no_cache else []


def model_cache_arguments():
    cache = os.environ.get('HF_HUB_CACHE')
    return {'cache_dir': cache} if cache else {}


def _resolve_snapshot_files(snapshot, repo, revision, files):
    from krea_weights import InvalidCheckpoint, resolve_cached_file
    resolved = {}
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
            resolved[name] = candidate
            total += size
        except (OSError, InvalidCheckpoint):
            missing.append(name)
    return resolved, missing, total


def _cached_model_files(repo, revision, files):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo) or not re.fullmatch(r'[a-f0-9]{40}', revision):
        raise ValueError('Local model reuse requires a pinned repository/revision')
    cache = os.environ.get('HF_HUB_CACHE') or os.environ.get('HUGGINGFACE_HUB_CACHE')
    if not cache:
        default_home = Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache') / 'huggingface'
        cache = Path(os.environ.get('HF_HOME') or default_home) / 'hub'
    snapshot = Path(cache).expanduser().resolve() / ('models--' + repo.replace('/', '--')) / 'snapshots' / revision
    if not snapshot.is_dir():
        print(f'[models] {repo}: pinned snapshot not cached at {snapshot}; using the normal download path.', flush=True)
        return None
    resolved, missing, total = _resolve_snapshot_files(snapshot, repo, revision, files)
    if missing:
        print(f'[models] {repo}: {len(missing)} required file(s) missing/incomplete; existing cached files are kept.', flush=True)
        return None
    print(f'[models] Reusing {repo} from {snapshot} ({total / 1024**3:.2f} GiB); no Hub download requested.', flush=True)
    return snapshot, resolved


def cached_model_snapshot(repo, revision, files):
    """Find a complete pinned snapshot for loaders that stage and validate its files."""
    found = _cached_model_files(repo, revision, files)
    return str(found[0]) if found is not None else None


@contextmanager
def model_load_source(repo, revision, files, *, download=False):
    """Give eager loaders a readable directory or let them fetch absent snapshots."""
    cached = _cached_model_files(repo, revision, files)
    if cached is not None:
        snapshot, resolved = cached
    elif download:
        from huggingface_hub import snapshot_download
        snapshot = Path(snapshot_download(repo, revision=revision, allow_patterns=list(files),
                                          **model_cache_arguments())).resolve()
        resolved, missing, _ = _resolve_snapshot_files(snapshot, repo, revision, files)
        if missing:
            raise RuntimeError(f'{repo} snapshot is incomplete or unreadable: {", ".join(missing)}. '
                               'Check the configured model cache and Drive mount.')
    else:
        yield repo
        return

    # Native symlinks already point to the payload. An older Drive text link
    # is a regular file with a blob path as its contents, so eager loaders
    # need a readable directory with the same filenames.
    if all((snapshot / name).resolve() == source for name, source in resolved.items()):
        yield str(snapshot)
        return
    temp_root = Path(tempfile.gettempdir()).resolve()
    with tempfile.TemporaryDirectory(prefix='bridge-model-view-', dir=temp_root) as directory:
        view = Path(directory).resolve()
        if view.parent != temp_root:
            raise RuntimeError('Model view resolved outside its temporary directory')
        for name, source in resolved.items():
            link = view / name
            link.parent.mkdir(parents=True, exist_ok=True)
            try:
                link.symlink_to(source)
            except OSError as exc:
                raise RuntimeError(f'Unable to link the older Drive cache for {repo}. '
                                   'Load it on Colab or use a workstation cache with regular model files.') from exc
        print(f'[models] Opened readable links for {repo}; loading from the existing cache.', flush=True)
        yield str(view)


def turbo_model_files(*, xl=False, fp16=False):
    """Files used by the pinned Turbo pipelines and their text encoders."""
    variant = '.fp16' if fp16 else ''
    files = ['model_index.json', 'scheduler/scheduler_config.json']
    for component in ('unet', 'vae'):
        files.extend((f'{component}/config.json', f'{component}/diffusion_pytorch_model{variant}.safetensors'))
    encoder_suffixes = ('', '_2') if xl else ('',)
    for suffix in encoder_suffixes:
        encoder, tokenizer = 'text_encoder' + suffix, 'tokenizer' + suffix
        files.extend((f'{encoder}/config.json', f'{encoder}/model{variant}.safetensors'))
        files.extend(f'{tokenizer}/{name}' for name in ('special_tokens_map.json', 'tokenizer_config.json',
                                                       'vocab.json', 'merges.txt'))
    return files


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
