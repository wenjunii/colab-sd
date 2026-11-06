"""Install and run the bridge on a Windows/Linux NVIDIA CUDA workstation."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import urllib.request
import venv

from colab_launcher import BridgeSession, TUNNEL_PROTOCOLS
from colab_setup import CORE_DEPENDENCIES, UPSTREAM_REVISION
from bridge_protocol import OUTPUT_RESOLUTIONS, LTX_OUTPUT_RESOLUTIONS, LTX_FRAME_COUNTS
from waypoint_setup import PROMPT_SEED_MODELS

PROJECT = Path(__file__).resolve().parent
DEFAULT_RUNTIME = PROJECT / '.workstation'
DEFAULT_WAYPOINT_RUNTIME = PROJECT / '.workstation-waypoint'
DEFAULT_KREA_RUNTIME = PROJECT / '.workstation-krea'
DEFAULT_LTX_RUNTIME = PROJECT / '.workstation-ltx'
RUNTIMES = {'sd-turbo': DEFAULT_RUNTIME, 'waypoint': DEFAULT_WAYPOINT_RUNTIME,
            'krea': DEFAULT_KREA_RUNTIME, 'ltx': DEFAULT_LTX_RUNTIME}
TORCH_VERSION = '2.13.0'


def run_pip(python, *arguments):
    environment = os.environ.copy()
    # This runtime uses public packages. Explicit indexes avoid an unrelated
    # user pip mirror silently selecting or blocking the CUDA installation.
    environment['PIP_INDEX_URL'] = 'https://pypi.org/simple'
    environment['PIP_EXTRA_INDEX_URL'] = 'https://pypi.org/simple'
    subprocess.run([str(python), '-m', 'pip', *arguments], check=True, env=environment)


def runtime_python(runtime):
    return Path(runtime) / 'venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def check_platform(python_version=(3, 11)):
    if platform.system() not in ('Windows', 'Linux') or platform.machine().lower() not in ('amd64', 'x86_64'):
        raise RuntimeError('Workstation mode needs Windows or Linux x86-64 with an NVIDIA CUDA GPU.')
    if sys.version_info[:2] != python_version:
        version = '.'.join(map(str, python_version))
        raise RuntimeError(f'Run this installer with Python {version} (Windows: py -{version} workstation.py setup).')


def gpu_check(python, root, backend='sd-turbo'):
    if backend in ('krea', 'ltx'):
        preflight = ('from ltx_backend import gpu_preflight; print(gpu_preflight())' if backend == 'ltx' else
                     'import torch; from krea_backend import gpu_preflight; print(gpu_preflight(torch))')
        subprocess.run([str(python), '-c', preflight], cwd=PROJECT, check=True)
        return
    capability = ('assert torch.cuda.get_device_capability()[0] >= 8, "Waypoint requires Ampere+ CUDA"; '
                  if backend == 'waypoint' else '')
    subprocess.run([str(python), '-c',
                    'import torch; '
                    'assert torch.cuda.is_available(), "CUDA unavailable: check NVIDIA driver and the selected PyTorch CUDA wheel"; '
                    + capability +
                    'print("GPU:", torch.cuda.get_device_name(0), "| Torch:", torch.__version__, "| CUDA:", torch.version.cuda)'],
                   cwd=root, check=True)


def setup(runtime, cuda='cu130', torch_version=TORCH_VERSION, model_cache=None):
    check_platform()
    if not re.fullmatch(r'cu\d{3}', cuda) or not re.fullmatch(r'\d+\.\d+\.\d+', torch_version):
        raise ValueError('Use a CUDA wheel tag such as cu130 and a Torch release such as 2.13.0.')
    runtime = Path(runtime).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    marker = runtime / 'workstation.json'
    if not marker.exists() and any(runtime.iterdir()):
        raise RuntimeError('Runtime folder is not empty or managed by this installer. Choose a new --runtime folder.')
    previous = {}
    if not marker.exists():
        marker.write_text(json.dumps({'managed_by': 'colab-sd', 'revision': UPSTREAM_REVISION}) + '\n', encoding='utf-8')
    else:
        configuration = json.loads(marker.read_text(encoding='utf-8'))
        if configuration.get('managed_by') != 'colab-sd' or configuration.get('backend', 'sd-turbo') != 'sd-turbo':
            raise RuntimeError('Runtime folder belongs to another application or backend; choose a separate --runtime.')
        previous = configuration
    selected = model_cache or previous.get('model_cache')
    # Existing StreamDiffusion installs used the caller's Hub cache. Preserve
    # that selection until an explicit override; do not redownload their models.
    cache = Path(selected or runtime / 'models').expanduser().resolve() if selected or not previous else None
    marker.write_text(json.dumps({'managed_by': 'colab-sd', 'revision': UPSTREAM_REVISION, 'ready': False,
                                 'model_cache': str(cache) if cache else None}) + '\n',
                      encoding='utf-8')
    if cache:
        waypoint_model_cache(runtime, cache)
    if shutil.which('git') is None:
        raise RuntimeError('Install Git and reopen your terminal before running workstation setup.')
    python = runtime_python(runtime)
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(runtime / 'venv')
    upstream = runtime / 'StreamDiffusion'
    if not upstream.exists():
        subprocess.run(['git', 'clone', 'https://github.com/cumulo-autumn/StreamDiffusion.git', str(upstream)], check=True)
    subprocess.run(['git', '-C', str(upstream), 'fetch', '--depth', '1', 'origin', UPSTREAM_REVISION], check=True)
    subprocess.run(['git', '-C', str(upstream), 'checkout', '--detach', UPSTREAM_REVISION], check=True)
    run_pip(python, 'install', '--upgrade', 'pip', 'setuptools>=84.1.0')
    run_pip(python, 'install', f'torch=={torch_version}+{cuda}', 'torchvision',
            '--index-url', f'https://download.pytorch.org/whl/{cuda}')
    gpu_check(python, upstream)
    # Keep legacy model dependencies inside this environment. Do not install
    # xformers or the upstream CUDA 11/12 TensorRT extras over the CUDA wheel.
    run_pip(python, 'uninstall', '-y', 'peft')
    dependencies = [dep for dep in CORE_DEPENDENCIES if dep not in ('opencv-python-headless', 'pillow')]
    dependencies += ['numpy==1.26.4', 'opencv-python-headless==4.10.0.84', 'Pillow>=13,<14', 'filelock>=4.2.1']
    run_pip(python, 'install', '-e', str(upstream), *dependencies)
    run_pip(python, 'check')
    subprocess.run([str(python), '-c',
                    'import sys; sys.path.insert(0, sys.argv[1]); import colab_server; '
                    'from utils.wrapper import StreamDiffusionWrapper',
                    str(PROJECT)], cwd=upstream, check=True)
    marker.write_text(json.dumps({'managed_by': 'colab-sd', 'revision': UPSTREAM_REVISION, 'ready': True,
                                 'torch': torch_version, 'cuda': cuda,
                                 'model_cache': str(cache) if cache else None}) + '\n', encoding='utf-8')
    print('Workstation setup complete. Run workstation.py start to load the models and stream.')


def waypoint_model_cache(runtime, selected=None):
    cache = Path(selected or Path(runtime) / 'models').expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    # Verify write access without modifying cached models or auth locations.
    with tempfile.TemporaryDirectory(prefix='.bridge-write-check-', dir=cache) as directory:
        probe = Path(directory).resolve()
        if probe.parent != cache:
            raise RuntimeError('Model cache probe resolved outside its cache folder')
        (probe / 'probe').write_bytes(b'cache check')
    return cache


def setup_waypoint(runtime, prompt_seeds=False, model_cache=None, prompt_seed_model='sd-turbo'):
    """Reuse the pinned Colab adapter/runtime in a separate workstation venv."""
    check_platform()
    if prompt_seed_model not in PROMPT_SEED_MODELS:
        raise ValueError('Prompt seed model must be sd-turbo or sdxl-turbo')
    runtime = Path(runtime).resolve()
    marker = runtime / 'workstation.json'
    if marker.exists():
        previous = json.loads(marker.read_text(encoding='utf-8'))
        if previous.get('managed_by') != 'colab-sd' or previous.get('backend') != 'waypoint':
            raise RuntimeError('Runtime folder belongs to another application or backend; choose a separate --runtime.')
    elif runtime.exists() and any(runtime.iterdir()):
        raise RuntimeError('Runtime folder is not empty or managed by this installer. Choose a new --runtime folder.')
    else:
        previous = {}
    if shutil.which('git') is None:
        raise RuntimeError('Install Git and reopen your terminal before running workstation setup.')
    from waypoint_setup import setup as install, RUNTIME_REVISION
    runtime.mkdir(parents=True, exist_ok=True)
    cache = Path(model_cache or previous.get('model_cache') or runtime / 'models').expanduser().resolve()
    configuration = {'managed_by': 'colab-sd', 'backend': 'waypoint', 'revision': RUNTIME_REVISION,
                     'ready': False, 'prompt_seeds': bool(prompt_seeds), 'model_cache': str(cache),
                     'prompt_seed_model': prompt_seed_model}
    marker.write_text(json.dumps(configuration, indent=2) + '\n', encoding='utf-8')
    waypoint_model_cache(runtime, cache)
    # Do not install the legacy StreamDiffusion environment or download a tunnel
    # for LAN users. Cloudflare is downloaded on demand at launch.
    install(runtime, prompt_seeds=prompt_seeds, install_tunnel=False)
    configuration['ready'] = True
    marker.write_text(json.dumps(configuration, indent=2) + '\n', encoding='utf-8')
    print('Waypoint workstation setup complete. Model cache:', cache)
    print('Run workstation.py start --backend waypoint to load the models and stream.')


def waypoint_environment(runtime, configuration, arguments):
    from waypoint_setup import DEPENDENCIES, SEED_DEPENDENCIES, RUNTIME_REVISION
    if configuration.get('revision') != RUNTIME_REVISION or not isinstance(configuration.get('prompt_seeds'), bool):
        raise RuntimeError('Waypoint setup is outdated. Run workstation.py setup --backend waypoint again.')
    installed = runtime / 'installed.json'
    expected = [*DEPENDENCIES, *(SEED_DEPENDENCIES if configuration['prompt_seeds'] else ())]
    if not installed.exists() or json.loads(installed.read_text(encoding='utf-8')).get('dependencies') != expected:
        raise RuntimeError('Waypoint dependencies changed. Run workstation.py setup --backend waypoint again.')
    model = arguments.prompt_seed_model or configuration.get('prompt_seed_model', 'sd-turbo')
    if model not in PROMPT_SEED_MODELS:
        raise ValueError('Prompt seed model must be sd-turbo or sdxl-turbo')
    if arguments.prompt_seed_model is not None and not configuration['prompt_seeds']:
        raise ValueError('Enable --prompt-seeds in Waypoint setup before selecting a prompt seed model.')
    cache = waypoint_model_cache(runtime, arguments.model_cache or configuration.get('model_cache'))
    return {'WAYPOINT_QUANT': arguments.quantization or 'intw8a8',
            'WAYPOINT_COMPILE_PROFILE': arguments.compile_profile or 'fast',
            'WAYPOINT_PROMPT_SEEDS': '1' if configuration['prompt_seeds'] else '0',
            'WAYPOINT_PROMPT_SEED_MODEL': model,
            'HF_HUB_CACHE': str(cache), 'HUGGINGFACE_HUB_CACHE': str(cache), 'TRANSFORMERS_CACHE': None,
            'TORCHINDUCTOR_CACHE_DIR': str(runtime / 'inductor-cache'),
            'TRITON_CACHE_DIR': str(runtime / 'triton-cache')}


def modern_installer(backend):
    if backend == 'ltx':
        import ltx_setup
        return ltx_setup
    if backend == 'krea':
        import krea_setup
        return krea_setup
    raise ValueError('Expected Krea or LTX backend')


def modern_revisions(backend):
    if backend == 'ltx':
        from ltx_assets import CODE_REVISION, CODE_SHA256, MODEL_REVISION
        return {'code': CODE_REVISION, 'code_sha256': CODE_SHA256, 'model': MODEL_REVISION}
    from krea_assets import MODEL_REVISION, WAN_REVISION
    return {'krea': MODEL_REVISION, 'wan': WAN_REVISION}


def modern_signature(backend):
    signature = modern_installer(backend).installation_signature()
    # Runtime identity is independent of the lightweight launcher's Python.
    # LTX's pinned PyAV 19 requires >=3.12; use Colab's verified Python 3.13.
    signature['python'] = [3, 13] if backend == 'ltx' else [3, 11]
    return signature


def setup_modern(runtime, backend, model_cache=None):
    """Use Colab's pinned installer, with backend-specific persistent storage."""
    check_platform((3, 13) if backend == 'ltx' else (3, 11))
    runtime = Path(runtime).resolve()
    marker = runtime / 'workstation.json'
    previous = {}
    if marker.exists():
        previous = json.loads(marker.read_text(encoding='utf-8'))
        if previous.get('managed_by') != 'colab-sd' or previous.get('backend') != backend:
            raise RuntimeError('Runtime folder belongs to another application or backend; choose a separate --runtime.')
    elif runtime.exists() and any(runtime.iterdir()):
        raise RuntimeError('Runtime folder is not empty or managed by this installer. Choose a new --runtime folder.')
    installer = modern_installer(backend)
    runtime.mkdir(parents=True, exist_ok=True)
    cache = Path(model_cache or previous.get('model_cache') or runtime / 'models').expanduser().resolve()
    configuration = {'managed_by': 'colab-sd', 'backend': backend, 'ready': False,
                     'revisions': modern_revisions(backend), 'installation': modern_signature(backend),
                     'model_cache': str(cache)}
    marker.write_text(json.dumps(configuration, indent=2) + '\n', encoding='utf-8')
    waypoint_model_cache(runtime, cache)
    installer.setup(runtime, install_tunnel=False, project_dir=PROJECT, check_port=False)
    configuration['ready'] = True
    marker.write_text(json.dumps(configuration, indent=2) + '\n', encoding='utf-8')
    print(f'{backend} workstation setup complete. Models download on first start. Cache:', cache)


def cache_environment(runtime, selected, legacy=False):
    cache = waypoint_model_cache(runtime, selected)
    return {'HF_HUB_CACHE': str(cache), 'HUGGINGFACE_HUB_CACHE': str(cache),
            'TRANSFORMERS_CACHE': str(cache) if legacy else None,
            'TORCHINDUCTOR_CACHE_DIR': str(runtime / 'inductor-cache')}


def modern_environment(runtime, configuration, arguments):
    backend = arguments.backend
    expected = modern_signature(backend)
    installed = runtime / 'installed.json'
    if (configuration.get('revisions') != modern_revisions(backend)
            or configuration.get('installation') != expected or not installed.is_file()
            or json.loads(installed.read_text(encoding='utf-8')) != expected):
        raise RuntimeError(f'{backend} setup is outdated. Run workstation.py setup --backend {backend} again.')
    environment = cache_environment(runtime, arguments.model_cache or configuration.get('model_cache'))
    if backend == 'krea':
        environment.update(KREA_INT8_PROFILE=arguments.int8_profile or 'fast',
            KREA_CLIP_BLOCKS=str(arguments.clip_blocks if arguments.clip_blocks is not None else 64),
            HF_MODULES_CACHE=str(runtime / 'modules-cache'),
            DIFFUSERS_ENABLE_HUB_KERNELS='0', DISABLE_SAGEATTENTION='1')
    else:
        image = str(arguments.input_image.expanduser().resolve(strict=True)) if arguments.input_image else ''
        environment.update(LTX_CLIP_FRAMES=str(arguments.clip_frames or 33), LTX_INPUT_IMAGE=image,
            LTX_PERFORMANCE_PROFILE=arguments.performance_profile or 'resident',
            LTX_CONTINUOUS='0' if arguments.continuous is False else '1',
            LTX_CONTINUE_SCENE='0' if arguments.continue_scene is False else '1',
            LTX_REFRESH_EVERY=str(arguments.refresh_every if arguments.refresh_every is not None else 4))
    return environment


def pairing_token(runtime):
    path = Path(runtime) / 'access-token'
    if not path.exists():
        # Exclusive creation also prevents concurrent launches replacing a key.
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            handle.write(secrets.token_urlsafe(32) + '\n')
    token = path.read_text(encoding='utf-8').strip()
    if len(token) < 32:
        raise RuntimeError('Invalid workstation access-token file; use at least 32 characters.')
    return token


def cloudflared_binary(runtime):
    installed = shutil.which('cloudflared')
    if installed:
        return Path(installed)
    name = 'cloudflared.exe' if os.name == 'nt' else 'cloudflared'
    target = Path(runtime) / name
    if not target.exists():
        asset = 'cloudflared-windows-amd64.exe' if os.name == 'nt' else 'cloudflared-linux-amd64'
        temporary = target.with_suffix('.download')
        try:
            urllib.request.urlretrieve(f'https://github.com/cloudflare/cloudflared/releases/latest/download/{asset}', temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    if os.name != 'nt':
        target.chmod(0o755)
    return target


def start(arguments):
    runtime = arguments.runtime.resolve()
    backend = arguments.backend
    if backend != 'sd-turbo':
        if arguments.model is not None:
            raise ValueError('--model applies only to the StreamDiffusion backend.')
        if (arguments.tensorrt or arguments.allow_tensorrt_resize or arguments.engine_cache
                or arguments.no_engine_cache):
            raise ValueError('TensorRT options apply only to the StreamDiffusion backend.')
    if backend == 'waypoint':
        if arguments.resolution not in (None, '1280x720'):
            raise ValueError('Waypoint uses fixed native 1280x720 output.')
    elif arguments.quantization or arguments.compile_profile or arguments.prompt_seed_model:
        raise ValueError('Waypoint options require --backend waypoint.')
    if backend != 'krea' and (arguments.int8_profile is not None or arguments.clip_blocks is not None):
        raise ValueError('Krea options require --backend krea.')
    if backend != 'ltx' and any(getattr(arguments, name) is not None for name in (
            'clip_frames', 'performance_profile', 'continuous', 'continue_scene', 'refresh_every', 'input_image')):
        raise ValueError('LTX options require --backend ltx.')
    if backend == 'krea' and arguments.clip_blocks is not None and not 1 <= arguments.clip_blocks <= 64:
        raise ValueError('Krea --clip-blocks must be 1 to 64.')
    if backend == 'ltx' and arguments.refresh_every is not None and not 0 <= arguments.refresh_every <= 32:
        raise ValueError('LTX --refresh-every must be 0 to 32.')
    defaults = {'sd-turbo': (512, 512), 'waypoint': (1280, 720), 'krea': (832, 480), 'ltx': (1024, 576)}
    output = tuple(map(int, arguments.resolution.split('x'))) if arguments.resolution else defaults[backend]
    allowed = {'sd-turbo': OUTPUT_RESOLUTIONS, 'waypoint': ((1280, 720),),
               'krea': ((832, 480),), 'ltx': LTX_OUTPUT_RESOLUTIONS}
    if output not in allowed[backend]:
        raise ValueError(f'Choose a supported startup resolution for {backend}.')
    if arguments.input_image and not arguments.input_image.expanduser().is_file():
        raise ValueError('LTX --input-image must be an existing image file.')
    python = runtime_python(runtime)
    upstream = runtime / 'StreamDiffusion' if backend == 'sd-turbo' else runtime
    marker = runtime / 'workstation.json'
    if (not marker.exists() or not python.exists()
            or (backend == 'sd-turbo' and not (upstream / 'utils/wrapper.py').exists())):
        raise RuntimeError('Run workstation.py setup first, using the same --runtime folder.')
    configuration = json.loads(marker.read_text(encoding='utf-8'))
    if (configuration.get('managed_by') != 'colab-sd' or configuration.get('backend', 'sd-turbo') != backend
            or configuration.get('ready') is not True):
        raise RuntimeError('Workstation setup is outdated or unmanaged. Run workstation.py setup again.')
    if backend == 'waypoint':
        environment = waypoint_environment(runtime, configuration, arguments)
    elif backend in ('krea', 'ltx'):
        environment = modern_environment(runtime, configuration, arguments)
    else:
        if configuration.get('revision') != UPSTREAM_REVISION:
            raise RuntimeError('Workstation setup is outdated. Run workstation.py setup again.')
        selected = arguments.model_cache or configuration.get('model_cache')
        environment = cache_environment(runtime, selected, legacy=True) if selected else {}
    gpu_check(python, upstream, backend=backend)
    ice_file = getattr(arguments, 'ice_servers_file', None)
    if ice_file:
        from webrtc_transport import validate_ice_servers
        if ice_file.stat().st_size > 32768:
            raise ValueError('ICE credentials file is too large')
        environment['BRIDGE_ICE_SERVERS_JSON'] = json.dumps(validate_ice_servers(
            json.loads(ice_file.read_text(encoding='utf-8'))))
    token = pairing_token(runtime)
    tunnel = arguments.transport == 'cloudflare'
    binary = cloudflared_binary(runtime) if tunnel else None
    host = arguments.bind or ('127.0.0.1' if tunnel else '0.0.0.0')
    session = BridgeSession(root=upstream, tensorrt=arguments.tensorrt, max_fps=arguments.max_fps,
                            host=host, port=arguments.port, tunnel=tunnel,
                            advertise_host=arguments.advertise_host or socket.gethostname(),
                            server_script=PROJECT / 'colab_server.py', python_executable=python,
                            cloudflared=binary, access_token=token, startup_timeout=arguments.startup_timeout,
                            output_size=output,
                            allow_tensorrt_resize=arguments.allow_tensorrt_resize,
                            engine_cache=arguments.engine_cache, reuse_engines=not arguments.no_engine_cache,
                            tunnel_protocol=arguments.tunnel_protocol, backend=backend, server_environment=environment,
                            diffusion_model=arguments.model or 'sd-turbo')
    try:
        url = session.start()
        print(f'\n{backend} workstation is ready. In TouchDesigner select RTX Workstation.')
        print('Server WebSocket URL:', url)
        print('Workstation Access Token:', token)
        if not tunnel:
            print('Use the workstation LAN/VPN IP if its hostname does not resolve; --advertise-host sets the printed address.')
        if backend == 'waypoint':
            print('World page: seed an input image' + (' or choose Prompt Model (A/B).' if configuration['prompt_seeds'] else '.'))
            print('Prompt seeds:', environment['WAYPOINT_PROMPT_SEED_MODEL'] if configuration['prompt_seeds'] else 'disabled',
                  '| Model cache:', environment['HF_HUB_CACHE'])
            print('Native 1280 x 720 |', environment['WAYPOINT_QUANT'], '| Compiler:', environment['WAYPOINT_COMPILE_PROFILE'])
        elif backend == 'krea':
            print('Krea page: Pause / Reset. Native 832 x 480 | INT8:', environment['KREA_INT8_PROFILE'],
                  '| Clip blocks:', environment['KREA_CLIP_BLOCKS'])
        elif backend == 'ltx':
            print('LTX page: Generate or Restart. Steady playback buffers two clips, then renders ahead.')
            print('Scene refresh can cause cuts. New prompts still load weights. Native',
                  f'{output[0]} x {output[1]} |', environment['LTX_PERFORMANCE_PROFILE'])
        print('Ctrl+C stops this server and its tunnel. Logs:', upstream / 'bridge_logs', flush=True)
        session.wait()
    finally:
        session.stop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    install = commands.add_parser('setup', help='Install a separate CUDA runtime for the selected backend')
    install.add_argument('--backend', choices=tuple(RUNTIMES), default='sd-turbo')
    install.add_argument('--runtime', type=Path, help='Default: .workstation or .workstation-BACKEND')
    install.add_argument('--cuda', help='SD-Turbo CUDA wheel tag (default: cu130); other backends pin cu130')
    install.add_argument('--torch-version', help=f'SD-Turbo default: {TORCH_VERSION}; Waypoint/LTX pin 2.11.0; Krea pins 2.13.0')
    install.add_argument('--prompt-seeds', action='store_true', help='Waypoint: install and enable optional Turbo prompt seeds')
    install.add_argument('--prompt-seed-model', choices=PROMPT_SEED_MODELS,
                         help='Waypoint seed generator (default: sd-turbo); requires --prompt-seeds')
    install.add_argument('--model-cache', type=Path, help='Persistent model cache (default: runtime/models); auth stays outside')
    launch = commands.add_parser('start', help='Run generation with LAN/VPN or Cloudflare transport')
    launch.add_argument('--backend', choices=tuple(RUNTIMES), default='sd-turbo')
    launch.add_argument('--model', choices=('sd-turbo', 'sdxl-turbo'),
                        help='StreamDiffusion model (default: sd-turbo); selection applies at startup')
    launch.add_argument('--prompt-seed-model', choices=PROMPT_SEED_MODELS,
                        help='Waypoint seed model override for this launch; setup must enable prompt seeds')
    launch.add_argument('--runtime', type=Path, help='Use the matching setup runtime folder')
    launch.add_argument('--quantization', choices=('intw8a8', 'bf16'), help='Waypoint only (default: intw8a8)')
    launch.add_argument('--compile-profile', choices=('fast', 'max-autotune'), help='Waypoint only (default: fast)')
    launch.add_argument('--model-cache', type=Path, help='Model cache override for this launch, for any backend')
    launch.add_argument('--ice-servers-file', type=Path,
                        help='Private JSON ICE server list (STUN/TURN); credentials stay out of URLs and logs')
    launch.add_argument('--int8-profile', choices=('fast', 'balanced'), help='Krea only (default: fast)')
    launch.add_argument('--clip-blocks', type=int, help='Krea only, 1-64 (default: 64)')
    launch.add_argument('--clip-frames', type=int, choices=LTX_FRAME_COUNTS, help='LTX only (default: 33)')
    launch.add_argument('--performance-profile', choices=('resident', 'low_memory'), help='LTX only (default: resident)')
    launch.add_argument('--continuous', action=argparse.BooleanOptionalAction, default=None,
                        help='LTX automatic generation after Generate (default: enabled)')
    launch.add_argument('--continue-scene', action=argparse.BooleanOptionalAction, default=None,
                        help='LTX condition on previous ending frame (default: enabled)')
    launch.add_argument('--refresh-every', type=int, help='LTX scene refresh interval, 0-32 clips (default: 4; 0 disables)')
    launch.add_argument('--input-image', type=Path, help='LTX startup image; empty uses Prompt A text-to-video')
    launch.add_argument('--transport', choices=('lan', 'cloudflare'), default='lan')
    launch.add_argument('--tunnel-protocol', choices=TUNNEL_PROTOCOLS, default='http2',
                        help='Cloudflare transport (default: http2, matching Colab); compare auto/quic for your route')
    launch.add_argument('--bind', help='Listening address (default: all interfaces for LAN, loopback for Cloudflare)')
    launch.add_argument('--port', type=int, default=8000)
    launch.add_argument('--advertise-host', help='Workstation LAN/VPN IP or DNS name shown in the client URL')
    launch.add_argument('--max-fps', type=float, help='Default: SD-Turbo 60; Waypoint 20; Krea/LTX 24')
    launch.add_argument('--tensorrt', action='store_true', help='Compile both UNets; workstation GPU validation required')
    launch.add_argument('--resolution', choices=[*[f'{w}x{h}' for w, h in OUTPUT_RESOLUTIONS], '1280x720', '832x480', '768x512'],
                        help='Startup size; default SD 512x512, World 1280x720, Krea 832x480, LTX 1024x576')
    launch.add_argument('--allow-tensorrt-resize', action='store_true', help='Allow live profile loads/builds; streaming pauses')
    launch.add_argument('--engine-cache', type=Path, help='Persistent full-engine profile cache directory')
    launch.add_argument('--no-engine-cache', action='store_true', help='Rebuild immutable engines instead of reusing archives')
    launch.add_argument('--startup-timeout', type=float, default=None)
    arguments = parser.parse_args(argv)
    arguments.runtime = arguments.runtime or RUNTIMES[arguments.backend]
    try:
        if arguments.command == 'setup':
            if arguments.backend == 'waypoint':
                if arguments.prompt_seed_model is not None and not arguments.prompt_seeds:
                    raise ValueError('--prompt-seed-model requires --prompt-seeds during setup.')
                if arguments.cuda not in (None, 'cu130') or arguments.torch_version not in (None, '2.11.0'):
                    raise ValueError('The pinned Waypoint runtime requires Torch 2.11.0 / cu130; SD-Turbo uses a separate environment.')
                setup_waypoint(arguments.runtime, arguments.prompt_seeds, arguments.model_cache,
                               prompt_seed_model=arguments.prompt_seed_model or 'sd-turbo')
            elif arguments.backend in ('krea', 'ltx'):
                if arguments.prompt_seeds or arguments.prompt_seed_model:
                    raise ValueError('Prompt seed setup options require --backend waypoint.')
                pinned_torch = '2.11.0' if arguments.backend == 'ltx' else '2.13.0'
                if arguments.cuda not in (None, 'cu130') or arguments.torch_version not in (None, pinned_torch):
                    raise ValueError(f'{arguments.backend} requires pinned Torch {pinned_torch} / cu130.')
                setup_modern(arguments.runtime, arguments.backend, arguments.model_cache)
            else:
                if arguments.prompt_seeds or arguments.prompt_seed_model:
                    raise ValueError('Prompt seed setup options require --backend waypoint.')
                options = {'model_cache': arguments.model_cache} if arguments.model_cache else {}
                setup(arguments.runtime, arguments.cuda or 'cu130', arguments.torch_version or TORCH_VERSION, **options)
        else:
            if arguments.max_fps is None:
                arguments.max_fps = {'sd-turbo': 60, 'waypoint': 20, 'krea': 24, 'ltx': 24}[arguments.backend]
            if arguments.startup_timeout is None and arguments.backend != 'sd-turbo':
                arguments.startup_timeout = 3600 if arguments.backend == 'ltx' else 1800
            start(arguments)
        return 0
    except KeyboardInterrupt:
        print('\nWorkstation bridge stopped.')
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f'Workstation bridge failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
