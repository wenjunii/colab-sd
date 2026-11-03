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
import urllib.request
import venv

from colab_launcher import BridgeSession, TUNNEL_PROTOCOLS
from colab_setup import CORE_DEPENDENCIES, UPSTREAM_REVISION
from bridge_protocol import OUTPUT_RESOLUTIONS

PROJECT = Path(__file__).resolve().parent
DEFAULT_RUNTIME = PROJECT / '.workstation'
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


def check_platform():
    if platform.system() not in ('Windows', 'Linux') or platform.machine().lower() not in ('amd64', 'x86_64'):
        raise RuntimeError('Workstation mode needs Windows or Linux x86-64 with an NVIDIA CUDA GPU.')
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError('Run this installer with Python 3.11 (Windows: py -3.11 workstation.py setup).')


def gpu_check(python, root):
    subprocess.run([str(python), '-c',
                    'import torch; '
                    'assert torch.cuda.is_available(), "CUDA unavailable: check NVIDIA driver and the selected PyTorch CUDA wheel"; '
                    'print("GPU:", torch.cuda.get_device_name(0), "| Torch:", torch.__version__, "| CUDA:", torch.version.cuda)'],
                   cwd=root, check=True)


def setup(runtime, cuda='cu130', torch_version=TORCH_VERSION):
    check_platform()
    if not re.fullmatch(r'cu\d{3}', cuda) or not re.fullmatch(r'\d+\.\d+\.\d+', torch_version):
        raise ValueError('Use a CUDA wheel tag such as cu130 and a Torch release such as 2.13.0.')
    runtime = Path(runtime).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    marker = runtime / 'workstation.json'
    if not marker.exists() and any(runtime.iterdir()):
        raise RuntimeError('Runtime folder is not empty or managed by this installer. Choose a new --runtime folder.')
    if not marker.exists():
        marker.write_text(json.dumps({'managed_by': 'colab-sd', 'revision': UPSTREAM_REVISION}) + '\n', encoding='utf-8')
    elif json.loads(marker.read_text(encoding='utf-8')).get('managed_by') != 'colab-sd':
        raise RuntimeError('Runtime folder belongs to another application.')
    marker.write_text(json.dumps({'managed_by': 'colab-sd', 'revision': UPSTREAM_REVISION, 'ready': False}) + '\n',
                      encoding='utf-8')
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
                                 'torch': torch_version, 'cuda': cuda}) + '\n', encoding='utf-8')
    print('Workstation setup complete. Run workstation.py start to load the models and stream.')


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
    python, upstream = runtime_python(runtime), runtime / 'StreamDiffusion'
    marker = runtime / 'workstation.json'
    if not marker.exists() or not python.exists() or not (upstream / 'utils/wrapper.py').exists():
        raise RuntimeError('Run workstation.py setup first, using the same --runtime folder.')
    configuration = json.loads(marker.read_text(encoding='utf-8'))
    if (configuration.get('managed_by') != 'colab-sd' or configuration.get('revision') != UPSTREAM_REVISION
            or configuration.get('ready') is not True):
        raise RuntimeError('Workstation setup is outdated or unmanaged. Run workstation.py setup again.')
    gpu_check(python, upstream)
    token = pairing_token(runtime)
    tunnel = arguments.transport == 'cloudflare'
    binary = cloudflared_binary(runtime) if tunnel else None
    host = arguments.bind or ('127.0.0.1' if tunnel else '0.0.0.0')
    session = BridgeSession(root=upstream, tensorrt=arguments.tensorrt, max_fps=arguments.max_fps,
                            host=host, port=arguments.port, tunnel=tunnel,
                            advertise_host=arguments.advertise_host or socket.gethostname(),
                            server_script=PROJECT / 'colab_server.py', python_executable=python,
                            cloudflared=binary, access_token=token, startup_timeout=arguments.startup_timeout,
                            output_size=tuple(map(int, arguments.resolution.split('x'))),
                            allow_tensorrt_resize=arguments.allow_tensorrt_resize,
                            engine_cache=arguments.engine_cache, reuse_engines=not arguments.no_engine_cache,
                            tunnel_protocol=arguments.tunnel_protocol)
    try:
        url = session.start()
        print('\nStreamDiffusion workstation is ready. In TouchDesigner select RTX Workstation.')
        print('Server WebSocket URL:', url)
        print('Workstation Access Token:', token)
        if not tunnel:
            print('Use the workstation LAN/VPN IP if its hostname does not resolve; --advertise-host sets the printed address.')
        print('Ctrl+C stops this server and its tunnel. Logs:', upstream / 'bridge_logs', flush=True)
        session.wait()
    finally:
        session.stop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    install = commands.add_parser('setup', help='Create a dedicated runtime and install CUDA StreamDiffusion')
    install.add_argument('--runtime', type=Path, default=DEFAULT_RUNTIME)
    install.add_argument('--cuda', default='cu130', help='Official PyTorch wheel index tag (default: cu130)')
    install.add_argument('--torch-version', default=TORCH_VERSION)
    launch = commands.add_parser('start', help='Run generation with LAN/VPN or Cloudflare transport')
    launch.add_argument('--runtime', type=Path, default=DEFAULT_RUNTIME)
    launch.add_argument('--transport', choices=('lan', 'cloudflare'), default='lan')
    launch.add_argument('--tunnel-protocol', choices=TUNNEL_PROTOCOLS, default='http2',
                        help='Cloudflare transport (default: http2, matching Colab); compare auto/quic for your route')
    launch.add_argument('--bind', help='Listening address (default: all interfaces for LAN, loopback for Cloudflare)')
    launch.add_argument('--port', type=int, default=8000)
    launch.add_argument('--advertise-host', help='Workstation LAN/VPN IP or DNS name shown in the client URL')
    launch.add_argument('--max-fps', type=float, default=60)
    launch.add_argument('--tensorrt', action='store_true', help='Compile both UNets; workstation GPU validation required')
    launch.add_argument('--resolution', choices=[f'{w}x{h}' for w, h in OUTPUT_RESOLUTIONS], default='512x512')
    launch.add_argument('--allow-tensorrt-resize', action='store_true', help='Allow live profile loads/builds; streaming pauses')
    launch.add_argument('--engine-cache', type=Path, help='Persistent full-engine profile cache directory')
    launch.add_argument('--no-engine-cache', action='store_true', help='Rebuild immutable engines instead of reusing archives')
    launch.add_argument('--startup-timeout', type=float, default=None)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == 'setup':
            setup(arguments.runtime, arguments.cuda, arguments.torch_version)
        else:
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
