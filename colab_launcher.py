"""Own and monitor a bridge server and optional tunnel on Colab or a workstation."""
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from bridge_protocol import resolution

TUNNEL_PROTOCOLS = ('auto', 'http2', 'quic')


class BridgeSession:
    def __init__(self, tensorrt=False, max_fps=60, root=None, startup_timeout=None, tunnel_timeout=90,
                 host='0.0.0.0', port=8000, tunnel=True, advertise_host=None,
                 server_script='server.py', python_executable=None, cloudflared=None, access_token=None,
                 output_size=(512, 512), allow_tensorrt_resize=False, engine_cache=None, reuse_engines=True,
                 tunnel_protocol='auto', backend='sd-turbo', server_environment=None, diffusion_model='sd-turbo'):
        if backend not in ('sd-turbo', 'waypoint', 'krea', 'ltx'):
            raise ValueError('Backend must be sd-turbo, waypoint, krea, or ltx')
        if backend != 'sd-turbo' and tensorrt:
            raise ValueError('This backend uses its own runtime, not the SD-Turbo TensorRT engines')
        self.backend = backend
        if diffusion_model not in ('sd-turbo', 'sdxl-turbo'):
            raise ValueError('Diffusion model must be sd-turbo or sdxl-turbo')
        if backend != 'sd-turbo' and diffusion_model != 'sd-turbo':
            raise ValueError('Diffusion model selection applies only to the StreamDiffusion backend')
        self.diffusion_model = diffusion_model
        self._expected_prompt_seed_model = None
        # Per-launch options belong to the child, not the caller's process.
        # None removes an inherited variable (e.g. a legacy model-cache alias).
        self.server_environment = dict(server_environment or {})
        if any(not isinstance(key, str) or not key or '=' in key or '\0' in key
               or (value is not None and (not isinstance(value, str) or '\0' in value))
               for key, value in self.server_environment.items()):
            raise ValueError('Server environment must map valid names to strings or None')
        if not math.isfinite(float(max_fps)) or not 1 <= float(max_fps) <= 120:
            raise ValueError('Maximum FPS must be between 1 and 120')
        self.root = Path(root or Path.cwd()).resolve()
        if not 1 <= int(port) <= 65535:
            raise ValueError('Port must be between 1 and 65535')
        self.host, self.port = host, int(port)
        self.tunnel = bool(tunnel)
        if tunnel_protocol not in TUNNEL_PROTOCOLS:
            raise ValueError('Tunnel protocol must be auto, http2, or quic')
        self.tunnel_protocol = tunnel_protocol
        self.advertise_host = advertise_host or ('127.0.0.1' if host == '0.0.0.0' else '::1' if host == '::' else host)
        self.health_host = '127.0.0.1' if host == '0.0.0.0' else '::1' if host == '::' else host
        self.server_script = (self.root / server_script).resolve()
        self.python_executable = str(python_executable or sys.executable)
        self.cloudflared = str(cloudflared or self.root / ('cloudflared.exe' if os.name == 'nt' else 'cloudflared'))
        self.access_token = access_token
        self.session_id = os.urandom(16).hex()
        if backend == 'ltx':
            from ltx_backend import SIZES
            if tuple(output_size) not in SIZES:
                raise ValueError('Choose an LTX startup resolution preset')
        self.output_size = ((1280, 720) if backend == 'waypoint' else (832, 480) if backend == 'krea'
                            else tuple(output_size) if backend == 'ltx'
                            else resolution({'width': output_size[0], 'height': output_size[1]}))
        self.allow_tensorrt_resize = bool(allow_tensorrt_resize)
        self.engine_cache = Path(engine_cache or self.root / 'engines/torch-tensorrt').resolve()
        self.reuse_engines = bool(reuse_engines)
        self.tensorrt = bool(tensorrt)
        self.max_fps = float(max_fps)
        self.startup_timeout = startup_timeout or (1800 if tensorrt else 600)
        self.tunnel_timeout = tunnel_timeout
        self.processes = {}
        self.readers = {}
        self.writers = {}
        self.tunnel_url = None
        self.tunnel_tail = ''

    def endpoint(self, host=None, scheme='http', path=''):
        host = host or self.health_host
        host = f'[{host}]' if ':' in host and not host.startswith('[') else host
        return f'{scheme}://{host}:{self.port}{path}'

    def _spawn(self, name, command, env=None):
        logs = self.root / 'bridge_logs'
        logs.mkdir(exist_ok=True)
        path = logs / (name + '.log')
        writer = path.open('w', encoding='utf-8')
        self.writers[name] = writer
        # Child processes write to regular files, so a quiet server or a full
        # unread pipe cannot stall notebook startup or tunnel operation.
        process = subprocess.Popen(command, cwd=self.root, stdout=writer,
                                   stderr=subprocess.STDOUT, env=env)
        self.processes[name] = process
        self.readers[name] = path.open('r', encoding='utf-8', errors='replace')
        return process

    def pump_logs(self):
        for name, reader in self.readers.items():
            chunk = reader.read(65536)
            if chunk:
                print(f'[{name}] {chunk}', end='' if chunk.endswith('\n') else '\n', flush=True)
                if name == 'tunnel':
                    self.tunnel_tail = (self.tunnel_tail + chunk)[-65536:]
                    match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', self.tunnel_tail)
                    if match:
                        self.tunnel_url = match.group(0)

    def _check_processes(self):
        for name, process in self.processes.items():
            code = process.poll()
            if code is not None:
                raise RuntimeError(f'{name} exited with code {code}. See {self.root / "bridge_logs" / (name + ".log")}')

    def wait_for_ready(self):
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            self.pump_logs()
            self._check_processes()
            try:
                with urllib.request.urlopen(self.endpoint(path='/health'), timeout=2) as response:
                    status = json.load(response)
                    if status.get('ready') is True:
                        # A process that wins a bind race must not be published
                        # as this launcher's successfully started GPU server.
                        if status.get('session_id') != self.session_id:
                            raise RuntimeError('Health endpoint belongs to a different server session.')
                        if self.backend != 'sd-turbo' and status.get('backend') != self.backend:
                            raise RuntimeError(f'{self.backend} was requested but the server reports a different backend.')
                        if self.diffusion_model == 'sdxl-turbo' and status.get('diffusion_model') != self.diffusion_model:
                            raise RuntimeError('SDXL-Turbo was requested but the server reports a different model.')
                        if (self._expected_prompt_seed_model == 'sdxl-turbo'
                                and status.get('world_seed_model') != 'sdxl-turbo'):
                            raise RuntimeError('SDXL-Turbo prompt seeds were requested but are not active; update the Waypoint server scripts.')
                        if self.tensorrt and not str(status.get('engine', '')).startswith('TensorRT'):
                            raise RuntimeError('TensorRT was requested but is not active. Inspect bridge_logs/server.log; no endpoint was published.')
                        return
            except (OSError, ValueError, urllib.error.URLError):
                pass
            time.sleep(0.2)
        raise TimeoutError('Server startup timed out. Review bridge_logs/server.log; TensorRT compilation may need more time.')

    def wait_for_tunnel(self):
        deadline = time.monotonic() + self.tunnel_timeout
        while time.monotonic() < deadline:
            self.pump_logs()
            self._check_processes()
            if self.tunnel_url:
                return self.tunnel_url.replace('https://', 'wss://') + '/ws/stream'
            time.sleep(0.2)
        raise TimeoutError('Cloudflare did not provide a tunnel URL. Review bridge_logs/tunnel.log.')

    def start(self):
        self.stop()
        self.tunnel_url = None
        self.tunnel_tail = ''
        self._expected_prompt_seed_model = None
        try:
            with socket.create_connection((self.health_host, self.port), timeout=0.5):
                raise RuntimeError(f'Port {self.port} is already in use. Stop the previous server before launching this session.')
        except OSError:
            pass
        try:
            environment = os.environ.copy()
            for key, value in self.server_environment.items():
                if value is None:
                    environment.pop(key, None)
                else:
                    environment[key] = value
            environment['BRIDGE_BACKEND'] = self.backend
            if self.backend in ('krea', 'ltx'):
                from krea_quant import configure_allocator
                configure_allocator(environment)
                # Read by Hub at import time in this child, before any downloads.
                environment['HF_HUB_DISABLE_SYMLINKS'] = '1'
                environment['HF_HUB_DISABLE_SHARED_BLOBS'] = '1'
            environment['BRIDGE_DIFFUSION_MODEL'] = self.diffusion_model
            if self.backend == 'waypoint' and environment.get('WAYPOINT_PROMPT_SEEDS') == '1':
                selected = environment.get('WAYPOINT_PROMPT_SEED_MODEL', 'sd-turbo')
                if selected not in ('sd-turbo', 'sdxl-turbo'):
                    raise ValueError('Prompt seed model must be sd-turbo or sdxl-turbo')
                self._expected_prompt_seed_model = selected
            environment['MAX_STREAM_FPS'] = str(self.max_fps)
            environment['ENABLE_TENSORRT'] = '1' if self.tensorrt else '0'
            environment['BRIDGE_HOST'] = self.host
            environment['BRIDGE_PORT'] = str(self.port)
            environment['BRIDGE_SESSION_ID'] = self.session_id
            environment['BRIDGE_INITIAL_WIDTH'], environment['BRIDGE_INITIAL_HEIGHT'] = map(str, self.output_size)
            environment['BRIDGE_TRT_ALLOW_RESIZE'] = '1' if self.allow_tensorrt_resize else '0'
            environment['BRIDGE_TRT_CACHE_DIR'] = str(self.engine_cache)
            environment['BRIDGE_TRT_ENGINE_REUSE'] = '1' if self.reuse_engines else '0'
            if self.access_token is not None:
                environment['BRIDGE_ACCESS_TOKEN'] = self.access_token
            if self.tensorrt:
                if Path(self.python_executable).resolve() == Path(sys.executable).resolve():
                    from tensorrt_backend import ensure_dependencies
                    ensure_dependencies(self.root)
                else:
                    subprocess.run([self.python_executable, '-c',
                                    'import sys; sys.path.insert(0, sys.argv[1]); '
                                    'from tensorrt_backend import ensure_dependencies; ensure_dependencies(sys.argv[2])',
                                    str(self.server_script.parent), str(self.root)], check=True, env=environment)
            self._spawn('server', [self.python_executable, '-u', str(self.server_script)], environment)
            self.wait_for_ready()
            if self.tunnel:
                print('Cloudflare tunnel protocol:', self.tunnel_protocol, flush=True)
                self._spawn('tunnel', [self.cloudflared, 'tunnel', '--protocol',
                                       self.tunnel_protocol, '--url', self.endpoint()])
                return self.wait_for_tunnel()
            return self.endpoint(self.advertise_host, 'ws', '/ws/stream')
        except BaseException:
            self.stop()
            raise

    def wait(self):
        while self.processes:
            self.pump_logs()
            self._check_processes()
            time.sleep(0.2)

    def stop(self):
        # Only stop children owned by this instance, rather than killing any
        # unrelated process that happens to hold port 8000.
        for process in reversed(list(self.processes.values())):
            if process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                except ProcessLookupError:
                    pass
        self.processes.clear()
        for handle in list(self.readers.values()) + list(self.writers.values()):
            handle.close()
        self.readers.clear()
        self.writers.clear()
