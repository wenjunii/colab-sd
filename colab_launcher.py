"""Own and monitor the Colab server and tunnel without blocking on log pipes."""
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


class BridgeSession:
    def __init__(self, tensorrt=False, max_fps=60, root=None, startup_timeout=None, tunnel_timeout=90):
        if not math.isfinite(float(max_fps)) or not 1 <= float(max_fps) <= 120:
            raise ValueError('Maximum FPS must be between 1 and 120')
        self.root = Path(root or Path.cwd()).resolve()
        self.tensorrt = bool(tensorrt)
        self.max_fps = float(max_fps)
        self.startup_timeout = startup_timeout or (1800 if tensorrt else 600)
        self.tunnel_timeout = tunnel_timeout
        self.processes = {}
        self.readers = {}
        self.writers = {}
        self.tunnel_url = None
        self.tunnel_tail = ''

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
                with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2) as response:
                    if json.load(response).get('ready') is True:
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
        try:
            with socket.create_connection(('127.0.0.1', 8000), timeout=0.5):
                raise RuntimeError('Port 8000 is already in use. Stop the previous server before launching this session.')
        except OSError:
            pass
        try:
            if self.tensorrt:
                result = subprocess.run([sys.executable, '-m', 'streamdiffusion.tools.install-tensorrt'], cwd=self.root)
                if result.returncode:
                    print('TensorRT setup failed; starting PyTorch FP16 instead.', flush=True)
                    self.tensorrt = False
            environment = os.environ.copy()
            environment['MAX_STREAM_FPS'] = str(self.max_fps)
            environment['ENABLE_TENSORRT'] = '1' if self.tensorrt else '0'
            self._spawn('server', [sys.executable, '-u', 'server.py'], environment)
            self.wait_for_ready()
            self._spawn('tunnel', [str(self.root / 'cloudflared'), 'tunnel', '--url', 'http://127.0.0.1:8000'])
            return self.wait_for_tunnel()
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
