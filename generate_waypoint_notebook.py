"""Generate the isolated, self-contained open-source Waypoint Colab notebook."""
import argparse
import json
from pathlib import Path
from textwrap import dedent

from generate_notebook import PROJECT_DIR, code_cell

SOURCES = ('colab_server.py', 'bridge_protocol.py', 'colab_launcher.py',
           'tensorrt_backend.py', 'colab_setup.py', 'waypoint_backend.py', 'waypoint_setup.py', 'waypoint_seed.py')


def make_notebook(project_dir=PROJECT_DIR):
    root = Path(project_dir)
    intro = '''# Waypoint world model → TouchDesigner (experimental)
Open-source, image-seeded interactive video, targeting Colab A100 40 GB.
No Scope service or package is required. Model: Apache-2.0. Inference library: GPL-3.0.

Run the cells in order in a **fresh GPU runtime**. This notebook installs into a separate
environment; it does not install StreamDiffusion or replace the notebook kernel packages.
Setup creates the environment without ensurepip, then bootstraps pip into that interpreter.
Rerunning setup repairs a partial pip-less environment while retaining installed packages.
Do not run both GPU servers on the same GPU simultaneously.

1. Optional Drive storage reuses pinned model and VAE weights on later runtimes.
2. Launch with INT8 for Ampere/A100; BF16 is available for comparison.
3. Open the updated project/component in TouchDesigner, paste the URL and enable Connect.
   The handshake selects **World Model** and native **1280 × 720** output.
4. Choose an Input Source containing a scene image. Pulse **Seed World from Input**.
   The source is center-cropped to 16:9. First generation compiles kernels and can take minutes.
5. Set **World Button Codes** (comma-separated Owl-Control codes) and **Camera X/Y per Step**;
   values are held until changed. Set them to zero/empty to stop motion. Pause and Reset reuse weights.

This checkpoint has no direct text conditioning. For prompt-driven starting images, enable
**ENABLE_PROMPT_SEEDS** in Step 2 and choose **PROMPT_SEED_MODEL = sd-turbo / sdxl-turbo**.
In the updated TD component select **World Seed Source = Prompt Model (A/B)**.
The selected model generates a 768 × 432 seed inside Colab; Waypoint
animates it at native 720p. **Auto Reseed** defaults On, with an adjustable **5-second**
interval. Off keeps the current world until **Generate Seed from Prompt** / **Submit Prompts**.
Each reseed resets world history and can cause a scene cut. The timer starts after the first
usable batch and waits for output credit/batch boundaries; slow delivery can extend it.
Both models share one GPU worker. Combined GPU performance/memory use still need validation;
earlier Waypoint-only measurements do not establish this new mode's performance.
SDXL-Turbo blends both token and pooled text embeddings; only one seed model is loaded.
Its original VAE and both text encoders are retained for the seed pipeline. It uses more
GPU memory than SD-Turbo. Combined A100 memory, quality and speed need live validation.
Negative prompts, denoise strength, and audio reactivity do not steer prompt seeds.
Model selection applies at server restart. Existing cached weights are preserved.
Both seed models have separate Stability AI commercial-use terms; the bridge's MIT
license does not replace the model licenses: https://stability.ai/license.
Waypoint's native output is 720p, not 1024 × 768.
It generates four frames per controller step. GPU Inference shows the whole batch time.
Start with Fast video quality and Video Buffer 3. HTTP/2 is the default tunnel transport.
The default **fast** compiler profile keeps full-graph compilation but disables exhaustive
autotuning, coordinate-descent tuning, and CUDA-graph capture. **max-autotune** restores the
upstream policy for throughput comparisons after a successful first-frame test; it can take
much longer to start. Neither profile changes native resolution or quantization.
For INT8, fast also resets GemLite's independent tuner to its default after model construction.
Cold quantization warmup otherwise leaves exhaustive GemLite tuning enabled, even when
TorchInductor tuning is disabled. Initial quantization warmup and first-use compilation remain.
The updated TouchDesigner Status and server `/health` report seed preparation and first-batch
generation separately, with elapsed time. Model-loaded readiness does not mean a frame is ready.

Keep the Colab launch cell and TouchDesigner running. Interrupt the launch cell to stop its
server and tunnel. Drive persists weights, not the running GPU or compiled TensorRT engines.
Waypoint uses TorchInductor, not the SD-Turbo TensorRT cache. First-run compilation is still
required in a fresh runtime; cross-VM compiled-kernel reuse is not enabled here.

Basic native-720p INT8 generation was tested on an A100 40 GB with the fast profile and an
already-used VM/cache. Fresh-VM startup, long-running world quality, other profiles and sustained
TouchDesigner FPS still need validation. See the README measurements; this notebook does not
promise an A100 frame rate.

[Model](https://huggingface.co/Overworld/Waypoint-1.5-1B) ·
[Runtime and licenses](https://github.com/Overworldai/world_engine) ·
[Bridge source](https://github.com/wenjunii/colab-sd)
'''
    check = dedent('''\
        # Step 1: Check the GPU
        import subprocess
        import torch
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
            raise RuntimeError('Select an A100 GPU runtime (Ampere or newer is required).')
        subprocess.run(['nvidia-smi'], check=True)
    ''')
    setup = dedent('''\
        # @title Step 2: Install isolated Waypoint runtime and choose storage
        # @markdown Repairs partial environment setup without ensurepip; keeps existing model caches.
        USE_GOOGLE_DRIVE = False # @param {type:"boolean"}
        DRIVE_MODEL_FOLDER = "StreamDiffusion/waypoint-models" # @param {type:"string"}
        # @markdown Optional prompt-to-world pipeline. Both models share one GPU worker; new seeds reset the world.
        ENABLE_PROMPT_SEEDS = False # @param {type:"boolean"}
        PROMPT_SEED_MODEL = "sd-turbo" # @param ["sd-turbo", "sdxl-turbo"]

        import os
        import sys
        from pathlib import Path
        runtime_root = Path('/content/colab-sd-waypoint')
        runtime_root.mkdir(exist_ok=True)
    ''')
    sources = {name: (root / name).read_text(encoding='utf-8') for name in SOURCES}
    setup += 'sources = ' + repr(sources) + '\n'
    setup += dedent('''\
        for name, content in sources.items():
            (runtime_root / name).write_text(content, encoding='utf-8')
        sys.path.insert(0, str(runtime_root))
        import importlib
        import colab_setup
        import waypoint_setup
        importlib.reload(colab_setup)
        importlib.reload(waypoint_setup)
        colab_setup.configure_model_cache(USE_GOOGLE_DRIVE, DRIVE_MODEL_FOLDER)
        # Modern Hub uses HF_HUB_CACHE; avoid the legacy Transformers cache alias.
        os.environ.pop('TRANSFORMERS_CACHE', None)
        waypoint_python, waypoint_tunnel = waypoint_setup.setup(runtime_root, prompt_seeds=ENABLE_PROMPT_SEEDS)
        os.environ['WAYPOINT_PROMPT_SEEDS'] = '1' if ENABLE_PROMPT_SEEDS else '0'
        os.environ['WAYPOINT_PROMPT_SEED_MODEL'] = PROMPT_SEED_MODEL
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(runtime_root / 'inductor-cache')
        print('Waypoint environment ready. Run Step 3.')
    ''')
    launch = dedent('''\
        # @title Step 3: Launch Waypoint and Cloudflare
        QUANTIZATION = "intw8a8" # @param ["intw8a8", "bf16"]
        # @markdown Fast reduces first-frame compilation work. Max-autotune trades startup time for possible throughput.
        COMPILE_PROFILE = "fast" # @param ["fast", "max-autotune"]
        MAX_FPS = 20 # @param {type:"integer"}
        TUNNEL_PROTOCOL = "http2" # @param ["http2", "auto", "quic"]
        if 'waypoint_session' in globals():
            waypoint_session.stop()
        import importlib
        import colab_launcher
        importlib.reload(colab_launcher)
        os.environ['WAYPOINT_QUANT'] = QUANTIZATION
        os.environ['WAYPOINT_COMPILE_PROFILE'] = COMPILE_PROFILE
        os.environ['WAYPOINT_PROMPT_SEED_MODEL'] = PROMPT_SEED_MODEL
        waypoint_session = colab_launcher.BridgeSession(
            root=runtime_root, backend='waypoint', server_script='colab_server.py',
            python_executable=waypoint_python, cloudflared=waypoint_tunnel,
            startup_timeout=1800, max_fps=MAX_FPS, tunnel_protocol=TUNNEL_PROTOCOL)
        try:
            url = waypoint_session.start()
            print('Paste this URL into the updated TouchDesigner component:')
            print(url, flush=True)
            print('World page: choose Prompt Model (A/B) if enabled, or Seed World from Input.')
            waypoint_session.wait()
        except KeyboardInterrupt:
            print('Stopping this Waypoint session...')
        finally:
            waypoint_session.stop()
    ''')
    return {'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': intro.splitlines(keepends=True)},
                      *[code_cell(source) for source in (check, setup, launch)]],
            'metadata': {'accelerator': 'GPU', 'colab': {'gpuType': 'A100', 'provenance': []},
                         'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 0}


def generate_notebook(project_dir=PROJECT_DIR, check=False):
    target = Path(project_dir) / 'Waypoint_Colab_Server.ipynb'
    content = json.dumps(make_notebook(project_dir), indent=2) + '\n'
    if check:
        if not target.exists() or target.read_text(encoding='utf-8') != content:
            raise RuntimeError('Waypoint notebook is stale. Run python generate_notebook.py.')
    else:
        temporary = target.with_suffix('.ipynb.tmp')
        temporary.write_text(content, encoding='utf-8', newline='\n')
        temporary.replace(target)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    print(generate_notebook(check=parser.parse_args().check))
