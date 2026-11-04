"""Generate a separate Krea HF modular-runtime trial for Colab A100 40 GB+."""
import argparse
import json
from pathlib import Path
from textwrap import dedent

from generate_notebook import PROJECT_DIR, code_cell

SOURCES = ('colab_server.py', 'bridge_protocol.py', 'colab_launcher.py', 'tensorrt_backend.py',
           'colab_setup.py', 'waypoint_setup.py', 'krea_setup.py', 'krea_assets.py', 'krea_backend.py')


def make_notebook(project_dir=PROJECT_DIR):
    intro = '''# Krea Realtime → TouchDesigner (experimental, GPU validation pending)
Text-driven video through the official **Hugging Face modular runtime**, whose model card
declares **Apache-2.0**, with pinned Wan components (Apache-2.0). The similarly named
Krea GitHub application is **CC-BY-NC-SA** and is **not installed or copied** here.
Application code and model licenses are separate; retain their notices and check the linked
terms for your intended distribution. This trial is selected for the commercial-use requirement.

Use a **fresh A100 40 GB or larger GPU runtime**. Do not run these cells in the runtime
currently serving SD-Turbo or Waypoint. Setup refuses an occupied bridge port and never
stops another server. Its isolated environment does not replace notebook kernel packages.

1. Run Step 1. L4/T4 and GPUs with less than 35 GiB total / 30 GiB free VRAM are rejected.
2. Step 2 can store weights in Drive, under `StreamDiffusion/krea-models`. Missing files
   download once; later sessions reuse the pinned snapshots. Allow roughly 45 GB of storage.
3. Launch Step 3, open the updated TD project/component, paste the URL and enable Connect.
   The handshake selects **Krea Realtime** and fixed **832 × 480** output.
4. Edit **Prompt A** / pulse **Submit Prompts**. Changes apply at the next generated chunk.
   Use the **Krea** page to Pause or Reset Krea Clip. Seed changes restart the clip.

This first trial uses an **INT8 transformer**, CPU-offloaded text encoder, resident FP16 VAE,
four denoise steps, and PyTorch attention. INT8 preserves room for temporal caches on a 40 GB
GPU; memory use, speed and visual quality still require a real GPU test. No FPS is promised.
The model's published B200 benchmark does not establish A100 performance. First generation
may compile kernels; Status reports this separately from model-loaded readiness.

Only Prompt A and Seed steer this text-to-video mode. Prompt B/blending, negative prompts,
input video, denoise strength, audio reactivity and Waypoint movement controls are unused.
The pinned custom transformer assumes 832 × 480; other sizes are not enabled.
Generation is chunked (usually 9 frames first, then 12). GPU Inference shows **chunk time**.
At most one decoded chunk plus the selected delivery window is queued. Start with Fast
video quality and Video Buffer 3. Client acknowledgements gate further generation.

Clips are bounded to **64 chunks** by default (about 32 seconds of video at 24 FPS).
At the boundary, temporal history resets and the seed increments; a scene cut can occur.
This is a repeating-clip trial, not an uninterrupted infinite-history stream.
Pause waits for a running chunk to finish. Generation errors stop retries until Reset.
Keep the Colab launch cell and TouchDesigner running; interrupt the cell to stop only its
owned server/tunnel. Drive preserves weights, not GPU state or first-use compilation.

[HF runtime/model and declared license](https://huggingface.co/krea/krea-realtime-video/tree/9a2ceea280e5d584c6a28d384a9c19e38c810f8c) ·
[Wan components/license](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/tree/38ec498cb3208fb688890f8cc7e94ede2cbd7f68) ·
[Bridge source and validation](https://github.com/wenjunii/colab-sd)
'''
    check = dedent('''\
        # Step 1: Check the fresh GPU runtime before installing/downloading
        import torch
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
            raise RuntimeError('Select an A100 40 GB or larger GPU runtime (Ampere+).')
        free, total = torch.cuda.mem_get_info()
        print(torch.cuda.get_device_name(), f'{free/1024**3:.1f}/{total/1024**3:.1f} GiB free/total')
        if total < 35 * 1024**3 or free < 30 * 1024**3:
            raise RuntimeError('Krea trial needs 35 GiB total / 30 GiB free. L4/T4 are not enabled.')
    ''')
    setup = dedent('''\
        # @title Step 2: Install isolated Krea runtime and choose storage
        USE_GOOGLE_DRIVE = True # @param {type:"boolean"}
        DRIVE_MODEL_FOLDER = "StreamDiffusion/krea-models" # @param {type:"string"}
        import os
        import sys
        import socket
        from pathlib import Path
        # Refuse before writing files or changing caches; leave any active bridge alone.
        try:
            connection = socket.create_connection(('127.0.0.1', 8000), timeout=.5)
        except OSError:
            pass
        else:
            connection.close()
            raise RuntimeError('An active bridge uses port 8000. Use a fresh GPU runtime for Krea.')
        runtime_root = Path('/content/colab-sd-krea')
        runtime_root.mkdir(exist_ok=True)
    ''')
    sources = {name: (Path(project_dir) / name).read_text(encoding='utf-8') for name in SOURCES}
    setup += 'sources = ' + repr(sources) + '\n'
    setup += dedent('''\
        for name, content in sources.items():
            (runtime_root / name).write_text(content, encoding='utf-8')
        sys.path.insert(0, str(runtime_root))
        import importlib
        import colab_setup
        import krea_setup
        importlib.reload(colab_setup)
        importlib.reload(krea_setup)
        colab_setup.configure_model_cache(USE_GOOGLE_DRIVE, DRIVE_MODEL_FOLDER)
        os.environ.pop('TRANSFORMERS_CACHE', None)
        os.environ['HF_MODULES_CACHE'] = str(runtime_root / 'modules-cache')
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(runtime_root / 'inductor-cache')
        os.environ['DIFFUSERS_ENABLE_HUB_KERNELS'] = '0'
        os.environ['DISABLE_SAGEATTENTION'] = '1'
        krea_python, krea_tunnel = krea_setup.setup(runtime_root)
        print('Krea environment ready. Run Step 3; model loading begins there.')
    ''')
    launch = dedent('''\
        # @title Step 3: Launch Krea and Cloudflare
        MAX_FPS = 24 # @param {type:"integer"}
        CLIP_CHUNKS = 64 # @param {type:"integer"}
        TUNNEL_PROTOCOL = "http2" # @param ["http2", "auto", "quic"]
        if not 1 <= CLIP_CHUNKS <= 64:
            raise ValueError('CLIP_CHUNKS must be between 1 and 64.')
        if 'krea_session' in globals():
            krea_session.stop()
        import importlib
        import colab_launcher
        importlib.reload(colab_launcher)
        os.environ['KREA_CLIP_BLOCKS'] = str(CLIP_CHUNKS)
        krea_session = colab_launcher.BridgeSession(
            root=runtime_root, backend='krea', server_script='colab_server.py',
            python_executable=krea_python, cloudflared=krea_tunnel,
            startup_timeout=1800, max_fps=MAX_FPS, tunnel_protocol=TUNNEL_PROTOCOL)
        try:
            url = krea_session.start()
            print('Paste this URL into the updated TouchDesigner component:')
            print(url, flush=True)
            print('Use Prompt A / Submit Prompts, and the Krea page for Pause / Reset.')
            krea_session.wait()
        except KeyboardInterrupt:
            print('Stopping this Krea session...')
        finally:
            krea_session.stop()
    ''')
    return {'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': intro.splitlines(keepends=True)},
                      *[code_cell(source) for source in (check, setup, launch)]],
            'metadata': {'accelerator': 'GPU', 'colab': {'gpuType': 'A100', 'provenance': []},
                         'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 0}


def generate_notebook(project_dir=PROJECT_DIR, check=False):
    target = Path(project_dir) / 'Krea_Colab_Server.ipynb'
    content = json.dumps(make_notebook(project_dir), indent=2) + '\n'
    if check:
        if not target.exists() or target.read_text(encoding='utf-8') != content:
            raise RuntimeError('Krea notebook is stale. Run python generate_notebook.py.')
    else:
        temporary = target.with_suffix('.ipynb.tmp')
        temporary.write_text(content, encoding='utf-8', newline='\n')
        temporary.replace(target)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    print(generate_notebook(check=parser.parse_args().check))
