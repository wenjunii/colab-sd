"""Generate the isolated LTX-2.5 Colab clip-generation trial."""
import json
from pathlib import Path
from textwrap import dedent
from generate_notebook import PROJECT_DIR, STREAMING_GUIDE, code_cell

SOURCES = ('colab_server.py', 'bridge_protocol.py', 'colab_launcher.py', 'webrtc_transport.py', 'tensorrt_backend.py',
           'colab_setup.py', 'waypoint_setup.py', 'krea_setup.py', 'krea_quant.py', 'krea_weights.py',
           'ltx_assets.py', 'ltx_setup.py', 'ltx_runtime.py', 'ltx_backend.py', 'check_ltx_gpu.py', 'check_ltx_stream.py')


def make_notebook(project_dir=PROJECT_DIR):
    intro = '''# LTX-2.5 → TouchDesigner (experimental clip trial)
Use an **A100 40 GB+**, with 30 GiB free VRAM. The Apache-2.0 LTX application code
is separate from the **LTX-2.x Community License** for model weights. Eligible entities
below $10M annual revenue can use the model commercially under its conditions;
entities at or above $10M generally need a paid license. Revenue includes affiliates
under common control. Your Hugging Face account must already have accepted model access.
[Model terms](https://github.com/Lightricks/LTX-2/blob/4f521619ae0029e8fbb64d555030c2b121a33b4f/LICENSE-2_x)
apply independently of the bridge's MIT and runtime's Apache licenses.

1. Run Step 1, then set up a **read token** in Colab Secrets named `HF_TOKEN`,
   with Notebook access enabled. Keep the token out of notebook source and outputs.
2. Run Step 2. The runtime uses its own `/content/colab-sd-ltx/venv`; models default
   to Drive `MyDrive/StreamDiffusion/ltx25-models`. Five pinned checkpoints total
   **71.1 GB (66.2 GiB)**. Allow at least 72 GB in Drive and 72 GB on the VM for
   validated local staging, plus environment space. Without Drive, allow both copies
   on VM disk. Valid files are reused; other backend caches are preserved.
3. Step 3 chooses actual generation resolution and clip length. The default is
   **1024 × 576 (16:9) / 33 frames**. Other choices are 512 × 512, 768 × 512 and
   1024 × 768. Use 512 × 512 for a smaller initial GPU test; performance is unverified.
   Both dimensions must be multiples of 64 for this two-stage pipeline.
   Leave `INPUT_IMAGE` empty for text-to-video, or upload an image through Colab Files
   and provide its absolute path for image-to-video. It is a startup setting.
4. Open the updated TouchDesigner `.toe` / `.tox`, paste the printed URL and Connect.
   The handshake selects **LTX-2.5** and the locked startup resolution. Set Prompt A
   and Seed, then pulse **Generate or Restart** on the **LTX** page. With **Continuous
   Generation** On, new clips render automatically with advancing seeds. **Continue
   Scene from Last Frame** conditions each next clip on the previous ending image.
   Turn it Off for independent clips. Continuous playback first buffers two clips,
   measures warm rendering speed, and then plays at a sustainable, slower rate while
   the GPU renders one clip ahead. No old clips are replayed. Pause gates delivery
   immediately and stops scheduling after current GPU work finishes.
   Another Generate pulse restarts from Prompt A / Seed and the startup image.

This uses the official distilled **two-stage** pipeline, FP8-cast transformer storage,
the lighter **convolutional VAE** and a one-prompt embedding cache. The default
**resident** profile keeps the quantized transformer on GPU across stages and clips;
the text encoder loads on GPU only for a new prompt and is released after encoding.
A changed prompt releases the transformer before building the text encoder, so the
two large models are not held together. **low_memory** streams both large models
from CPU while retaining the one-prompt cache. FP8-cast saves weight memory;
it does not claim native FP8 compute acceleration on A100. No TensorRT, NATTEN, CUDA
kernel build, prompt enhancer or quality-detailing LoRA is installed. Upstream also
generates audio internally; this trial transmits video only. Prompt B/blend, negative
prompts, denoise strength, live TOP input and World/Krea controls do not steer this mode.

GPU Inference is **whole-clip time**. Generation Speed is actual new FPS; **Steady
Playback** reports the slower delivery rate. Continuous playback uses the slowest
of the last four warm renders with a 20% time allowance, capped at 24 FPS. First
Generate and a changed prompt still load weights lazily, then prepare a warm lookahead.
Prompt/seed edits immediately invalidate queued and in-flight old results and reset
scene continuation; another Generate pulse is not needed once rendering is armed.
The last frame stays visible during startup, prompt reloading or a buffer underrun.
GPU render, CPU delivery and controls overlap; GPU calls remain serialized. At most
two CPU clips (17–121 frames each) plus ACK-limited delivery credits are queued.
Errors stop automatic retries. Network stalls or unexpectedly slow renders can still
cause gaps; this is slower motion, not 24 newly generated FPS or guaranteed seamless motion.

Continuation PNGs use `crf=0` to skip the runtime's lossy H.264 input re-compression.
VAE feedback and generative drift can still accumulate. **REFRESH_SCENE_EVERY = 4**
returns to Prompt A / the startup image after every four clips, breaking that chain.
It may create a scene cut. Change **Scene Refresh (0=Off)** live in TD, or set
0 to disable periodic refresh. Prompt edits always reset the old scene immediately.

For opt-in GPU validation, stop Step 3 and run the existing isolated Python:
`python check_ltx_gpu.py --output captures/ltx-validation --frames 17`.
It retains first/last JPEGs and a timing/memory report. Add `--image /content/scene.png`
to validate image conditioning. No neural weights or CUDA packages are used by CPU tests.
Keep the Colab launch cell and TouchDesigner running for playback.
'''
    check = dedent('''\
        # Step 1: Check GPU without installing anything
        import torch
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
            raise RuntimeError('Select an A100 40 GB+ runtime (Ampere or newer).')
        free, total = torch.cuda.mem_get_info()
        print(torch.cuda.get_device_name(), f'{free/1024**3:.1f}/{total/1024**3:.1f} GiB free/total')
        if total < 35 * 1024**3 or free < 30 * 1024**3:
            raise RuntimeError('This LTX trial needs 35 GiB total / 30 GiB free VRAM.')
    ''')
    setup = dedent('''\
        # @title Step 2: Install isolated LTX runtime and choose storage
        USE_GOOGLE_DRIVE = True # @param {type:"boolean"}
        DRIVE_MODEL_FOLDER = "StreamDiffusion/ltx25-models" # @param {type:"string"}
        import os, sys, socket
        from pathlib import Path
        try:
            connection=socket.create_connection(('127.0.0.1',8000),timeout=.5)
        except OSError:
            pass
        else:
            connection.close()
            raise RuntimeError('An active bridge uses port 8000; it was left running. Stop its launch cell first.')
        runtime_root=Path('/content/colab-sd-ltx')
        runtime_root.mkdir(exist_ok=True)
    ''')
    setup += 'sources = ' + repr({n: (Path(project_dir) / n).read_text(encoding='utf-8') for n in SOURCES}) + '\n'
    setup += dedent('''\
        for name, content in sources.items():
            (runtime_root/name).write_text(content,encoding='utf-8')
        sys.path.insert(0,str(runtime_root))
        from google.colab import userdata
        from huggingface_hub import get_token
        token=get_token()
        if not token:
            try:
                token=userdata.get('HF_TOKEN')
            except Exception:
                token=None
        if not token:
            raise RuntimeError('Add a Hugging Face read token as Colab Secret HF_TOKEN and enable Notebook access.')
        os.environ['HF_TOKEN']=token
        del token
        import importlib, colab_setup, ltx_setup
        importlib.reload(colab_setup)
        importlib.reload(ltx_setup)
        colab_setup.configure_model_cache(USE_GOOGLE_DRIVE,DRIVE_MODEL_FOLDER)
        os.environ.pop('TRANSFORMERS_CACHE',None)
        os.environ['HF_HUB_DISABLE_SYMLINKS']='1'
        os.environ['HF_HUB_DISABLE_SHARED_BLOBS']='1'
        os.environ['TORCHINDUCTOR_CACHE_DIR']=str(runtime_root/'inductor-cache')
        ltx_python,ltx_tunnel=ltx_setup.setup(runtime_root)
        print('LTX environment ready. Step 3 checks access, downloads/stages missing weights and starts the bridge.')
    ''')
    launch = dedent('''\
        # @title Step 3: Launch LTX-2.5 clip server
        STARTUP_RESOLUTION = "1024x576" # @param ["1024x576", "512x512", "768x512", "1024x768"]
        CLIP_FRAMES = 33 # @param [17, 33, 65, 121] {type:"raw"}
        PERFORMANCE_PROFILE = "resident" # @param ["resident", "low_memory"]
        CONTINUOUS_GENERATION = True # @param {type:"boolean"}
        CONTINUE_SCENE = True # @param {type:"boolean"}
        REFRESH_SCENE_EVERY = 4 # @param [0, 2, 4, 8, 16, 32] {type:"raw"}
        INPUT_IMAGE = "" # @param {type:"string"}
        TUNNEL_PROTOCOL = "http2" # @param ["http2", "auto", "quic"]
        if 'ltx_session' in globals():
            ltx_session.stop()
        import importlib, colab_launcher
        importlib.reload(colab_launcher)
        width,height=map(int,STARTUP_RESOLUTION.split('x'))
        ltx_session=colab_launcher.BridgeSession(root=runtime_root,backend='ltx',
            server_script='colab_server.py',python_executable=ltx_python,cloudflared=ltx_tunnel,
            output_size=(width,height),startup_timeout=3600,max_fps=24,tunnel_protocol=TUNNEL_PROTOCOL,
            server_environment={'LTX_CLIP_FRAMES':str(CLIP_FRAMES),'LTX_INPUT_IMAGE':INPUT_IMAGE,
                'LTX_PERFORMANCE_PROFILE':PERFORMANCE_PROFILE,
                'LTX_CONTINUOUS':'1' if CONTINUOUS_GENERATION else '0',
                'LTX_CONTINUE_SCENE':'1' if CONTINUE_SCENE else '0',
                'LTX_REFRESH_EVERY':str(REFRESH_SCENE_EVERY)})
        try:
            url=ltx_session.start()
            print('Paste this URL into the updated TouchDesigner component:')
            print(url,flush=True)
            print('Use Prompt A / Seed, then LTX page / Generate or Restart. Continuous mode advances seeds automatically.')
            print('Continuous playback buffers two clips, then paces delivery to warm rendering speed while rendering ahead.')
            print('First use or a changed prompt holds the last frame while loading. Periodic scene refresh can cause cuts.')
            ltx_session.wait()
        except KeyboardInterrupt:
            print('Stopping this LTX session...')
        finally:
            ltx_session.stop()
    ''')
    return {'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': (intro + STREAMING_GUIDE).splitlines(keepends=True)},
                      *[code_cell(s) for s in (check, setup, launch)]],
            'metadata': {'accelerator': 'GPU', 'colab': {'gpuType': 'A100', 'provenance': []},
                         'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 0}


def generate_notebook(project_dir=PROJECT_DIR, check=False):
    target = Path(project_dir) / 'LTX25_Colab_Server.ipynb'
    content = json.dumps(make_notebook(project_dir), indent=2) + '\n'
    if check:
        if not target.exists() or target.read_text(encoding='utf-8') != content:
            raise RuntimeError('LTX notebook is stale; run python generate_notebook.py')
    else:
        temporary = target.with_suffix('.ipynb.tmp')
        temporary.write_text(content, encoding='utf-8', newline='\n')
        temporary.replace(target)
    return target
