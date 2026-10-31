"""Generate a self-contained Colab notebook from the checked-in Python sources."""
import argparse
import json
from pathlib import Path
from textwrap import dedent

PROJECT_DIR = Path(__file__).resolve().parent


def code_cell(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


def make_notebook(project_dir=PROJECT_DIR):
    root = Path(project_dir)
    intro = """# StreamDiffusion Remote TouchDesigner Bridge
Run SD-Turbo on a Colab GPU and stream output to TouchDesigner over a Cloudflare tunnel.

1. Select a GPU runtime, then run the four code cells in order.
2. The launcher checks actual server readiness before creating the tunnel.
3. Paste the displayed `wss://.../ws/stream` URL into TouchDesigner and enable Connect.

Supports text/image generation, prompt blending, audio controls, telemetry, snapshots,
and 16 output presets up to 1024 x 768 (or 768 x 1024 portrait). Default: 512 x 512.
Resolution changes reload model buffers; TensorRT engines are cached per width/height.

Setup pins the StreamDiffusion source revision and core compatibility dependencies.
Re-run the launch cell to replace its own previous server and tunnel. Interrupt it to stop both.
Server and tunnel logs are saved in `bridge_logs/` inside the Colab runtime.
[Project source](https://github.com/wenjunii/colab-sd)
"""
    gpu_check = dedent("""\
        # Step 1: Check GPU
        import subprocess
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Select Runtime > Change runtime type > GPU, then run this cell again.')
        subprocess.run(['nvidia-smi'], check=True)
    """)
    setup = '# Step 2: Install the compatible environment\n' + (root / 'colab_setup.py').read_text(encoding='utf-8')
    write_server = '# Step 3: Write server and standalone helpers\n'
    # Keep the first assignment as server_code for simple source verification.
    for variable, source_name, target_name in (
        ('server_code', 'colab_server.py', 'server.py'),
        ('protocol_code', 'bridge_protocol.py', 'bridge_protocol.py'),
        ('launcher_code', 'colab_launcher.py', 'colab_launcher.py'),
    ):
        write_server += variable + ' = ' + json.dumps((root / source_name).read_text(encoding='utf-8')) + '\n'
        write_server += f"with open({target_name!r}, 'w', encoding='utf-8') as handle:\n    handle.write({variable})\n"
    write_server += "print('Bridge server and helpers written successfully.')\n"
    launch = dedent("""\
        # @title Step 4: Launch Server and Cloudflare Tunnel
        # @markdown TensorRT may take several minutes to compile engines for a new output size.
        ENABLE_TENSORRT = False # @param {type:"boolean"}
        MAX_FPS = 60 # @param {type:"integer"}

        from html import escape
        from IPython.display import display, HTML
        if 'bridge_session' in globals():
            bridge_session.stop()
        import importlib
        import colab_launcher
        importlib.reload(colab_launcher)
        bridge_session = colab_launcher.BridgeSession(tensorrt=ENABLE_TENSORRT, max_fps=MAX_FPS)
        try:
            ws_url = bridge_session.start()
            display(HTML(f'<h3>StreamDiffusion is ready</h3><p>Paste this URL into TouchDesigner:</p><pre>{escape(ws_url)}</pre>'))
            bridge_session.wait()
        except KeyboardInterrupt:
            print('Stopping this bridge session...')
        finally:
            bridge_session.stop()
    """)
    cells = [{"cell_type": "markdown", "metadata": {}, "source": intro.splitlines(keepends=True)}]
    cells.extend(code_cell(source) for source in (gpu_check, setup, write_server, launch))
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
            "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}


def generate_notebook(project_dir=PROJECT_DIR, check=False):
    root = Path(project_dir)
    target = root / 'StreamDiffusion_Colab_Server.ipynb'
    content = json.dumps(make_notebook(root), indent=2) + '\n'
    if check:
        if not target.exists() or target.read_text(encoding='utf-8') != content:
            raise RuntimeError('The Colab notebook is stale. Run python generate_notebook.py.')
    else:
        temporary = target.with_suffix('.ipynb.tmp')
        temporary.write_text(content, encoding='utf-8', newline='\n')
        temporary.replace(target)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Fail if the generated notebook differs from its sources')
    arguments = parser.parse_args()
    result = generate_notebook(check=arguments.check)
    print(('Verified ' if arguments.check else 'Generated ') + str(result))
