import json

with open("colab_server.py", "r", encoding="utf-8") as f:
    server_py_content = f.read()

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# ⚡ StreamDiffusion Colab Server for TouchDesigner\n",
            "This notebook runs **StreamDiffusion (SD-Turbo)** accelerated on Google Colab Pro GPU and bridges real-time streaming directly to your local **TouchDesigner** via a secure Cloudflare Tunnel.\n",
            "\n",
            "### 🌟 Features Supported:\n",
            "- **Dual-Prompt Latent Morphing**: Blend smoothly between Prompt A and Prompt B using real-time GPU latent interpolation.\n",
            "- **Dynamic Denoise Strength**: Interactive 0.0 - 1.0 slider controlling diffusion timestep limits.\n",
            "- **Audio-Reactivity**: CHOP connector and beat detection for music-synchronized visuals and random seed triggers.\n",
            "- **Side-by-Side & Split Wipe**: Live comparison views between webcam/noise input and AI output.\n",
            "- **High-Res Snapshots**: Save PNG frames with complete JSON parameter and telemetry sidecars.\n",
            "- **Remote GPU Telemetry**: Live VRAM and GPU temperature reporting sent back to TouchDesigner.\n",
            "\n",
            "### Instructions:\n",
            "1. Make sure your runtime has a GPU: Go to **Runtime -> Change runtime type -> Hardware accelerator: GPU** (Select **A100**, **L4**, or **T4**).\n",
            "2. Click **Runtime -> Run all** (or run each cell sequentially).\n",
            "3. Cell 4 will warm up the pipeline and print your **TouchDesigner WebSocket URL** (`wss://....trycloudflare.com/ws/stream`)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 1: Check GPU\n",
            "!nvidia-smi"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 2: Clone StreamDiffusion & Install Compatible Dependencies\n",
            "import os\n",
            "print('Setting up StreamDiffusion environment...')\n",
            "\n",
            "# Clone repo if not already present\n",
            "if not os.path.exists('StreamDiffusion'):\n",
            "    !git clone https://github.com/cumulo-autumn/StreamDiffusion.git\n",
            "\n",
            "%cd StreamDiffusion\n",
            "!pip install -q -e .\n",
            "# Pin compatible versions and remove conflicting peft\n",
            "!pip install -q 'transformers<4.40.0' 'huggingface-hub<0.26.0'\n",
            "!pip uninstall -y peft || true\n",
            "!pip install -q diffusers accelerate fastapi uvicorn websockets python-multipart opencv-python-headless pillow\n",
            "\n",
            "# Download cloudflared for tunnel\n",
            "!wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared\n",
            "!chmod +x cloudflared\n",
            "\n",
            "print('✅ Setup complete!')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 3: Write server.py\n",
            "server_code = " + json.dumps(server_py_content) + "\n\n"
            "with open('server.py', 'w') as f:\n"
            "    f.write(server_code.strip())\n"
            "print('✅ server.py written successfully!')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 4: Launch Server, Warm Up & Start Cloudflare Tunnel\n",
            "import os\n",
            "import subprocess\n",
            "import time\n",
            "import sys\n",
            "import re\n",
            "from IPython.display import display, HTML\n",
            "\n",
            "# Kill any lingering process on port 8000\n",
            "os.system('fuser -k 8000/tcp 2>/dev/null || true')\n",
            "\n",
            "print('Starting Python server and downloading SD-Turbo weights (~2.5GB)...')\n",
            "server_proc = subprocess.Popen(\n",
            "    [sys.executable, 'server.py'],\n",
            "    stdout=subprocess.PIPE,\n",
            "    stderr=subprocess.STDOUT,\n",
            "    text=True,\n",
            "    bufsize=1\n",
            ")\n",
            "\n",
            "# Stream logs until model is fully ready\n",
            "ready = False\n",
            "for line in server_proc.stdout:\n",
            "    print(line, end='', flush=True)\n",
            "    if 'READY!' in line or 'Application startup complete' in line or 'Uvicorn running' in line:\n",
            "        ready = True\n",
            "        break\n",
            "    if server_proc.poll() is not None:\n",
            "        break\n",
            "\n",
            "if ready:\n",
            "    print('\\nStarting Cloudflare Tunnel to expose port 8000...')\n",
            "    tunnel_proc = subprocess.Popen(\n",
            "        ['./cloudflared', 'tunnel', '--url', 'http://127.0.0.1:8000'],\n",
            "        stdout=subprocess.PIPE,\n",
            "        stderr=subprocess.STDOUT,\n",
            "        text=True,\n",
            "        bufsize=1\n",
            "    )\n",
            "\n",
            "    tunnel_url = None\n",
            "    for line in tunnel_proc.stdout:\n",
            "        match = re.search(r'https://[a-zA-Z0-9-]+\\.trycloudflare\\.com', line)\n",
            "        if match:\n",
            "            tunnel_url = match.group(0)\n",
            "            break\n",
            "\n",
            "    if tunnel_url:\n",
            "        ws_url = tunnel_url.replace('https://', 'wss://') + '/ws/stream'\n",
            "        display(HTML(f'''\n",
            "        <div style=\"background:#1a2e1a; border:2px solid #00ff88; padding:20px; border-radius:10px; margin:20px 0;\">\n",
            "            <h2 style=\"color:#00ff88; margin:0 0 10px 0;\">🚀 StreamDiffusion Server is READY & STREAMING!</h2>\n",
            "            <p style=\"color:#eee; font-size:16px; margin:0 0 10px 0;\">Copy this URL and paste it into TouchDesigner:</p>\n",
            "            <div style=\"background:#000; padding:12px; border-radius:6px; font-family:monospace; font-size:18px; color:#58a6ff;\">\n",
            "                {ws_url}\n",
            "            </div>\n",
            "        </div>\n",
            "        '''))\n",
            "else:\n",
            "    print('\\n❌ Server failed to start. Review error logs above.')\n",
            "\n",
            "# Keep server running\n",
            "try:\n",
            "    while True:\n",
            "        line = server_proc.stdout.readline()\n",
            "        if line:\n",
            "            print(line, end='', flush=True)\n",
            "        time.sleep(0.01)\n",
            "except KeyboardInterrupt:\n",
            "    print('Stopping server...')\n",
            "    server_proc.terminate()\n",
            "    if 'tunnel_proc' in locals():\n",
            "        tunnel_proc.terminate()\n"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "T4",
            "provenance": []
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

with open(r"c:\Users\wenju\.gemini\antigravity\scratch\colab-sd\StreamDiffusion_Colab_Server.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Updated StreamDiffusion_Colab_Server.ipynb successfully!")
