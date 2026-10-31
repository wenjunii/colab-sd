import os
import sys
import traceback

# 1. Compatibility shims for huggingface_hub
import huggingface_hub
import huggingface_hub.constants

if not hasattr(huggingface_hub.constants, 'hf_cache_home'):
    huggingface_hub.constants.hf_cache_home = getattr(
        huggingface_hub.constants, 'HF_HOME', os.path.expanduser('~/.cache/huggingface')
    )

if not hasattr(huggingface_hub, 'HfFolder'):
    class HfFolder:
        @classmethod
        def get_token(cls):
            try:
                from huggingface_hub import get_token
                return get_token()
            except Exception:
                return None
        @classmethod
        def save_token(cls, token):
            pass
    huggingface_hub.HfFolder = HfFolder

if not hasattr(huggingface_hub, 'cached_download'):
    from huggingface_hub import hf_hub_download
    huggingface_hub.cached_download = hf_hub_download

# 2. Compatibility shims for transformers
import transformers
import transformers.utils

if not hasattr(transformers.utils, 'FLAX_WEIGHTS_NAME'):
    transformers.utils.FLAX_WEIGHTS_NAME = 'flax_model.msgpack'
if not hasattr(transformers.utils, 'FLAX_WEIGHTS_INDEX_NAME'):
    transformers.utils.FLAX_WEIGHTS_INDEX_NAME = 'flax_model.msgpack.index.json'

if not hasattr(transformers, 'EncoderDecoderCache'):
    class EncoderDecoderCache:
        pass
    transformers.EncoderDecoderCache = EncoderDecoderCache
    try:
        import transformers.cache_utils
        transformers.cache_utils.EncoderDecoderCache = EncoderDecoderCache
    except Exception:
        pass

import time
import asyncio
import json
import logging
import cv2
import numpy as np
from PIL import Image
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# Disable autograd globally for fast inference
torch.set_grad_enabled(False)

sys.path.append('.')
try:
    from utils.wrapper import StreamDiffusionWrapper
    from streamdiffusion.image_utils import postprocess_image
except ImportError:
    StreamDiffusionWrapper = None
    postprocess_image = None

# Patch image_utils.pt_to_numpy just in case
try:
    import streamdiffusion.image_utils
    orig_pt_to_numpy = streamdiffusion.image_utils.pt_to_numpy
    def safe_pt_to_numpy(images):
        return images.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    streamdiffusion.image_utils.pt_to_numpy = safe_pt_to_numpy
except Exception:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StreamDiffusionServer")
app = FastAPI()

stream_txt2img = None
stream_img2img = None
pipeline_ready = False

# Engine acceleration state
enable_tensorrt = os.environ.get("ENABLE_TENSORRT", "0") in ("1", "true", "True") or "--tensorrt" in sys.argv
active_engine = "TensorRT (Turbo)" if enable_tensorrt else "PyTorch (FP16)"

# Prompt state
current_prompt_a = "cinematic cyberpunk cityscape, neon lights, 8k, detailed"
current_prompt_b = "bioluminescent alien jungle, glowing flora, ethereal, 8k"
current_prompt_blend = 0.0
current_neg_prompt = "blurry, low quality, distorted, watermark"

current_mode = "txt2img"
current_seed = 42
current_strength = 0.6

def set_seed(s):
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def apply_prompts_and_blend(prompt_a, prompt_b="", blend=0.0, neg_prompt=""):
    global current_prompt_a, current_prompt_b, current_prompt_blend, current_neg_prompt
    if prompt_a is not None and str(prompt_a).strip():
        current_prompt_a = str(prompt_a).strip()
    if prompt_b is not None:
        current_prompt_b = str(prompt_b).strip()
    if blend is not None:
        current_prompt_blend = max(0.0, min(1.0, float(blend)))
    if neg_prompt is not None:
        current_neg_prompt = str(neg_prompt).strip()

    if not pipeline_ready or stream_txt2img is None:
        return

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        core_pipe = stream_txt2img.stream.pipe

        with torch.no_grad():
            embed_a = core_pipe.encode_prompt(
                prompt=current_prompt_a,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=current_neg_prompt,
            )[0]

            if current_prompt_b and current_prompt_blend > 0.001:
                embed_b = core_pipe.encode_prompt(
                    prompt=current_prompt_b,
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    negative_prompt=current_neg_prompt,
                )[0]
                alpha = current_prompt_blend
                blended = (1.0 - alpha) * embed_a + alpha * embed_b
            else:
                blended = embed_a

            # Apply blended embedding tensor to both pipelines
            for s in [stream_txt2img, stream_img2img]:
                if s is not None:
                    target = s.stream if hasattr(s, "stream") else s
                    if hasattr(target, "prompt_embeds"):
                        target.prompt_embeds = blended.repeat(target.batch_size, 1, 1)

        logger.info(f"Prompts updated: A='{current_prompt_a[:25]}...', B='{current_prompt_b[:25]}...', blend={current_prompt_blend:.2f}")
    except Exception as e:
        logger.error(f"Error in prompt blending: {traceback.format_exc()}")

def apply_strength(new_strength):
    global current_strength, stream_img2img
    current_strength = float(new_strength)
    try:
        if stream_img2img is not None:
            if hasattr(stream_img2img, "update_strength"):
                stream_img2img.update_strength(current_strength)
            elif hasattr(stream_img2img, "stream") and hasattr(stream_img2img.stream, "t_list"):
                t0 = max(5, min(44, int(45 - current_strength * 40)))
                stream_img2img.stream.t_list = [t0, 45]
                stream_img2img.prepare(prompt=current_prompt_a, num_inference_steps=50)
                logger.info(f"Updated img2img denoise strength to {current_strength:.2f} (t_index_list={[t0, 45]})")
    except Exception as e:
        logger.error(f"Error updating strength: {e}")

def get_gpu_telemetry():
    vram_str = "--"
    temp_str = "--"
    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            pct = (allocated / total) * 100
            vram_str = f"{allocated:.1f}/{total:.1f}GB ({pct:.0f}%)"
        except Exception:
            pass
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                timeout=1,
                text=True
            ).strip()
            temp_str = f"{out}°C"
        except Exception:
            pass
    return vram_str, temp_str

def run_model_inference(input_image=None):
    """
    Runs StreamDiffusion inference under torch.no_grad() inside the executor thread.
    Never use torch.inference_mode() because StreamDiffusion caches tensors between steps.
    """
    with torch.no_grad():
        if current_mode in ("img2img", "v2v") and input_image is not None and stream_img2img is not None:
            return stream_img2img(input_image)
        if stream_txt2img is not None:
            return stream_txt2img()
        return None

def compress_frame(arr_bgr, max_bytes=48000):
    """
    Dynamically compresses JPEG to guarantee payload stays strictly under TouchDesigner's
    64KB WebSocket limit (typically around 25KB-42KB).
    """
    for q in (60, 50, 40, 30):
        ret, buf = cv2.imencode('.jpg', arr_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ret and len(buf) <= max_bytes:
            return buf.tobytes()
    h, w = arr_bgr.shape[:2]
    small = cv2.resize(arr_bgr, (int(w * 0.8), int(h * 0.8)), interpolation=cv2.INTER_AREA)
    for q in (45, 30, 20):
        ret, buf = cv2.imencode('.jpg', small, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ret and len(buf) <= max_bytes:
            return buf.tobytes()
    return buf.tobytes()

def load_pipeline():
    global stream_txt2img, stream_img2img, pipeline_ready, active_engine
    device = "cuda" if torch.cuda.is_available() else "cpu"
    accel_mode = "tensorrt" if enable_tensorrt else "none"
    print(f">>> Initializing StreamDiffusion (SD-Turbo) on {device.upper()} with acceleration='{accel_mode}' (Engine: {active_engine})...")

    # 1. Text-to-Image pipeline: [0, 16, 32, 45] produces full, coherent, prompt-matching frames
    print(">>> Initializing Text-to-Image pipeline (t_index_list=[0, 16, 32, 45])...")
    try:
        stream_txt2img = StreamDiffusionWrapper(
            model_id_or_path="stabilityai/sd-turbo",
            t_index_list=[0, 16, 32, 45],
            frame_buffer_size=1,
            width=512,
            height=512,
            warmup=5,
            acceleration=accel_mode,
            engine_dir="engines",
            mode="txt2img",
            use_lcm_lora=False,
            use_tiny_vae=True,
            cfg_type="none",
            use_denoising_batch=True,
            device=device,
            seed=current_seed,
        )
    except Exception as e:
        if accel_mode == "tensorrt":
            print(f">>> TensorRT initialization failed ({e}), falling back to PyTorch Native mode...")
            accel_mode = "none"
            active_engine = "PyTorch (FP16)"
            stream_txt2img = StreamDiffusionWrapper(
                model_id_or_path="stabilityai/sd-turbo",
                t_index_list=[0, 16, 32, 45],
                frame_buffer_size=1,
                width=512,
                height=512,
                warmup=5,
                acceleration="none",
                mode="txt2img",
                use_lcm_lora=False,
                use_tiny_vae=True,
                cfg_type="none",
                use_denoising_batch=True,
                device=device,
                seed=current_seed,
            )
        else:
            raise
    set_seed(current_seed)

    # 2. Image-to-Image / StreamV2V pipeline: initial t_index_list based on current_strength
    t0_init = max(5, min(44, int(45 - current_strength * 40)))
    print(f">>> Initializing Image-to-Image / StreamV2V pipeline (t_index_list=[{t0_init}, 45], strength={current_strength:.2f})...")
    try:
        from streamdiffusion import StreamDiffusion
        core_pipe = stream_txt2img.stream.pipe
        stream_core = StreamDiffusion(
            pipe=core_pipe,
            t_index_list=[t0_init, 45],
            torch_dtype=core_pipe.dtype,
            width=512,
            height=512,
            frame_buffer_size=1,
            use_denoising_batch=True,
            cfg_type="none",
        )
        if hasattr(stream_txt2img.stream, "vae") and stream_txt2img.stream.vae:
            stream_core.vae = stream_txt2img.stream.vae

        class SharedImg2ImgWrapper:
            def __init__(self, core):
                self.stream = core
            def prepare(self, prompt, **kwargs):
                self.stream.prepare(prompt, **kwargs)
            def update_prompt(self, prompt):
                self.stream.update_prompt(prompt)
            def update_strength(self, strength: float):
                # Dynamically maps strength [0.1..1.0] to start timestep index [44..5]
                t0 = max(5, min(44, int(45 - strength * 40)))
                new_t_list = [t0, 45]
                if getattr(self.stream, "t_list", None) != new_t_list:
                    self.stream.t_list = new_t_list
                    self.stream.prepare(prompt=current_prompt_a, num_inference_steps=50)
                    logger.info(f"Updated img2img denoise strength to {strength:.2f} (t_index_list={new_t_list})")
            def __call__(self, image):
                x_output = self.stream(image)
                pil_out = postprocess_image(x_output, output_type="pil")
                return pil_out[0] if isinstance(pil_out, list) else pil_out

        stream_img2img = SharedImg2ImgWrapper(stream_core)
    except Exception as e:
        print(">>> Fallback to standalone img2img wrapper:", e)
        stream_img2img = StreamDiffusionWrapper(
            model_id_or_path="stabilityai/sd-turbo",
            t_index_list=[t0_init, 45],
            frame_buffer_size=1,
            width=512,
            height=512,
            warmup=5,
            acceleration="none",
            mode="img2img",
            use_lcm_lora=False,
            use_tiny_vae=True,
            cfg_type="none",
            use_denoising_batch=True,
            device=device,
            seed=current_seed,
        )

    print(">>> Preparing both pipelines with initial prompt...")
    stream_txt2img.prepare(prompt=current_prompt_a, num_inference_steps=50)
    stream_img2img.prepare(prompt=current_prompt_a, num_inference_steps=50)

    print(">>> Warming up models...")
    with torch.no_grad():
        for _ in range(4):
            stream_txt2img()
        dummy_img = Image.new("RGB", (512, 512), color=(128, 128, 128))
        for _ in range(3):
            stream_img2img(dummy_img)

    pipeline_ready = True
    print(">>> StreamDiffusion pipeline is READY! <<<", flush=True)

@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_pipeline)

@app.get("/")
def index():
    return HTMLResponse("""
    <html>
        <head><title>StreamDiffusion TouchDesigner Server</title></head>
        <body style="font-family: sans-serif; padding: 40px; background: #121212; color: #fff;">
            <h2>StreamDiffusion (SD-Turbo) Colab Server</h2>
            <p>Status: <span style="color: #0f0;">Running</span></p>
            <p>Connect your TouchDesigner client via the WebSocket endpoint: <code>/ws/stream</code></p>
        </body>
    </html>
    """)

@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    global current_prompt_a, current_prompt_b, current_prompt_blend, current_neg_prompt
    global current_mode, current_seed, current_strength
    await websocket.accept()
    logger.info("TouchDesigner client connected to WebSocket.")

    vram_s, temp_s = get_gpu_telemetry()
    await websocket.send_text(json.dumps({
        "type": "welcome",
        "message": "StreamDiffusion server connected",
        "engine": active_engine,
        "prompt_a": current_prompt_a,
        "prompt_b": current_prompt_b,
        "blend": current_prompt_blend,
        "mode": current_mode,
        "seed": current_seed,
        "strength": current_strength,
        "vram": vram_s,
        "temp": temp_s,
    }))

    CHUNK_SIZE = 2048
    frame_counter = 0

    async def send_chunked_frame(ws: WebSocket, jpg_bytes: bytes):
        nonlocal frame_counter
        frame_counter = (frame_counter + 1) % 256
        total_chunks = (len(jpg_bytes) + CHUNK_SIZE - 1) // CHUNK_SIZE
        for chunk_idx in range(total_chunks):
            start = chunk_idx * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, len(jpg_bytes))
            chunk = jpg_bytes[start:end]
            header = bytes([frame_counter, chunk_idx, total_chunks])
            try:
                await ws.send_bytes(header + chunk)
            except Exception:
                break

    txt2img_task = None
    stop_event = asyncio.Event()
    img2img_counter = 0
    td_chunk_buffers = {}

    async def txt2img_loop():
        target_fps = 25
        frame_delay = 1.0 / target_fps
        frame_count = 0
        logger.info(f"txt2img_loop started! (pipeline_ready={pipeline_ready}, current_mode={current_mode})")
        while not stop_event.is_set():
            start_time = time.time()
            if pipeline_ready and current_mode == "txt2img":
                try:
                    loop = asyncio.get_event_loop()
                    t0 = time.time()
                    output_tensor = await loop.run_in_executor(None, run_model_inference)
                    inf_ms = (time.time() - t0) * 1000

                    if output_tensor is not None:
                        if isinstance(output_tensor, Image.Image):
                            arr_bgr = cv2.cvtColor(np.array(output_tensor), cv2.COLOR_RGB2BGR)
                        elif hasattr(output_tensor, "cpu"):
                            arr = (output_tensor.detach().squeeze().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                            arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                        else:
                            arr_bgr = np.array(output_tensor)

                        jpg_bytes = compress_frame(arr_bgr)
                        await send_chunked_frame(websocket, jpg_bytes)

                        frame_count += 1
                        if frame_count % 10 == 0:
                            try:
                                v_str, t_str = get_gpu_telemetry()
                                await websocket.send_text(json.dumps({
                                    "type": "stats",
                                    "engine": active_engine,
                                    "inf_ms": round(inf_ms, 1),
                                    "vram": v_str,
                                    "temp": t_str,
                                    "mode": current_mode,
                                    "strength": current_strength
                                }))
                            except Exception:
                                pass

                        if frame_count % 25 == 1:
                            logger.info(f"StreamDiffusion active: frame #{frame_count} ({inf_ms:.1f}ms, {len(jpg_bytes)} bytes)")
                except Exception as e:
                    if not stop_event.is_set():
                        logger.error(f"Inference error in txt2img: {traceback.format_exc()}")

            elapsed = time.time() - start_time
            delay = max(0.005, frame_delay - elapsed)
            await asyncio.sleep(delay)

    txt2img_task = asyncio.create_task(txt2img_loop())

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                logger.info("TouchDesigner client disconnected normally.")
                break

            if "text" in message:
                try:
                    data = json.loads(message["text"])
                    msg_type = data.get("type", "")

                    if msg_type == "control" or "prompt" in data or "mode" in data or "seed" in data or "strength" in data or "blend" in data:
                        p_a = data.get("prompt_a") or data.get("prompt")
                        p_b = data.get("prompt_b", "")
                        blend_val = data.get("blend", 0.0)
                        neg = data.get("neg_prompt", "")

                        new_mode = data.get("mode")
                        new_seed = data.get("seed")
                        new_strength = data.get("strength")

                        apply_prompts_and_blend(p_a, p_b, blend_val, neg)

                        if new_mode is not None:
                            mode_str = str(new_mode).lower()
                            if "txt" in mode_str or "text" in mode_str:
                                current_mode = "txt2img"
                            elif "img" in mode_str or "webcam" in mode_str or "v2v" in mode_str or "noise" in mode_str:
                                current_mode = "img2img"
                            else:
                                current_mode = "img2img"
                            logger.info(f"Switching mode to: {current_mode}")

                        if new_seed is not None and new_seed != current_seed:
                            current_seed = new_seed
                            set_seed(current_seed)

                        if new_strength is not None and abs(float(new_strength) - current_strength) > 0.01:
                            apply_strength(float(new_strength))

                    elif msg_type == "ping":
                        await websocket.send_text(json.dumps({"type": "pong", "time": time.time()}))

                except Exception as e:
                    logger.error(f"Error handling text message: {e}")

            elif "bytes" in message:
                if pipeline_ready and current_mode in ("img2img", "v2v"):
                    raw = message["bytes"]
                    try:
                        frame_bytes = None
                        if raw.startswith(b"FRM:") and len(raw) >= 7:
                            frame_id = raw[4]
                            chunk_idx = raw[5]
                            total_chunks = raw[6]
                            chunk_data = raw[7:]

                            if frame_id not in td_chunk_buffers:
                                td_chunk_buffers.clear()
                                td_chunk_buffers[frame_id] = [None] * total_chunks

                            if chunk_idx < total_chunks:
                                td_chunk_buffers[frame_id][chunk_idx] = chunk_data

                            if all(c is not None for c in td_chunk_buffers[frame_id]):
                                frame_bytes = b"".join(td_chunk_buffers[frame_id])
                                del td_chunk_buffers[frame_id]
                        elif raw.startswith(b"\xff\xd8"):
                            frame_bytes = raw
                        else:
                            frame_bytes = raw

                        if frame_bytes is not None:
                            nparr = np.frombuffer(frame_bytes, np.uint8)
                            in_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if in_bgr is not None:
                                in_rgb = cv2.cvtColor(in_bgr, cv2.COLOR_BGR2RGB)
                                in_pil = Image.fromarray(in_rgb).resize((512, 512))

                                loop = asyncio.get_event_loop()
                                t0 = time.time()
                                output = await loop.run_in_executor(None, run_model_inference, in_pil)
                                inf_ms = (time.time() - t0) * 1000

                                if isinstance(output, Image.Image):
                                    out_bgr = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)
                                elif hasattr(output, "cpu"):
                                    out_arr = (output.detach().squeeze().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                                    out_bgr = cv2.cvtColor(out_arr, cv2.COLOR_RGB2BGR)
                                else:
                                    out_bgr = np.array(output)

                                out_bytes = compress_frame(out_bgr)
                                await send_chunked_frame(websocket, out_bytes)

                                img2img_counter += 1
                                if img2img_counter % 10 == 0:
                                    try:
                                        v_str, t_str = get_gpu_telemetry()
                                        await websocket.send_text(json.dumps({
                                            "type": "stats",
                                            "engine": active_engine,
                                            "inf_ms": round(inf_ms, 1),
                                            "vram": v_str,
                                            "temp": t_str,
                                            "mode": current_mode,
                                            "strength": current_strength
                                        }))
                                    except Exception:
                                        pass

                                if img2img_counter % 25 == 1:
                                    logger.info(f"StreamDiffusion img2img: frame #{img2img_counter} ({inf_ms:.1f}ms, {len(out_bytes)} bytes)")
                    except Exception as e:
                        if not stop_event.is_set():
                            logger.error(f"Error in img2img processing: {traceback.format_exc()}")

    except WebSocketDisconnect:
        logger.info("TouchDesigner client disconnected gracefully.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        stop_event.set()
        if txt2img_task:
            txt2img_task.cancel()
            try:
                await txt2img_task
            except asyncio.CancelledError:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
