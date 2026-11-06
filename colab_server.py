import os
import sys
import traceback
import gc
import hmac

backend_name = os.environ.get('BRIDGE_BACKEND', 'sd-turbo')
is_world_backend = backend_name == 'waypoint'
is_krea_backend = backend_name == 'krea'
diffusion_model = os.environ.get('BRIDGE_DIFFUSION_MODEL', 'sd-turbo')
if diffusion_model not in ('sd-turbo', 'sdxl-turbo'):
    raise ValueError('Diffusion model must be sd-turbo or sdxl-turbo')


def model_label():
    if is_world_backend:
        return 'Waypoint 1.5-1B'
    if is_krea_backend:
        return 'Krea Realtime 14B'
    return 'SDXL-Turbo' if diffusion_model == 'sdxl-turbo' else 'SD-Turbo'

if not (is_world_backend or is_krea_backend):
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
import threading
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from PIL import Image
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from tensorrt_backend import compile_unet as compile_tensorrt_unet
from bridge_protocol import (OUTPUT_RESOLUTIONS, PROTOCOL_VERSION, OUTPUT_CHUNK_BYTES,
                             MAX_INPUT_FRAME_BYTES, VIDEO_QUALITY_LIMITS, FrameAssembler,
                             resolution, validate_controls, video_byte_budget)

# Disable autograd globally for fast inference
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

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
access_token = os.environ.get('BRIDGE_ACCESS_TOKEN', '')

stream_txt2img = None
stream_img2img = None
pipeline_ready = False
# The two modes share model weights and mutable state. Keep all GPU work on
# one worker, including control updates, without blocking WebSocket handling.
model_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="diffusion")
try:
    max_stream_fps = float(os.environ.get("MAX_STREAM_FPS", "60"))
    if not np.isfinite(max_stream_fps) or not 1 <= max_stream_fps <= 120:
        raise ValueError()
except ValueError:
    logger.warning("Invalid MAX_STREAM_FPS; using 60")
    max_stream_fps = 60.0
prompt_embedding_cache = {}
applied_prompt_state = None
telemetry_lock = threading.Lock()
telemetry_cache = (0.0, "--", "--")
active_client = None
world_backend = None
krea_backend = None
pipeline_revision = 0
last_pipeline_error = None
jpeg_quality_hints = {}

# Engine acceleration state
enable_tensorrt = os.environ.get("ENABLE_TENSORRT", "0") in ("1", "true", "True") or "--tensorrt" in sys.argv
active_engine = "PyTorch (FP16)"
img2img_engine = "PyTorch (FP16)"


def engine_for_mode(mode=None):
    if is_krea_backend:
        if krea_backend is not None:
            return krea_backend.engine
        from krea_quant import int8_engine_label
        return int8_engine_label(os.environ.get('KREA_INT8_PROFILE', 'fast'))
    if is_world_backend:
        return 'Waypoint (INT8)' if os.environ.get('WAYPOINT_QUANT', 'intw8a8') == 'intw8a8' else 'Waypoint (BF16)'
    if enable_tensorrt and not pipeline_ready:
        return 'TensorRT (Unavailable)' if last_pipeline_error else 'TensorRT (Preparing)'
    return img2img_engine if (mode or current_mode) == "img2img" else active_engine

# Prompt state
current_prompt_a = "cinematic cyberpunk cityscape, neon lights, 8k, detailed"
current_prompt_b = "bioluminescent alien jungle, glowing flora, ethereal, 8k"
current_prompt_blend = 0.0
current_neg_prompt = "blurry, low quality, distorted, watermark"

current_mode = "txt2img"
current_seed = 42
current_strength = 0.6
output_width, output_height = ((832, 480) if is_krea_backend else (1280, 720) if is_world_backend else
    resolution({'width': int(os.environ.get('BRIDGE_INITIAL_WIDTH', '512')),
                'height': int(os.environ.get('BRIDGE_INITIAL_HEIGHT', '512'))}))
output_resolutions = OUTPUT_RESOLUTIONS
if is_world_backend:
    current_mode = 'world'
elif is_krea_backend:
    current_mode = 'krea'
allow_tensorrt_resize = os.environ.get('BRIDGE_TRT_ALLOW_RESIZE', '0') == '1'


def resolution_locked():
    return is_world_backend or is_krea_backend or (enable_tensorrt and not allow_tensorrt_resize)


def requested_resolution(data):
    if is_krea_backend:
        return (832, 480)
    if is_world_backend:
        return (1280, 720)
    return resolution(data, (output_width, output_height))


def validated_controls(data, current):
    if is_krea_backend:
        from krea_backend import validate_krea_controls
        return validate_krea_controls(data)
    if is_world_backend:
        from waypoint_backend import validate_world_controls
        return validate_world_controls(data, current)
    return validate_controls(data, current)


def world_progress():
    if is_krea_backend and krea_backend is not None:
        return krea_backend.progress()
    return world_backend.progress() if is_world_backend and world_backend is not None else {}

def set_seed(s):
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def apply_prompts_and_blend(prompt_a=None, prompt_b=None, blend=None, neg_prompt=None):
    global current_prompt_a, current_prompt_b, current_prompt_blend, current_neg_prompt
    global applied_prompt_state
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

    state = (current_prompt_a, current_prompt_b, current_prompt_blend, current_neg_prompt)
    if state == applied_prompt_state:
        return

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        core_pipe = stream_txt2img.stream.pipe

        def encode_cached(prompt):
            key = (id(core_pipe), prompt, current_neg_prompt)
            embedding = prompt_embedding_cache.pop(key, None)
            if embedding is None:
                encoded = core_pipe.encode_prompt(
                    prompt=prompt,
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    negative_prompt=current_neg_prompt,
                )
                embedding = (encoded[0], encoded[2]) if diffusion_model == 'sdxl-turbo' else encoded[0]
                # Bound retained GPU memory while editing prompts interactively.
                if len(prompt_embedding_cache) >= 4:
                    del prompt_embedding_cache[next(iter(prompt_embedding_cache))]
            prompt_embedding_cache[key] = embedding
            return embedding

        with torch.no_grad():
            embed_a = encode_cached(current_prompt_a)

            if current_prompt_b and current_prompt_blend > 0.001:
                embed_b = encode_cached(current_prompt_b)
                alpha = current_prompt_blend
                blended = (tuple((1.0 - alpha) * a + alpha * b for a, b in zip(embed_a, embed_b))
                           if diffusion_model == 'sdxl-turbo' else (1.0 - alpha) * embed_a + alpha * embed_b)
            else:
                blended = embed_a

            # Apply blended embedding tensor to both pipelines
            for s in [stream_txt2img, stream_img2img]:
                if s is not None:
                    target = s.stream if hasattr(s, "stream") else s
                    if hasattr(target, "prompt_embeds"):
                        tokens = blended[0] if diffusion_model == 'sdxl-turbo' else blended
                        target.prompt_embeds = tokens.repeat(target.batch_size, 1, 1)
                        if diffusion_model == 'sdxl-turbo':
                            target.unet.set_conditioning(blended[1])

        applied_prompt_state = state
        logger.debug("Prompt embeddings updated (blend=%.2f)", current_prompt_blend)
    except Exception as e:
        logger.error(f"Error in prompt blending: {traceback.format_exc()}")
        raise

def apply_strength(new_strength):
    global current_strength, stream_img2img, applied_prompt_state
    current_strength = max(0.1, min(1.0, float(new_strength)))
    try:
        if stream_img2img is not None:
            if hasattr(stream_img2img, "update_strength"):
                stream_img2img.update_strength(current_strength)
            elif hasattr(stream_img2img, "stream") and hasattr(stream_img2img.stream, "t_list"):
                t0 = max(5, min(44, int(45 - current_strength * 40)))
                stream_img2img.stream.t_list = [t0, 45]
                stream_img2img.prepare(prompt=current_prompt_a, num_inference_steps=50)
                logger.info(f"Updated img2img denoise strength to {current_strength:.2f} (t_index_list={[t0, 45]})")
            # prepare() replaces prompt_embeds; restore the cached blend afterward.
            applied_prompt_state = None
    except Exception as e:
        logger.error(f"Error updating strength: {e}")
        raise

def get_gpu_telemetry():
    global telemetry_cache
    # nvidia-smi can take tens of milliseconds. Sample at most once every 2s,
    # and call this function off the asyncio event loop.
    with telemetry_lock:
        now = time.perf_counter()
        if now - telemetry_cache[0] < 2.0:
            return telemetry_cache[1:]
        vram_str, temp_str = sample_gpu_telemetry()
        telemetry_cache = (time.perf_counter(), vram_str, temp_str)
        return vram_str, temp_str


def sample_gpu_telemetry():
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

def run_model_inference(input_image=None, mode=None):
    """
    Runs StreamDiffusion inference under torch.no_grad() inside the executor thread.
    Never use torch.inference_mode() because StreamDiffusion caches tensors between steps.
    """
    with torch.no_grad():
        mode = mode or current_mode
        if mode in ("img2img", "v2v") and input_image is not None and stream_img2img is not None:
            return stream_img2img(input_image)
        if stream_txt2img is not None:
            return stream_txt2img()
        return None


def process_frame(mode, frame_bytes=None, max_frame_bytes=None):
    """Decode, infer, and encode on the worker; return complete frame timings."""
    started = time.perf_counter()
    if is_world_backend or is_krea_backend:
        if is_world_backend and frame_bytes is not None:
            world_backend.set_seed_image(frame_bytes)
        result = (krea_backend if is_krea_backend else world_backend).next_frame()
        if result is None:
            return None
        out_bgr, batch_ms = result
        jpg = compress_frame(out_bgr, max_frame_bytes, allow_oversize=True)
        # The GPU readout is the generated chunk duration, not single-frame latency.
        return jpg, batch_ms, (time.perf_counter() - started) * 1000
    input_image = None
    if frame_bytes is not None:
        in_bgr = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        if in_bgr is None:
            return None
        input_image = Image.fromarray(cv2.cvtColor(in_bgr, cv2.COLOR_BGR2RGB)).resize((output_width, output_height))

    inference_started = time.perf_counter()
    output = run_model_inference(input_image, mode=mode)
    inf_ms = (time.perf_counter() - inference_started) * 1000
    if output is None:
        return None
    if isinstance(output, Image.Image):
        out_bgr = cv2.cvtColor(np.asarray(output), cv2.COLOR_RGB2BGR)
    elif torch.is_tensor(output):
        arr = output.detach()
        if arr.ndim == 4:
            arr = arr[0]
        arr = arr.cpu().permute(1, 2, 0).float().numpy()
        out_bgr = cv2.cvtColor((np.clip(arr, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    else:
        out_bgr = np.asarray(output)
    jpg_bytes = compress_frame(out_bgr, max_bytes=max_frame_bytes,
                               allow_oversize=max_frame_bytes is not None)
    return jpg_bytes, inf_ms, (time.perf_counter() - started) * 1000


def apply_controls(data):
    """Apply an entire update between inference calls on the same worker."""
    global current_mode, current_seed, output_width, output_height, last_pipeline_error, pipeline_ready
    global pipeline_revision, current_prompt_a
    if is_krea_backend:
        data = validated_controls(data, (output_width, output_height))
        previous_seed = krea_backend.seed
        krea_backend.configure(data)
        current_seed, current_prompt_a = krea_backend.seed, krea_backend.prompt
        if data.get('krea_reset') or previous_seed != current_seed:
            pipeline_revision += 1
        return
    if is_world_backend:
        data = validated_controls(data, (output_width, output_height))
        world_backend.configure(data)
        current_seed = world_backend.seed
        if data.get('world_reset'):
            pipeline_revision += 1
        return
    data = validate_controls(data, (output_width, output_height))
    width, height = requested_resolution(data)
    if resolution_locked() and (width, height) != (output_width, output_height):
        raise ValueError('TensorRT resolution is locked; select the startup resolution and restart the server')
    state_fields = ('current_mode', 'current_seed', 'current_strength', 'current_prompt_a',
                    'current_prompt_b', 'current_prompt_blend', 'current_neg_prompt',
                    'output_width', 'output_height')
    previous = {name: globals()[name] for name in state_fields}
    resizing = (width, height) != (output_width, output_height) or not pipeline_ready
    try:
        _apply_validated_controls(data, width, height, resizing)
        last_pipeline_error = None
    except Exception as exc:
        globals().update(previous)
        last_pipeline_error = "Configuration failed; see server logs"
        # A failed constructor can retain tensors in its traceback. Release
        # those frames before attempting to allocate the old model buffers.
        exc.__traceback__ = None
        try:
            if resizing:
                load_pipeline()
            else:
                apply_strength(current_strength)
                set_seed(current_seed)
                apply_prompts_and_blend()
        except Exception:
            pipeline_ready = False
            logger.exception("Could not restore the previous pipeline configuration")
        raise RuntimeError(f"Configuration failed: {exc}") from exc


def _apply_validated_controls(data, width, height, resizing):
    global current_mode, current_seed, output_width, output_height
    global current_prompt_a, current_prompt_b, current_prompt_blend, current_neg_prompt, current_strength
    if resizing:
        # Configure state before allocating fresh buffers; avoid preparing the
        # old models only to discard them immediately afterward.
        for key, name in (('prompt_a', 'current_prompt_a'), ('prompt_b', 'current_prompt_b'),
                          ('blend', 'current_prompt_blend'), ('neg_prompt', 'current_neg_prompt'),
                          ('strength', 'current_strength'), ('seed', 'current_seed'), ('mode', 'current_mode')):
            if key in data:
                globals()[name] = data[key]
        output_width, output_height = width, height
        load_pipeline()
        return
    mode = data.get("mode")
    if mode is not None:
        new_mode = mode
        if new_mode != current_mode:
            current_mode = new_mode
            logger.info("Switching mode to: %s", current_mode)
    seed = data.get("seed")
    if seed is not None and int(seed) != current_seed:
        current_seed = int(seed)
        set_seed(current_seed)
    strength = data.get("strength")
    if strength is not None and abs(float(strength) - current_strength) > 0.01:
        apply_strength(strength)
    apply_prompts_and_blend(
        data.get("prompt_a", data.get("prompt")), data.get("prompt_b"),
        data.get("blend"), data.get("neg_prompt"),
    )

def compress_frame(arr_bgr, max_bytes=None, allow_oversize=False):
    """
    Bound payload size without silently changing the selected output resolution.
    """
    if max_bytes is None:
        # Give larger outputs a proportional byte budget, retaining the tested
        # 48KB default for <=512px frames. TLS packets are still only 2KB each.
        height, width = arr_bgr.shape[:2]
        max_bytes = max(48000, round(48000 * width * height / (512 * 512)))
    qualities = (60, 50, 40, 30, 20, 10, 5, 1)
    key = (*arr_bgr.shape[:2], max_bytes)
    hint, count = jpeg_quality_hints.pop(key, (0, 0))
    # Start near the previous successful quality instead of encoding a large
    # image up to eight times on every frame. Periodically probe one step up.
    start = max(0, hint - 1) if count % 30 == 0 else hint
    smallest = None
    for index in range(start, len(qualities)):
        ret, buf = cv2.imencode('.jpg', arr_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), qualities[index],
                                               int(cv2.IMWRITE_JPEG_OPTIMIZE), 1])
        if ret and (smallest is None or len(buf) < len(smallest)):
            smallest = buf
        if ret and len(buf) <= max_bytes:
            if len(jpeg_quality_hints) >= 32:
                del jpeg_quality_hints[next(iter(jpeg_quality_hints))]
            next_hint = max(0, index - 1) if len(buf) < max_bytes / 2 else index
            jpeg_quality_hints[key] = (next_hint, count + 1)
            return buf.tobytes()
    # Some full-resolution noise/pattern images exceed a small target even at
    # minimum quality. Keep streaming their pixels within the protocol limit.
    if allow_oversize and smallest is not None and len(smallest) <= 255 * OUTPUT_CHUNK_BYTES:
        if len(jpeg_quality_hints) >= 32:
            del jpeg_quality_hints[next(iter(jpeg_quality_hints))]
        jpeg_quality_hints[key] = (len(qualities) - 1, count + 1)
        if count % 30 == 0:
            logger.warning('JPEG at minimum quality exceeds the %d-byte target (%d bytes); '
                           'preserving full output resolution', max_bytes, len(smallest))
        return smallest.tobytes()
    raise ValueError("Could not encode an output JPEG within the frame byte budget")


def create_txt2img_pipeline():
    if diffusion_model == 'sdxl-turbo':
        from sdxl_stream import SDXLStreamWrapper
        return SDXLStreamWrapper.load(output_width, output_height)
    # StreamDiffusion's bundled TensorRT/Polygraphy path targets old CUDA 11/12
    # runtimes. Load the native models, then compile their UNets explicitly.
    return StreamDiffusionWrapper(
        model_id_or_path="stabilityai/sd-turbo",
        t_index_list=[0], frame_buffer_size=1,
        width=output_width, height=output_height, warmup=5,
        acceleration="none", mode="txt2img",
        use_lcm_lora=False, use_tiny_vae=True, cfg_type="none",
        use_denoising_batch=True,
        device="cuda" if torch.cuda.is_available() else "cpu", seed=current_seed,
    )


def load_pipeline():
    global stream_txt2img, stream_img2img, pipeline_ready, active_engine, applied_prompt_state
    global pipeline_revision, img2img_engine, world_backend, krea_backend
    if is_krea_backend:
        if enable_tensorrt:
            raise RuntimeError('Krea uses its own INT8 runtime; disable TensorRT')
        from pathlib import Path
        from krea_backend import KreaBackend
        pipeline_ready = False
        krea_backend = KreaBackend.load(Path(__file__).resolve().parent,
                                        clip_blocks=int(os.environ.get('KREA_CLIP_BLOCKS', '64')))
        pipeline_revision += 1
        pipeline_ready = True
        print('>>> Krea INT8 loaded. Submit Prompt A in TouchDesigner; first chunk may compile kernels.', flush=True)
        return
    if is_world_backend:
        if enable_tensorrt:
            raise RuntimeError('Waypoint uses its own INT8/BF16 runtime; disable TensorRT')
        from waypoint_backend import WaypointBackend
        pipeline_ready = False
        world_backend = WaypointBackend.load(os.environ.get('WAYPOINT_QUANT', 'intw8a8'),
                                            os.environ.get('WAYPOINT_COMPILE_PROFILE', 'fast'),
                                            prompt_seeds=os.environ.get('WAYPOINT_PROMPT_SEEDS', '0') == '1',
                                            prompt_seed_model=os.environ.get('WAYPOINT_PROMPT_SEED_MODEL', 'sd-turbo'))
        pipeline_revision += 1
        pipeline_ready = True
        print('>>> Waypoint loaded. Seed an image from TouchDesigner; first generation compiles kernels.', flush=True)
        return
    if StreamDiffusionWrapper is None and diffusion_model != 'sdxl-turbo':
        raise RuntimeError("StreamDiffusion is not installed. Run Colab setup or workstation.py setup first.")
    pipeline_revision += 1
    prompt_embedding_cache.clear()
    applied_prompt_state = None
    pipeline_ready = False
    # Release old model buffers before allocating a different resolution.
    stream_img2img = None
    stream_txt2img = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    active_engine = "PyTorch (FP16)"
    img2img_engine = "PyTorch (FP16)"
    requested = "TensorRT" if enable_tensorrt else "PyTorch FP16"
    print(f'>>> Initializing {model_label()} on {device.upper()}; requested backend: {requested}', flush=True)
    print(f'>>> Initializing Text-to-Image pipeline (single-image {model_label()}, t_index_list=[0])...', flush=True)
    stream_txt2img = create_txt2img_pipeline()
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
            width=output_width,
            height=output_height,
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
                if diffusion_model == 'sdxl-turbo':
                    return self.stream.image_processor.postprocess(x_output, output_type='pil')[0]
                pil_out = postprocess_image(x_output, output_type="pil")
                return pil_out[0] if isinstance(pil_out, list) else pil_out

        stream_img2img = SharedImg2ImgWrapper(stream_core)
    except Exception as e:
        if diffusion_model == 'sdxl-turbo':
            raise RuntimeError('SDXL img2img initialization failed; no other model was substituted') from e
        print(">>> Fallback to standalone img2img wrapper:", e)
        stream_img2img = StreamDiffusionWrapper(
            model_id_or_path="stabilityai/sd-turbo",
            t_index_list=[t0_init, 45],
            frame_buffer_size=1,
            width=output_width,
            height=output_height,
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
        img2img_engine = "PyTorch (FP16)"

    print(">>> Preparing both pipelines with initial prompt...")
    stream_txt2img.prepare(prompt=current_prompt_a, num_inference_steps=50)
    stream_img2img.prepare(prompt=current_prompt_a, num_inference_steps=50)

    if enable_tensorrt:
        # Batch one serves txt2img; the two-timestep img2img stream uses batch
        # two. Compile and validate both before reporting readiness.
        txt_source, img_source = stream_txt2img.stream.unet, stream_img2img.stream.unet
        is_xl = diffusion_model == 'sdxl-turbo'
        txt_unet = compile_tensorrt_unet(txt_source.model if is_xl else txt_source, output_width, output_height, 1)
        img_unet = compile_tensorrt_unet(img_source.model if is_xl else img_source, output_width, output_height, 2)
        stream_txt2img.stream.unet = txt_source.with_model(txt_unet) if is_xl else txt_unet
        stream_img2img.stream.unet = img_source.with_model(img_unet) if is_xl else img_unet
        active_engine = img2img_engine = "TensorRT (UNet)"
        print(">>> TensorRT UNet acceleration enabled for BOTH generation modes; TinyVAE remains PyTorch.", flush=True)

    print(">>> Warming up models...")
    with torch.no_grad():
        for _ in range(4):
            stream_txt2img()
        dummy_img = Image.new("RGB", (output_width, output_height), color=(128, 128, 128))
        for _ in range(3):
            stream_img2img(dummy_img)

    pipeline_ready = True
    apply_prompts_and_blend()
    print(">>> StreamDiffusion pipeline is READY! <<<", flush=True)

@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(model_executor, load_pipeline)

@app.get("/")
def index():
    return HTMLResponse("""
    <html>
        <head><title>StreamDiffusion TouchDesigner Server</title></head>
        <body style="font-family: sans-serif; padding: 40px; background: #121212; color: #fff;">
            <h2>StreamDiffusion (SD-Turbo) Bridge Server</h2>
            <p>Status: <span style="color: #0f0;">Running</span></p>
            <p>Connect your TouchDesigner client via the WebSocket endpoint: <code>/ws/stream</code></p>
        </body>
    </html>
    """)


@app.get("/health")
def health():
    return JSONResponse({
        "ready": pipeline_ready, "protocol": PROTOCOL_VERSION,
        "mode": current_mode, "engine": engine_for_mode(),
        "model": model_label(), "diffusion_model": diffusion_model,
        "width": output_width, "height": output_height,
        "client_connected": active_client is not None,
        "last_error": last_pipeline_error,
        "requested_engine": engine_for_mode() if is_world_backend or is_krea_backend else "TensorRT" if enable_tensorrt else "PyTorch (FP16)",
        "backend": backend_name,
        "session_id": os.environ.get('BRIDGE_SESSION_ID'),
        "authentication_required": bool(access_token),
        "resolution_locked": resolution_locked(),
        **world_progress(),
    }, status_code=200 if pipeline_ready else 503)


def configuration_error(code):
    """Public status contains no internal exception text or stack information."""
    if code == 'resolution_locked':
        status = f'Output fixed at {output_width} x {output_height}; select the matching backend and startup size'
    elif code == "invalid_control":
        status = "Invalid controls: check backend, output size, and control values"
    elif is_krea_backend:
        status = 'Krea configuration failed; see server logs, then reset the clip'
    elif is_world_backend:
        status = 'World configuration failed; see server logs, then reset or reseed the world'
    elif pipeline_ready:
        status = "Configuration failed; previous settings restored. See server logs for details"
    else:
        status = "Configuration failed; see server logs, then retry or restart the server"
    return {"type": "error", "code": code, "status": status,
            "ready": pipeline_ready, "mode": current_mode,
            "width": output_width, "height": output_height, 'engine': engine_for_mode(),
            'resolution_locked': resolution_locked()}

async def authorize_websocket(websocket):
    # Authenticate before reserving the single controller or exposing frames.
    # Tokens travel in a message, never a URL or a request log.
    await websocket.accept()
    if access_token:
        await websocket.send_text(json.dumps({'type': 'auth_required'}))
        try:
            message = await asyncio.wait_for(websocket.receive(), timeout=10)
            if message.get('type') == 'websocket.disconnect':
                return False
            data = json.loads(message.get('text', ''))
            authenticated = (isinstance(data, dict) and data.get('type') == 'auth'
                             and isinstance(data.get('token'), str)
                             and hmac.compare_digest(data['token'].encode(), access_token.encode()))
        except (asyncio.TimeoutError, ValueError, TypeError, WebSocketDisconnect):
            authenticated = False
        if not authenticated:
            await websocket.send_text(json.dumps({'type': 'error', 'code': 'unauthorized',
                                                 'status': 'Access token missing or incorrect'}))
            await websocket.close(code=1008)
            return False
    return True


@app.websocket('/ws/ping')
async def websocket_ping_endpoint(websocket: WebSocket):
    if not await authorize_websocket(websocket):
        return
    try:
        await websocket.send_text(json.dumps({'type': 'probe_ready'}))
        while True:
            message = await websocket.receive()
            if message.get('type') == 'websocket.disconnect':
                break
            if message.get('text') and len(message['text']) <= 1024:
                try:
                    data = json.loads(message['text'])
                    if isinstance(data, dict) and data.get('type') == 'ping':
                        await websocket.send_text(json.dumps({'type': 'pong', 'sent_at': data.get('sent_at')}))
                except (TypeError, ValueError):
                    pass
    except WebSocketDisconnect:
        pass


@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    global active_client
    if not await authorize_websocket(websocket):
        return
    # Both modes share one mutable model; an additional controller must not
    # silently overwrite the active performer's prompts and resolution.
    if active_client is not None:
        await websocket.send_text(json.dumps({"type": "error", "code": "server_busy",
                                             "status": "Server already has an active TouchDesigner client"}))
        await websocket.close(code=1013)
        return
    active_client = websocket
    logger.info("TouchDesigner client connected to WebSocket.")
    loop = asyncio.get_running_loop()
    send_lock = asyncio.Lock()
    frame_lock = asyncio.Lock()
    frame_credit = asyncio.Event()
    frame_credit.set()
    ack_enabled, frame_window = False, 3
    video_quality = 'balanced'
    outstanding_frames = {}
    stop_event = asyncio.Event()
    frame_available = asyncio.Event()
    control_available = asyncio.Event()
    pending_frame = None
    pending_control = {}
    assembler = FrameAssembler(timeout=15.0 if is_world_backend else 1.0)
    frame_counter = 0
    latest_stats = {}
    world_seed_until = 0.0

    async def send_json(data):
        async with send_lock:
            await websocket.send_text(json.dumps(data))

    async def wait_for_frame_credit():
        started = time.perf_counter()
        while ack_enabled and len(outstanding_frames) >= frame_window:
            frame_credit.clear()
            await frame_credit.wait()
        return (time.perf_counter() - started) * 1000

    async def send_chunked_frame(jpg_bytes):
        nonlocal frame_counter
        chunk_size = OUTPUT_CHUNK_BYTES
        total_chunks = (len(jpg_bytes) + chunk_size - 1) // chunk_size
        if not 1 <= total_chunks <= 255:
            raise ValueError("Output frame exceeds the binary protocol chunk limit")
        async with frame_lock:
            credit_ms = await wait_for_frame_credit()
            frame_counter = (frame_counter + 1) % 256
            started = time.perf_counter()
            if ack_enabled:
                outstanding_frames[frame_counter] = started
            for chunk_idx in range(total_chunks):
                chunk = jpg_bytes[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
                async with send_lock:
                    await websocket.send_bytes(bytes([frame_counter, chunk_idx, total_chunks]) + chunk)
                if chunk_idx % 4 == 3:
                    await asyncio.sleep(0)
            return (time.perf_counter() - started) * 1000, credit_ms

    async def telemetry_loop():
        next_log = time.perf_counter() + 10
        while not stop_event.is_set():
            await asyncio.sleep(1.0)
            vram, temp = await loop.run_in_executor(None, get_gpu_telemetry)
            progress = world_progress()
            if latest_stats or progress:
                await send_json({
                    "type": "stats", "engine": engine_for_mode(), "model": model_label(),
                    "vram": vram, "temp": temp, **latest_stats,
                    'ready': pipeline_ready, 'inflight_frames': len(outstanding_frames),
                    **progress,
                })
                if time.perf_counter() >= next_log:
                    if progress and progress['world_phase'] != 'streaming':
                        logger.info('%s progress: %s', backend_name, progress['world_status'])
                    logger.info('delivery: %s, %s bytes/frame, send=%sms, display ACK RTT=%sms, '
                                'buffer wait=%sms, client cook=%sms, window=%d, inflight=%d',
                                video_quality, latest_stats.get('frame_bytes', '--'),
                                latest_stats.get('send_ms', '--'), latest_stats.get('delivery_ms', '--'),
                                latest_stats.get('credit_wait_ms', '--'), latest_stats.get('display_ms', '--'),
                                frame_window, len(outstanding_frames))
                    next_log = time.perf_counter() + 10

    async def control_loop():
        nonlocal pending_control, pending_frame
        while not stop_event.is_set():
            await control_available.wait()
            control_available.clear()
            data, pending_control = pending_control, {}
            try:
                data = validated_controls(data, (output_width, output_height))
                width, height = requested_resolution(data)
                if resolution_locked() and (width, height) != (output_width, output_height):
                    await send_json(configuration_error('resolution_locked'))
                    continue
                resizing = (width, height) != (output_width, output_height) or not pipeline_ready
                if resizing:
                    pending_frame = None
                    frame_available.clear()
                    assembler.clear()
                    backend = 'TensorRT' if enable_tensorrt else 'PyTorch'
                    await send_json({'type': 'status', 'status': f'Loading {width} x {height} {backend} output...',
                                     'ready': False, 'engine': f'{backend} (Preparing)'})
                await loop.run_in_executor(model_executor, apply_controls, data)
                await send_json({"type": "configured", "status": "Connected (Streaming)",
                                 "width": output_width, "height": output_height, "engine": engine_for_mode(),
                                 "ready": pipeline_ready, "mode": current_mode,
                                 'resolution_locked': resolution_locked(), **world_progress()})
                if current_mode not in ('img2img', 'world'):
                    pending_frame = None
                    frame_available.clear()
                    assembler.clear()
            except ValueError:
                logger.exception("Invalid control message")
                await send_json(configuration_error("invalid_control"))
            except Exception:
                logger.exception("Error applying control message")
                await send_json(configuration_error("configuration_failed"))

    async def generation_loop(mode):
        nonlocal pending_frame
        frame_count = 0
        while not stop_event.is_set():
            try:
                frame_bytes = None
                if mode == "img2img":
                    await frame_available.wait()
                    frame_available.clear()
                    frame_bytes, pending_frame = pending_frame, None
                    if frame_bytes is None:
                        continue
                elif not pipeline_ready or current_mode != mode:
                    await asyncio.sleep(0.01)
                    continue

                if mode in ('world', 'krea'):
                    # Stateful generation requires bounded delivery and a configured controller.
                    if not ack_enabled:
                        await asyncio.sleep(0.02)
                        continue
                    if mode == 'world' and pending_frame is not None:
                        frame_bytes, pending_frame = pending_frame, None
                    elif not (krea_backend if mode == 'krea' else world_backend).can_generate:
                        await asyncio.sleep(0.02)
                        continue

                if not pipeline_ready or current_mode != mode:
                    continue
                credit_ms = await wait_for_frame_credit()
                if not pipeline_ready or current_mode != mode:
                    continue
                if mode == 'img2img' and pending_frame is not None:
                    frame_bytes, pending_frame = pending_frame, None
                    frame_available.clear()
                started = time.perf_counter()
                revision = pipeline_revision
                frame_quality = video_quality
                budget = video_byte_budget(output_width, output_height, frame_quality)
                result = await loop.run_in_executor(model_executor, process_frame, mode, frame_bytes, budget)
                # Discard a completed frame if its mode was switched during inference.
                if result is not None and current_mode == mode and revision == pipeline_revision:
                    jpg_bytes, inf_ms, process_ms = result
                    send_ms, extra_credit_ms = await send_chunked_frame(jpg_bytes)
                    frame_count += 1
                    latest_stats.update(inf_ms=round(inf_ms, 1), process_ms=round(process_ms, 1),
                                        mode=mode, strength=current_strength, frame_bytes=len(jpg_bytes),
                                        video_quality=frame_quality, send_ms=round(send_ms, 1),
                                        credit_wait_ms=round(credit_ms + extra_credit_ms, 1))
                    if frame_count % 100 == 1:
                        logger.info("%s: frame #%d (inference %.1fms, processing %.1fms, %d bytes)",
                                    mode, frame_count, inf_ms, process_ms, len(jpg_bytes))
                if mode in ('txt2img', 'world', 'krea'):
                    await asyncio.sleep(max(0.0, 1.0 / max_stream_fps - (time.perf_counter() - started)))
            except (WebSocketDisconnect, RuntimeError, OSError):
                logger.exception("Streaming stopped in %s", mode)
                stop_event.set()
                # Wake receive() too, rather than leaving an idle connection alive.
                await websocket.close()
                return
            except Exception:
                logger.exception("Frame processing failed in %s", mode)
                await asyncio.sleep(0.05)

    async def supervise(coroutine):
        try:
            await coroutine
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Connection worker failed")
            stop_event.set()
            try:
                await websocket.close()
            except Exception:
                pass

    tasks = []
    try:
        if is_world_backend or is_krea_backend:
            # Queue behind any last GPU call from the previous controller.
            await loop.run_in_executor(model_executor, (krea_backend if is_krea_backend else world_backend).release_controls)
        vram, temp = await loop.run_in_executor(None, get_gpu_telemetry)
        await send_json({
            "type": "welcome", "message": "StreamDiffusion server connected",
            "engine": engine_for_mode(), "prompt_a": current_prompt_a,
            "model": model_label(), "diffusion_model": diffusion_model,
            "prompt_b": current_prompt_b, "blend": current_prompt_blend,
            "mode": current_mode, "seed": current_seed, "strength": current_strength,
            "width": output_width, "height": output_height,
            "protocol": PROTOCOL_VERSION, "ready": pipeline_ready,
            "vram": vram, "temp": temp,
            'resolution_locked': resolution_locked(), 'latency_probe': True, 'frame_ack': True,
            'video_quality_profiles': list(VIDEO_QUALITY_LIMITS),
            'backend': backend_name,
            'supported_modes': ['krea'] if is_krea_backend else ['world'] if is_world_backend else ['txt2img', 'img2img'],
            'world_seed_required': is_world_backend and world_backend.seed_image is None,
            'inference_batch_frames': 12 if is_krea_backend else 4 if is_world_backend else 1,
            **world_progress(),
        })
        modes = ('krea',) if is_krea_backend else ('world',) if is_world_backend else ('txt2img', 'img2img')
        tasks = [asyncio.create_task(supervise(generation_loop(mode))) for mode in modes]
        tasks.append(asyncio.create_task(supervise(telemetry_loop())))
        tasks.append(asyncio.create_task(supervise(control_loop())))
        while not stop_event.is_set():
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("text") is not None:
                try:
                    data = json.loads(message["text"])
                    if not isinstance(data, dict):
                        continue
                    if data.get("type") == "ping":
                        await send_json({"type": "pong", "sent_at": data.get("sent_at")})
                    elif data.get('type') == 'world_seed' and is_world_backend:
                        assembler.clear()
                        world_seed_until = time.monotonic() + 15
                        logger.info('Waypoint seed upload requested')
                    elif data.get('type') == 'client_options':
                        quality = data.get('video_quality', video_quality)
                        if quality not in VIDEO_QUALITY_LIMITS:
                            raise ValueError('Invalid video quality')
                        value = data.get('frame_window', 3)
                        if data.get('frame_ack') is True and (not isinstance(value, int)
                                or isinstance(value, bool) or not 1 <= value <= 6):
                            raise ValueError('Invalid frame window')
                        video_quality = quality
                        if data.get('frame_ack') is True:
                            ack_enabled, frame_window = True, value
                            frame_credit.set()
                    elif data.get('type') == 'frame_ack':
                        frame_id = data.get('frame_id')
                        if isinstance(frame_id, int) and not isinstance(frame_id, bool):
                            sent = outstanding_frames.pop(frame_id, None)
                            if sent is not None:
                                frame_credit.set()
                                latest_stats['delivery_ms'] = round((time.perf_counter() - sent) * 1000, 1)
                                for name in ('display_ms', 'client_frame_ms'):
                                    value = data.get(name)
                                    if (isinstance(value, (int, float)) and not isinstance(value, bool)
                                            and 0 <= value <= 60000 and np.isfinite(value)):
                                        latest_stats[name] = round(value, 1)
                    elif data.get("type") == "control" or any(
                        key in data for key in ("prompt", "prompt_a", "mode", "seed", "strength", "blend")
                    ):
                        # Merge rapid slider/audio changes while the worker is
                        # busy; the latest value of each control wins.
                        world_pulses = {key: True for key in ('world_reset', 'world_reseed', 'krea_reset')
                                        if pending_control.get(key) is True}
                        pending_control.update(data)
                        if is_world_backend or is_krea_backend:
                            pending_control.update(world_pulses)
                        control_available.set()
                except (ValueError, TypeError):
                    await send_json({"type": "error", "code": "invalid_message",
                                     "status": "Send a valid JSON control object"})
            elif (message.get("bytes") is not None and pipeline_ready
                  and (current_mode == 'img2img' or (is_world_backend and time.monotonic() < world_seed_until))):
                raw = message["bytes"]
                try:
                    frame_bytes = assembler.feed(raw)
                except ValueError as exc:
                    logger.debug("Dropped invalid input frame: %s", exc)
                    continue
                if frame_bytes:
                    world_seed_until = 0.0
                    if is_world_backend:
                        logger.info('Waypoint seed upload received: %d bytes', len(frame_bytes))
                        await send_json({'type': 'world_progress', 'world_phase': 'seeding',
                                         'world_status': 'World seed received; preparing seed and kernels'})
                    # Only one pending input: replace stale watchdog/recovery frames.
                    # Receiving continues while the worker is busy with inference.
                    pending_frame = frame_bytes
                    frame_available.set()
    except WebSocketDisconnect:
        logger.info("TouchDesigner client disconnected.")
    except Exception:
        logger.exception("WebSocket error")
    finally:
        stop_event.set()
        for task in tasks:
            task.cancel()
        backend_to_release = krea_backend if is_krea_backend else world_backend if is_world_backend else None
        def release_controller():
            global active_client
            try:
                if backend_to_release is not None:
                    backend_to_release.release_controls()
            finally:
                if active_client is websocket:
                    active_client = None
        # Submit before any cancellable await. An ASGI cancellation must not
        # cancel this worker cleanup or leave its controller slot reserved.
        # The worker serializes it behind already-running model operations.
        cleanup = model_executor.submit(release_controller)
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            await asyncio.shield(asyncio.wrap_future(cleanup))


if __name__ == "__main__":
    import uvicorn
    # JPEG is already compressed; per-message deflate only adds CPU work.
    uvicorn.run(app, host=os.environ.get('BRIDGE_HOST', '0.0.0.0'),
                port=int(os.environ.get('BRIDGE_PORT', '8000')), log_level="info", ws_per_message_deflate=False,
                ws_max_size=MAX_INPUT_FRAME_BYTES + 7)
