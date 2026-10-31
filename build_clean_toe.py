import os
import struct
import shutil
import subprocess
from pathlib import Path
import tempfile
import argparse
import re
from bridge_protocol import OUTPUT_PRESETS


def resolution_parameter():
    options = ' '.join(f'{value} "{label}"' for value, label, _, _ in OUTPUT_PRESETS)
    return ('772804879 Resolution "Output Resolution / Ratio" 1 1 0 0 1 1 1 2 0 "" "" '
            f'StreamDiffusion 4097 {len(OUTPUT_PRESETS)} {options} 3')

SCRIPT_RECEIVER_CODE = '''import cv2
import numpy as np

# me - this DAT
# scriptOp - the OP which is cooking
decoded_bytes = None
decoded_frame = None

def onSetupParameters(scriptOp):
    return

def onPulse(par):
    return

def onCook(scriptOp):
    global decoded_bytes, decoded_frame
    p_mode = getattr(parent().par, 'Displaymode', None)
    disp_mode = str(p_mode.eval()) if p_mode is not None else 'ai'
    # Fetch latest reassembled frame from parent component storage
    data = parent().fetch('last_frame_bytes', None)
    if not data:
        for name in ['websocket1_callbacks', 'websocket1_callbacks1']:
            cb = op(name)
            if cb and hasattr(cb.module, 'last_frame_bytes') and cb.module.last_frame_bytes:
                data = cb.module.last_frame_bytes
                break
    
    if not data and disp_mode != 'input':
        return
        
    try:
        if data and disp_mode != 'input' and data is not decoded_bytes:
            decoded_frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            decoded_bytes = data
        ai_bgr = decoded_frame if data and disp_mode != 'input' else None
        if ai_bgr is None and disp_mode != 'input':
            return
            
        h, w = ai_bgr.shape[:2] if ai_bgr is not None else (None, None)
        
        final_bgr = ai_bgr
        
        if disp_mode in ('sidebyside', 'wipe', 'input'):
            cam = op('null_cam')
            cam_bgr = None
            if cam:
                cam_arr = cam.numpyArray()
                if cam_arr is not None:
                    c_rgb = (np.clip(cam_arr[:, :, :3], 0.0, 1.0) * 255).astype(np.uint8)
                    cam_bgr = cv2.cvtColor(c_rgb, cv2.COLOR_RGB2BGR)
                    cam_bgr = cv2.flip(cam_bgr, 0)
                    if h is None:
                        h, w = cam_bgr.shape[:2]
                    if cam_bgr.shape[:2] != (h, w):
                        cam_bgr = cv2.resize(cam_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
            
            if cam_bgr is not None:
                if disp_mode == 'input':
                    final_bgr = cam_bgr
                elif disp_mode == 'sidebyside':
                    sep = np.zeros((h, 4, 3), dtype=np.uint8)
                    sep[:, :] = (0, 230, 255)  # Cyan/Yellow separator line
                    final_bgr = np.hstack([cam_bgr, sep, ai_bgr])
                elif disp_mode == 'wipe':
                    mid = w // 2
                    wipe_img = np.empty_like(ai_bgr)
                    wipe_img[:, :mid] = cam_bgr[:, :mid]
                    wipe_img[:, mid:] = ai_bgr[:, mid:]
                    wipe_img[:, max(0, mid-1):min(w, mid+2)] = (0, 230, 255)
                    final_bgr = wipe_img
        
        if final_bgr is None:
            return
        out_h, out_w = final_bgr.shape[:2]
        if scriptOp.par.resolutionw.eval() != out_w:
            scriptOp.par.resolutionw = out_w
        if scriptOp.par.resolutionh.eval() != out_h:
            scriptOp.par.resolutionh = out_h
            
        rgba = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGBA)
        rgba_flipped = np.ascontiguousarray(np.flipud(rgba))
        scriptOp.copyNumpyArray(rgba_flipped)
    except Exception as e:
        print('[StreamDiffusion] Error in script_receiver onCook:', e)
'''

WEBSOCKET_CODE = '''import json
import time

# me - this DAT
# dat - the OP which is cooking

current_frame_id = -1
frame_chunks = {}
total_received = 0
last_frame_bytes = None
last_fps_time = time.perf_counter()
fps_frame_count = 0
last_error = None
current_chunk_count = 0
frame_started = 0.0

def onConnect(dat):
    global last_fps_time, fps_frame_count, last_error
    last_error = None
    last_fps_time = time.perf_counter()
    fps_frame_count = 0
    p_status = getattr(parent().par, 'Status', None)
    if p_status is not None:
        p_status.val = 'Connected (Streaming)'
    parent().store('can_send', True)
    parent().store('connected', True)
    parent().store('server_ready', False)
    parent().store('last_frame_receive', time.perf_counter())
    parent().store('last_send_time', 0.0)
    parent().store('last_ping_time', 0.0)
    print('[StreamDiffusion] WebSocket Connected to Colab!')
    send_control(force=True)

def onDisconnect(dat):
    p_status = getattr(parent().par, 'Status', None)
    if p_status is not None:
        p_status.val = last_error or 'Disconnected'
    parent().store('connected', False)
    parent().store('server_ready', False)
    p_eng = getattr(parent().par, 'Engine', None)
    if p_eng is not None:
        p_eng.val = '--'
    p_fps = getattr(parent().par, 'Fps', None)
    if p_fps is not None:
        p_fps.val = '0.0 FPS'
    p_gpu = getattr(parent().par, 'Gputime', None)
    if p_gpu is not None:
        p_gpu.val = '-- ms'
    p_ping = getattr(parent().par, 'Ping', None)
    if p_ping is not None:
        p_ping.val = '-- ms'
    p_vram = getattr(parent().par, 'Vram', None)
    if p_vram is not None:
        p_vram.val = '--'
    p_temp = getattr(parent().par, 'Gputemp', None)
    if p_temp is not None:
        p_temp.val = '--'
    for name in ['Frametime', 'Servertime']:
        par = getattr(parent().par, name, None)
        if par is not None:
            par.val = '-- ms'
    parent().store('can_send', True)
    print('[StreamDiffusion] WebSocket Disconnected.')
    global current_frame_id, frame_chunks
    current_frame_id = -1
    frame_chunks.clear()

def onReceiveText(dat, rowIndex, message):
    global last_error
    try:
        data = json.loads(message)
        if not isinstance(data, dict):
            return
        msg_type = data.get('type', '')
        if msg_type == 'welcome' and data.get('protocol') != 1:
            last_error = 'Bridge versions differ: reload the updated Colab notebook and component'
            parent().store('server_ready', False)
            p_status = getattr(parent().par, 'Status', None)
            if p_status is not None:
                p_status.val = last_error
            p_connect = getattr(parent().par, 'Connect', None)
            if p_connect is not None:
                p_connect.val = False
            print('[StreamDiffusion]', last_error)
            return
        if msg_type == 'error':
            last_error = str(data.get('status', 'Server error'))
            parent().store('server_ready', bool(data.get('ready', False)))
            if data.get('ready') and 'width' in data and 'height' in data:
                width, height = int(data['width']), int(data['height'])
                p_resolution = getattr(parent().par, 'Resolution', None)
                if p_resolution is not None:
                    p_resolution.val = str(width) if width == height else f"{width}x{height}"
                p_mode = getattr(parent().par, 'Mode', None)
                if p_mode is not None and data.get('mode') in ('txt2img', 'img2img'):
                    p_mode.val = data['mode']
            print('[StreamDiffusion]', last_error)
        elif msg_type in ('welcome', 'configured'):
            actual = (data.get('mode'), data.get('width'), data.get('height'))
            requested = parent().fetch('requested_configuration', actual)
            ready = bool(data.get('ready', True)) and actual == requested
            parent().store('server_ready', ready)
            if ready:
                last_error = None
        elif msg_type == 'status' and str(data.get('status', '')).startswith('Loading'):
            parent().store('server_ready', False)
        if msg_type == 'pong' and isinstance(data.get('sent_at'), (int, float)):
            p_ping = getattr(parent().par, 'Ping', None)
            if p_ping is not None:
                p_ping.val = f"{(time.perf_counter() - data['sent_at']) * 1000:.1f} ms"
        
        if 'engine' in data:
            p_eng = getattr(parent().par, 'Engine', None)
            if p_eng is not None:
                p_eng.val = data['engine']

        if msg_type == 'welcome':
            vram = data.get('vram', '--')
            temp = data.get('temp', '--')
            p_vram = getattr(parent().par, 'Vram', None)
            if p_vram is not None and vram != '--':
                p_vram.val = vram
            p_temp = getattr(parent().par, 'Gputemp', None)
            if p_temp is not None and temp != '--':
                p_temp.val = temp
            eng = data.get('engine', 'PyTorch')
            print(f"[StreamDiffusion] Handshake received. Engine: {eng}, VRAM: {vram}, Temp: {temp}")
            
        if 'status' in data:
            p_status = getattr(parent().par, 'Status', None)
            if p_status is not None:
                p_status.val = data['status']
                
        if 'inf_ms' in data:
            p_gpu = getattr(parent().par, 'Gputime', None)
            if p_gpu is not None:
                p_gpu.val = f"{data['inf_ms']:.1f} ms"
        if 'process_ms' in data:
            p_server = getattr(parent().par, 'Servertime', None)
            if p_server is not None:
                p_server.val = f"{data['process_ms']:.1f} ms"
                
        if 'vram' in data:
            p_vram = getattr(parent().par, 'Vram', None)
            if p_vram is not None and data['vram'] != '--':
                p_vram.val = data['vram']
                
        if 'temp' in data:
            p_temp = getattr(parent().par, 'Gputemp', None)
            if p_temp is not None and data['temp'] != '--':
                p_temp.val = data['temp']
    except Exception:
        pass

def onReceiveBinary(dat, contents):
    global current_frame_id, frame_chunks, total_received, last_frame_bytes
    global last_fps_time, fps_frame_count
    global current_chunk_count, frame_started
    
    if not contents or len(contents) < 4:
        return
        
    frame_id = contents[0]
    chunk_idx = contents[1]
    total_chunks = contents[2]
    chunk_data = contents[3:]
    if total_chunks == 0 or chunk_idx >= total_chunks:
        return
    
    now = time.perf_counter()
    if frame_id != current_frame_id or now - frame_started > 1.0:
        current_frame_id = frame_id
        frame_chunks = {}
        current_chunk_count = total_chunks
        frame_started = now
    if total_chunks != current_chunk_count or len(chunk_data) > 2048:
        frame_chunks.clear()
        current_frame_id = -1
        return
        
    frame_chunks[chunk_idx] = chunk_data
    
    if len(frame_chunks) == total_chunks:
        assembled = b"".join(frame_chunks[i] for i in range(total_chunks))
        last_frame_bytes = assembled
        parent().store('last_frame_bytes', assembled)
        parent().store('display_pending', True)
        parent().store('last_frame_receive', time.perf_counter())
        
        # Adaptive Flow Control: unlock sending immediately upon frame receipt
        parent().store('can_send', True)
        total_received += 1
        fps_frame_count += 1
        
        now = time.perf_counter()
        # Input-to-result round trip includes inference and image transport.
        last_send_time = parent().fetch('last_send_time', 0.0)
        if last_send_time > 0 and parent().par.Mode.eval() == 'img2img':
            rtt = (now - last_send_time) * 1000.0
            p_frame = getattr(parent().par, 'Frametime', None)
            if p_frame is not None:
                p_frame.val = f"{rtt:.1f} ms"
            
        # Telemetry: Effective FPS
        elapsed = now - last_fps_time
        if elapsed >= 1.0:
            current_fps = fps_frame_count / elapsed
            p_fps = getattr(parent().par, 'Fps', None)
            if p_fps is not None:
                p_fps.val = f"{current_fps:.1f} FPS"
            p_status = getattr(parent().par, 'Status', None)
            if p_status is not None and parent().fetch('server_ready', True):
                p_status.val = f"Connected ({current_fps:.1f} FPS)"
            last_fps_time = now
            fps_frame_count = 0
        
        # The frame-start callback displays only the newest frame once per TD
        # frame, rather than decoding every buffered network frame immediately.
        frame_chunks.clear()

def output_size():
    p_resolution = getattr(parent().par, 'Resolution', None)
    value = str(p_resolution.eval()) if p_resolution is not None else '512'
    dimensions = value.lower().split('x')
    return int(dimensions[0]), int(dimensions[-1])


def send_control(force=False, strength_override=None):
    p = parent().par
    mode = 'img2img' if getattr(p, 'Mode', None) and p.Mode.eval() == 'img2img' else 'txt2img'
    
    prompt_a = str(getattr(p, 'Prompta', getattr(p, 'Prompt', '')))
    prompt_b = str(getattr(p, 'Promptb', ''))
    blend_val = float(getattr(p, 'Promptblend', 0.0))
    neg_prompt = str(getattr(p, 'Negprompt', ''))
    
    if strength_override is not None:
        strength_val = float(strength_override)
    else:
        strength_val = float(p.Strength.eval()) if getattr(p, 'Strength', None) else 0.6
        
    seed_val = int(p.Seed.eval()) if getattr(p, 'Seed', None) else 42
    width, height = output_size()
    configuration = (mode, width, height)
    if configuration != parent().fetch('requested_configuration', None):
        parent().store('server_ready', False)
        parent().store('requested_configuration', configuration)
    
    msg = {
        'type': 'control',
        'mode': mode,
        'prompt_a': prompt_a,
        'prompt_b': prompt_b,
        'blend': blend_val,
        'neg_prompt': neg_prompt,
        'strength': strength_val,
        'seed': seed_val,
        'force': force,
        'width': width,
        'height': height,
    }
    ws = op('websocket1')
    if ws and ws.par.active.eval():
        try:
            ws.sendText(json.dumps(msg))
            if strength_override is None:
                print(f"[StreamDiffusion] Sent control: mode={mode}, output={msg['width']}x{msg['height']}, blend={blend_val:.2f}, seed={seed_val}")
        except Exception as e:
            print('[StreamDiffusion] Error sending control:', e)
    else:
        print(f"[StreamDiffusion] WebSocket offline, queued parameters.")
'''

EXEC_SEND_FRAME_CODE = '''import cv2
import numpy as np
import time
import random
import json

# me - this DAT
# frame - current timeline frame number

frame_count = 0
frame_id_seq = 0
last_audio_send_time = 0.0
last_audio_beat_time = 0.0
audio_armed = True

def onStart():
    return

def onCreate():
    return

def onExit():
    return

def handle_audio_reactivity(now):
    global last_audio_send_time, last_audio_beat_time, audio_armed
    p = parent().par
    p_aud = getattr(p, 'Audiorx', None)
    if not p_aud or not p_aud.eval():
        audio_armed = True
        return

    # Check connected in_audio CHOP or designated Audio CHOP Source
    audio_op = op('in_audio')
    if not audio_op or audio_op.numChans == 0:
        p_src = getattr(p, 'Audiochop', None)
        if p_src and p_src.eval():
            audio_op = op(p_src.eval())

    if not audio_op or audio_op.numChans == 0:
        return

    try:
        # Read peak audio energy across channels
        vals = [abs(float(audio_op[i].eval())) for i in range(audio_op.numChans)]
        amp = max(vals) if vals else 0.0
        
        # 1. Beat Detection / Trigger Seed Randomize
        thresh = float(getattr(p, 'Audseedthresh', 0.8))
        rising = audio_armed and amp >= thresh
        if amp >= thresh:
            audio_armed = False
        elif amp < thresh * 0.75:
            audio_armed = True
        if rising and (now - last_audio_beat_time > 0.35):
            last_audio_beat_time = now
            p_seed = getattr(p, 'Seed', None)
            if p_seed is not None:
                p_seed.val = random.randint(1, 999999)
                print(f"[StreamDiffusion] Audio Beat Detected (amp={amp:.2f} >= {thresh:.2f}) -> Random Seed {p_seed.val}")

        # Text mode reacts through seeds; preparing an unused img2img denoise
        # schedule on every audio change would compete with text inference.
        mode = getattr(p, 'Mode', None)
        if mode is not None and mode.eval() != 'img2img':
            return

        # 2. Audio-driven Denoise Modulation (throttled to max 10 updates/sec)
        gain = float(getattr(p, 'Audstrength', 0.3))
        base_str = float(getattr(p, 'Strength', 0.6))
        eff_str = min(1.0, max(0.1, base_str + amp * gain))
        
        last_sent = parent().fetch('last_sent_audio_str', base_str)
        if abs(eff_str - last_sent) > 0.06 and (now - last_audio_send_time > 0.1):
            last_audio_send_time = now
            parent().store('last_sent_audio_str', eff_str)
            for name in ['websocket1_callbacks', 'websocket1_callbacks1']:
                cb = op(name)
                if cb and hasattr(cb.module, 'send_control'):
                    cb.module.send_control(force=True, strength_override=eff_str)
                    break
    except Exception as e:
        pass

def onFrameStart(frame):
    global frame_count, frame_id_seq
    now = time.perf_counter()

    display_par = getattr(parent().par, 'Displaymode', None)
    input_preview = display_par is not None and display_par.eval() == 'input'
    if input_preview or parent().fetch('display_pending', False):
        parent().store('display_pending', False)
        receiver = op('script_receiver')
        if receiver:
            receiver.cook(force=True)

    if parent().fetch('control_pending', False):
        parent().store('control_pending', False)
        for name in ['websocket1_callbacks', 'websocket1_callbacks1']:
            cb = op(name)
            if cb and hasattr(cb.module, 'send_control'):
                cb.module.send_control(force=True)
                break

    ws = op('websocket1')
    if parent().fetch('connected', False) and now - parent().fetch('last_frame_receive', now) > 2.0:
        p_fps = getattr(parent().par, 'Fps', None)
        if p_fps is not None:
            p_fps.val = '0.0 FPS'
    if ws and ws.par.active.eval() and now - parent().fetch('last_ping_time', 0.0) >= 2.0:
        parent().store('last_ping_time', now)
        try:
            ws.sendText(json.dumps({'type': 'ping', 'sent_at': now}))
        except Exception:
            pass
    
    # Process audio reactivity across both txt2img and img2img modes
    handle_audio_reactivity(now)
    
    # Only active when in img2img mode
    mode_par = getattr(parent().par, 'Mode', None)
    if not mode_par or mode_par.eval() != 'img2img':
        return
        
    if not ws or not ws.par.active.eval():
        return
    if not parent().fetch('connected', False) or not parent().fetch('server_ready', False):
        return

    # Adaptive Flow Control:
    # Only send when server is ready (replied to previous frame)
    # or after 250ms watchdog timeout to recover from any dropped packet
    can_send = parent().fetch('can_send', True)
    last_send_time = parent().fetch('last_send_time', 0.0)
    
    if not can_send and (now - last_send_time < 0.25):
        return

    # Fit on the GPU before downloading the texture, reducing readback by 4x
    # at the default 512 output resolution. Keep the full-size comparison input.
    cam = op('fit_send') or op('null_cam')
    if not cam:
        return
        
    try:
        cam.cook(force=True)
        arr = cam.numpyArray()
        if arr is not None:
            # Float32 texture (0.0..1.0) to uint8 (0..255)
            img_rgb = (np.clip(arr[:, :, :3], 0.0, 1.0) * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(np.ascontiguousarray(img_rgb[::-1]), cv2.COLOR_RGB2BGR)
            
            # Keep the chosen aspect ratio, with upload's longest edge <=256.
            height, width = img_bgr.shape[:2]
            scale = min(1.0, 256.0 / max(width, height))
            img_small = img_bgr if scale == 1.0 else cv2.resize(
                img_bgr, (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA)
            ret, buf = cv2.imencode('.jpg', img_small, [int(cv2.IMWRITE_JPEG_QUALITY), 55])
            if ret:
                data = buf.tobytes()
                
                # Chunk into <= 4096 byte slices to strictly stay under OpenSSL TLS record limit (16KB)
                CHUNK_SIZE = 4096
                total_chunks = (len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE
                frame_id_seq = (frame_id_seq + 1) % 256
                
                for chunk_idx in range(total_chunks):
                    start = chunk_idx * CHUNK_SIZE
                    end = min(start + CHUNK_SIZE, len(data))
                    chunk = data[start:end]
                    # 7-byte chunk header: b"FRM:" (4B) + frame_id (1B) + chunk_idx (1B) + total_chunks (1B)
                    header = bytes([70, 82, 77, 58, frame_id_seq, chunk_idx, total_chunks])
                    ws.sendBinary(header + chunk)
                
                # Mark as waiting for server reply
                parent().store('can_send', False)
                parent().store('last_send_time', now)
                frame_count += 1
                if frame_count % 30 == 1:
                    print(f"[StreamDiffusion] Streaming input frame #{frame_count} ({len(data)} bytes in {total_chunks} chunk(s))")
    except Exception as e:
        print('[StreamDiffusion] Error in exec_send_frame:', e)

def onFrameEnd(frame):
    return

def onPlayStateChange(state):
    return

def onDeviceChange():
    return

def onProjectPreSave():
    return

def onProjectPostSave():
    return
'''

PAREXEC_CODE = '''import os
import json
import time
import datetime
import random
import cv2
import numpy as np
from urllib.parse import urlsplit, urlunsplit

# me - this DAT
# par - the parameter that changed
# prev - the previous value

def onValueChange(par, prev):
    try:
        if par.name == 'Serverurl':
            value = str(par.eval()).strip()
            if value:
                parsed = urlsplit(value)
                scheme = {'http': 'ws', 'https': 'wss'}.get(parsed.scheme, parsed.scheme)
                if scheme not in ('ws', 'wss') or not parsed.netloc:
                    parent().par.Connect = False
                    parent().par.Status = 'Paste a valid Colab WebSocket URL'
                    return
                path = '/ws/stream' if parsed.path in ('', '/') else parsed.path
                normalized = urlunsplit((scheme, parsed.netloc, path, parsed.query, ''))
                if normalized != value:
                    par.val = normalized
        elif par.name in ['Prompta', 'Promptb', 'Promptblend', 'Negprompt', 'Prompt', 'Mode', 'Source', 'Strength', 'Seed', 'Resolution']:
            # Coalesce simultaneous CHOP/slider changes into one control packet.
            parent().store('control_pending', True)
        elif par.name == 'Displaymode':
            rec = op('script_receiver')
            if rec:
                rec.cook(force=True)
        elif par.name == 'Connect':
            if par.eval() and not str(parent().par.Serverurl.eval()).strip():
                par.val = False
                parent().par.Status = 'Paste the Colab WebSocket URL first'
            elif not par.eval():
                p_status = getattr(parent().par, 'Status', None)
                if p_status is not None:
                    p_status.val = 'Disconnected'
    except Exception as e:
        print('[StreamDiffusion] parexec onValueChange error:', e)

def onPulse(par):
    try:
        if par.name == 'Randomseed':
            p_seed = getattr(parent().par, 'Seed', None)
            if p_seed is not None:
                p_seed.val = random.randint(1, 999999)
        elif par.name in ['Sendprompt', 'Submitprompt']:
            for name in ['websocket1_callbacks', 'websocket1_callbacks1']:
                cb = op(name)
                if cb and hasattr(cb.module, 'send_control'):
                    cb.module.send_control(force=True)
                    break
        elif par.name == 'Snapshot':
            save_snapshot()
    except Exception as e:
        print('[StreamDiffusion] parexec onPulse error:', e)

def save_snapshot():
    try:
        data = parent().fetch('last_frame_bytes', None)
        if not data:
            print('[StreamDiffusion] Cannot take snapshot: No AI frame received yet.')
            return
            
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            print('[StreamDiffusion] Cannot take snapshot: Failed to decode frame.')
            return
            
        snap_dir_par = getattr(parent().par, 'Snapdir', None)
        dir_name = snap_dir_par.eval() if snap_dir_par else 'captures'
        if not os.path.isabs(dir_name):
            dir_name = os.path.join(project.folder, dir_name)
        os.makedirs(dir_name, exist_ok=True)
        
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        base_fn = f"sd_capture_{ts}"
        img_path = os.path.join(dir_name, f"{base_fn}.png")
        meta_path = os.path.join(dir_name, f"{base_fn}.json")
        
        if not cv2.imwrite(img_path, img):
            raise OSError('Could not write snapshot PNG: ' + img_path)
        
        p = parent().par
        meta = {
            'timestamp': datetime.datetime.now().isoformat(),
            'prompt_a': str(getattr(p, 'Prompta', getattr(p, 'Prompt', ''))),
            'prompt_b': str(getattr(p, 'Promptb', '')),
            'prompt_blend': float(getattr(p, 'Promptblend', 0.0)),
            'neg_prompt': str(getattr(p, 'Negprompt', '')),
            'strength': float(getattr(p, 'Strength', 0.6)),
            'seed': int(getattr(p, 'Seed', 42)),
            'mode': str(getattr(p, 'Mode', 'txt2img')),
            'engine': str(getattr(p, 'Engine', '')),
            'fps': str(getattr(p, 'Fps', '')),
            'inference_time': str(getattr(p, 'Gputime', '')),
            'output_resolution': f"{img.shape[1]}x{img.shape[0]}",
            'output_width': img.shape[1],
            'output_height': img.shape[0],
            'server_processing': str(getattr(p, 'Servertime', '')),
            'frame_round_trip': str(getattr(p, 'Frametime', '')),
            'network_ping': str(getattr(p, 'Ping', '')),
            'vram': str(getattr(p, 'Vram', '')),
            'gpu_temp': str(getattr(p, 'Gputemp', ''))
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
            
        print(f"[StreamDiffusion] Snapshot saved successfully to:\\n  {img_path}\\n  {meta_path}")
        p_status = getattr(parent().par, 'Status', None)
        if p_status is not None:
            p_status.val = f"Saved: {base_fn}.png"
    except Exception as e:
        print('[StreamDiffusion] Error saving snapshot:', e)
'''

CPARM_CONTENT = f'''?
pages 1 StreamDiffusion
772804868 Serverurl "Colab WebSocket URL" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 0
772804867 Connect "Connect to Colab" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 1
772804879 Mode "Generation Mode" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 4097 2 txt2img "Text-to-Video" img2img "Image / Noise to Image" 2
{resolution_parameter()}
772804879 Source "Input Source" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 4097 3 in1 "External Input (in1)" cam "Webcam (cam_in)" noise "Internal Noise" 4
772804879 Displaymode "Display Mode" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 4097 4 ai "AI Output Only" sidebyside "Side-by-Side (Input | Output)" wipe "Split Wipe (50/50)" input "Input Only" 5
772804868 Prompta "Prompt A" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 6
772804868 Promptb "Prompt B" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 7
772804865 Promptblend "Prompt Blend (A <-> B)" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 8
772804868 Negprompt "Negative Prompt" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 9
772804869 Submitprompt "Submit Prompts" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 10
772804865 Strength "Denoise Strength" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 11
772804866 Seed Seed 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 12
772804869 Randomseed "Randomize Seed" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 13
772804869 Snapshot "Take Snapshot (PNG+JSON)" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 14
772804868 Snapdir "Snapshot Directory" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 15
772804867 Audiorx "Audio Reactivity" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 16
772804868 Audiochop "Audio CHOP Source" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 17
772804865 Audstrength "Audio -> Denoise Gain" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 18
772804865 Audseedthresh "Audio Beat Seed Threshold" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 19
772804868 Status Status 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 20
772804868 Engine "Acceleration Engine" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 21
772804868 Fps "Effective FPS" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 22
772804868 Gputime "Colab Inference" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 23
772804868 Ping "Network Ping (RTT)" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 24
772804868 Vram "Colab GPU VRAM" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 25
772804868 Gputemp "Colab GPU Temp" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 26
772804868 Servertime "Server Frame Processing" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 27
772804868 Frametime "Frame Round Trip" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 28
?
'''

PARM_CONTENT = '''?
pageindex 0 3
Serverurl 67108928 ""
Connect 67108928 off
Mode 67108928 txt2img
Resolution 67108928 512
Source 67108928 in1
Displaymode 67108928 ai
Prompta 67108928 "cinematic cyberpunk cityscape, neon lights, 8k, detailed"
Promptb 67108928 "bioluminescent alien jungle, glowing flora, ethereal, 8k"
Promptblend 67108928 0.0
Negprompt 67108928 "blurry, low quality, distorted, watermark"
Strength 67108928 0.6
Seed 67108928 42
Snapdir 67108928 captures
Audiorx 67108928 off
Audiochop 67108928 in_audio
Audstrength 67108928 0.3
Audseedthresh 67108928 0.8
Status 67109184 Disconnected
Engine 67109184 "--"
Fps 67109184 "0.0 FPS"
Gputime 67109184 "-- ms"
Ping 67109184 "-- ms"
Vram 67109184 "--"
Gputemp 67109184 "--"
Servertime 67109184 "-- ms"
Frametime 67109184 "-- ms"
?
'''

IN_AUDIO_N = '''CHOP:in
tile -550 -350 130 90
flags =  parlanguage 0
color 0.67 0.67 0.67 
end
'''

IN_AUDIO_PARM = '''?
label 201326609 "" me.name
?
'''

def make_td_text(code_str: str) -> bytes:
    code_bytes = code_str.encode('utf-8')
    header = (
        b'2\n'
        + struct.pack('<5I', 42, 1, 1, 1, 1)
        + b'\x02'
        + struct.pack('>I', len(code_bytes))
    )
    return header + code_bytes

def write_text_file(path: str, code_str: str):
    data = make_td_text(code_str)
    with open(path, 'wb') as f:
        f.write(data)
    print(f"Wrote {path} (total: {len(data)} bytes, code: {len(code_str.encode('utf-8'))} bytes)")

def find_collapse_executable(location=None):
    location = location or os.environ.get('TOUCHDESIGNER_BIN')
    candidates = []
    if location:
        path = Path(location).expanduser()
        candidates.append(path / 'toecollapse.exe' if path.is_dir() else path)
    else:
        executable = shutil.which('toecollapse')
        if executable:
            candidates.append(Path(executable))
        install_root = Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Derivative'
        candidates.append(install_root / 'TouchDesigner/bin/toecollapse.exe')
        if install_root.exists():
            candidates.extend(sorted(install_root.glob('TouchDesigner.*/bin/toecollapse.exe'), reverse=True))
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError('Cannot find toecollapse. Set TOUCHDESIGNER_BIN or pass --td-bin with your TouchDesigner bin folder.')


def publish_file(source, destination):
    # Copy to the destination volume before replacement, so a failed native
    # build or interrupted copy leaves the previous working artifact intact.
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + '.', suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def build_toe_and_tox(project_dir=None, collapse_exe=None):
    root = Path(project_dir or Path(__file__).resolve().parent).resolve()
    compiler = find_collapse_executable(collapse_exe)
    source = root / 'colab-sd.toe.dir'
    project = source / 'project1'
    component = project / 'StreamDiffusion_Remote'
    toc = root / 'colab-sd.toe.toc'
    for required in (component, toc, source / '.build'):
        if not required.exists():
            raise FileNotFoundError(f'Missing build source: {required}')
    callbacks = {
        'script_receiver_callbacks.text': SCRIPT_RECEIVER_CODE,
        'script_receiver_callbacks1.text': SCRIPT_RECEIVER_CODE,
        'websocket1_callbacks.text': WEBSOCKET_CODE,
        'websocket1_callbacks1.text': WEBSOCKET_CODE,
        'parexec_controls.text': PAREXEC_CODE,
        'exec_send_frame.text': EXEC_SEND_FRAME_CODE,
    }
    for name, code in callbacks.items():
        compile(code, name, 'exec')

    (project / 'StreamDiffusion_Remote.cparm').write_text(CPARM_CONTENT, encoding='utf-8', newline='\n')
    (project / 'StreamDiffusion_Remote.parm').write_text(PARM_CONTENT, encoding='utf-8', newline='\n')
    (component / 'in_audio.n').write_text(IN_AUDIO_N, encoding='utf-8', newline='\n')
    (component / 'in_audio.parm').write_text(IN_AUDIO_PARM, encoding='utf-8', newline='\n')
    controls = 'Serverurl Connect Mode Resolution Source Displaymode Prompta Promptb Promptblend Negprompt Submitprompt Strength Seed Randomseed Snapshot Snapdir Audiorx Audiochop Audstrength Audseedthresh'
    (component / 'parexec_controls.parm').write_text(
        '?\nfromop 201326609 "" parent()\nop 67108864 ..\n'
        f'pars 67108864 "{controls}"\nlanguage 67108864 python\n?\n', encoding='utf-8', newline='\n')
    (component / 'fit512.parm').write_text(
        '?\nfit 0 fitoutside\noutputresolution 0 custom\n'
        'resolutionw 17 512 int(str(parent().par.Resolution.eval()).split("x")[0])\n'
        'resolutionh 17 512 int(str(parent().par.Resolution.eval()).split("x")[-1])\n'
        'outputaspect 0 resolution\n?\n', encoding='utf-8', newline='\n')
    (component / 'fit_send.n').write_text(
        'TOP:fit\ntile 50 -400 130 90\nflags =  parlanguage 0\ninputs\n{\n0 \tnull_cam\n}\ncolor 0.67 0.67 0.67\nend\n', encoding='utf-8', newline='\n')
    (component / 'fit_send.parm').write_text(
        '?\nfit 0 fitoutside\noutputresolution 0 custom\n'
        'resolutionw 17 256 max(1,round(op("null_cam").width*256/max(op("null_cam").width,op("null_cam").height)))\n'
        'resolutionh 17 256 max(1,round(op("null_cam").height*256/max(op("null_cam").width,op("null_cam").height)))\n'
        'outputaspect 0 resolution\n?\n', encoding='utf-8', newline='\n')
    for name, code in callbacks.items():
        write_text_file(component / name, code)
    entries = [line.strip() for line in toc.read_text(encoding='utf-8').splitlines() if line.strip()]
    for name in ('in_audio.n', 'in_audio.parm', 'fit_send.n', 'fit_send.parm'):
        entry = 'project1/StreamDiffusion_Remote/' + name
        if entry not in entries:
            entries.append(entry)
    toc.write_text('\n'.join(entries) + '\n', encoding='utf-8', newline='\n')

    # Build both archives away from working outputs. No existing project or
    # reusable component is replaced unless both native builds succeed.
    with tempfile.TemporaryDirectory(prefix='colab-sd-build-') as staging_name:
        staging = Path(staging_name).resolve()
        assert staging.is_relative_to(Path(tempfile.gettempdir()).resolve())
        shutil.copytree(source, staging / source.name)
        shutil.copy2(toc, staging / toc.name)
        tox_source = staging / 'StreamDiffusion_Remote.tox.dir'
        shutil.copytree(component, tox_source)
        for extension in ('n', 'parm', 'cparm'):
            shutil.copy2(project / f'StreamDiffusion_Remote.{extension}', tox_source / f'.{extension}')
        # A reusable component must not reference a TOP in the source project's
        # parent network. Its in1 connector remains available for new wiring.
        root_node = tox_source / '.n'
        root_node.write_text(re.sub(r'(?ms)^inputs\s*\n\{.*?^\}\n', '', root_node.read_text(encoding='utf-8')),
                             encoding='utf-8', newline='\n')
        shutil.copy2(source / '.build', tox_source / '.build')
        tox_entries = ['# 4 0 0 0 1'] + sorted(path.name for path in tox_source.iterdir() if path.is_file())
        (staging / 'StreamDiffusion_Remote.tox.toc').write_text('\n'.join(tox_entries) + '\n', encoding='utf-8', newline='\n')
        names = ('colab-sd.toe', 'StreamDiffusion_Remote.tox')
        for name in names:
            result = subprocess.run([str(compiler), name], cwd=staging, capture_output=True, text=True)
            artifact = staging / name
            if result.returncode != 0 or not artifact.is_file() or artifact.stat().st_size < 128:
                raise RuntimeError(f'Build failed for {name}: {result.stderr or result.stdout or "no valid archive produced"}')
        for name in names:
            publish_file(staging / name, root / name)
            print(f'Built {name}: {(root / name).stat().st_size} bytes')
    return tuple(root / name for name in names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build the TouchDesigner project and standalone component.')
    parser.add_argument('--td-bin', help='TouchDesigner bin folder or toecollapse executable')
    arguments = parser.parse_args()
    build_toe_and_tox(collapse_exe=arguments.td_bin)
