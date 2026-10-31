import os
import struct
import shutil
import subprocess

SCRIPT_RECEIVER_CODE = '''import cv2
import numpy as np

# me - this DAT
# scriptOp - the OP which is cooking

def onSetupParameters(scriptOp):
    return

def onPulse(par):
    return

def onCook(scriptOp):
    # Fetch latest reassembled frame from parent component storage
    data = parent().fetch('last_frame_bytes', None)
    if not data:
        for name in ['websocket1_callbacks', 'websocket1_callbacks1']:
            cb = op(name)
            if cb and hasattr(cb.module, 'last_frame_bytes') and cb.module.last_frame_bytes:
                data = cb.module.last_frame_bytes
                break
    
    if not data:
        return
        
    try:
        nparr = np.frombuffer(data, np.uint8)
        ai_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if ai_bgr is None:
            return
            
        h, w = ai_bgr.shape[:2]
        
        # Determine display mode: ai | sidebyside | wipe | input
        p_mode = getattr(parent().par, 'Displaymode', None)
        disp_mode = str(p_mode.eval()) if p_mode is not None else 'ai'
        
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
last_fps_time = time.time()
fps_frame_count = 0

def onConnect(dat):
    p_status = getattr(parent().par, 'Status', None)
    if p_status is not None:
        p_status.val = 'Connected (Streaming)'
    parent().store('in_flight', 0)
    parent().store('can_send', True)
    print('[StreamDiffusion] WebSocket Connected to Colab!')
    send_control(force=True)

def onDisconnect(dat):
    p_status = getattr(parent().par, 'Status', None)
    if p_status is not None:
        p_status.val = 'Disconnected'
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
    parent().store('in_flight', 0)
    parent().store('can_send', True)
    print('[StreamDiffusion] WebSocket Disconnected.')
    global current_frame_id, frame_chunks
    current_frame_id = -1
    frame_chunks.clear()

def onReceiveText(dat, rowIndex, message):
    try:
        data = json.loads(message)
        msg_type = data.get('type', '')
        
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
    
    if not contents or len(contents) < 4:
        return
        
    frame_id = contents[0]
    chunk_idx = contents[1]
    total_chunks = contents[2]
    chunk_data = contents[3:]
    
    if frame_id != current_frame_id:
        current_frame_id = frame_id
        frame_chunks = {}
        
    frame_chunks[chunk_idx] = chunk_data
    
    if len(frame_chunks) == total_chunks:
        assembled = b"".join(frame_chunks[i] for i in range(total_chunks))
        last_frame_bytes = assembled
        parent().store('last_frame_bytes', assembled)
        
        # Pipelined Flow Control: decrement in-flight frame counter
        in_flight = parent().fetch('in_flight', 0)
        parent().store('in_flight', max(0, in_flight - 1))
        parent().store('can_send', True)
        
        total_received += 1
        fps_frame_count += 1
        
        now = time.time()
        # Telemetry: Network Ping (Round-Trip Time)
        last_send_time = parent().fetch('last_send_time', 0.0)
        if last_send_time > 0:
            rtt = (now - last_send_time) * 1000.0
            p_ping = getattr(parent().par, 'Ping', None)
            if p_ping is not None:
                p_ping.val = f"{rtt:.1f} ms"
            
        # Telemetry: Effective FPS
        elapsed = now - last_fps_time
        if elapsed >= 1.0:
            current_fps = fps_frame_count / elapsed
            p_fps = getattr(parent().par, 'Fps', None)
            if p_fps is not None:
                p_fps.val = f"{current_fps:.1f} FPS"
            p_status = getattr(parent().par, 'Status', None)
            if p_status is not None:
                p_status.val = f"Connected ({current_fps:.1f} FPS)"
            last_fps_time = now
            fps_frame_count = 0
        
        receiver = op('script_receiver')
        if receiver:
            receiver.cook(force=True)
            
        frame_chunks.clear()

def send_control(force=False, strength_override=None):
    p = parent().par
    mode = 'img2img' if getattr(p, 'Mode', None) and p.Mode.eval() == 'img2img' else 'txt2img'
    
    steps_par = getattr(p, 'Steps', None)
    steps_val = int(steps_par.eval()) if steps_par is not None else 2
    
    prompt_a = str(getattr(p, 'Prompta', getattr(p, 'Prompt', '')))
    prompt_b = str(getattr(p, 'Promptb', ''))
    blend_val = float(getattr(p, 'Promptblend', 0.0))
    neg_prompt = str(getattr(p, 'Negprompt', ''))
    
    if strength_override is not None:
        strength_val = float(strength_override)
    else:
        strength_val = float(p.Strength.eval()) if getattr(p, 'Strength', None) else 0.6
        
    seed_val = int(p.Seed.eval()) if getattr(p, 'Seed', None) else 42
    
    msg = {
        'type': 'control',
        'mode': mode,
        'steps': steps_val,
        'prompt_a': prompt_a,
        'prompt_b': prompt_b,
        'blend': blend_val,
        'neg_prompt': neg_prompt,
        'strength': strength_val,
        'seed': seed_val,
        'force': force,
        'width': 512,
        'height': 512,
    }
    ws = op('websocket1')
    if ws and ws.par.active.eval():
        try:
            ws.sendText(json.dumps(msg))
            print(f"[StreamDiffusion] Sent control: mode={mode}, steps={steps_val}, A='{prompt_a[:15]}...', str={strength_val:.2f}, seed={seed_val}")
        except Exception as e:
            print('[StreamDiffusion] Error sending control:', e)
    else:
        print(f"[StreamDiffusion] WebSocket offline, queued parameters.")
'''

EXEC_SEND_FRAME_CODE = '''import cv2
import numpy as np
import time
import random

# me - this DAT
# frame - current timeline frame number

frame_count = 0
frame_id_seq = 0
last_audio_send_time = 0.0
last_audio_beat_time = 0.0

def onStart():
    return

def onCreate():
    return

def onExit():
    return

def handle_audio_reactivity(now):
    global last_audio_send_time, last_audio_beat_time
    p = parent().par
    p_aud = getattr(p, 'Audiorx', None)
    if not p_aud or not p_aud.eval():
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
        if amp >= thresh and (now - last_audio_beat_time > 0.35):
            last_audio_beat_time = now
            p_seed = getattr(p, 'Seed', None)
            if p_seed is not None:
                p_seed.val = random.randint(1, 999999)
                print(f"[StreamDiffusion] Audio Beat Detected (amp={amp:.2f} >= {thresh:.2f}) -> Random Seed {p_seed.val}")

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
    now = time.time()
    
    # Process audio reactivity across both txt2img and img2img modes
    handle_audio_reactivity(now)
    
    # Only active when in img2img mode
    mode_par = getattr(parent().par, 'Mode', None)
    if not mode_par or mode_par.eval() != 'img2img':
        return
        
    ws = op('websocket1')
    if not ws or not ws.par.active.eval():
        return

    # Double-Buffered / Pipelined Flow Control:
    # Allows 2 frames in-flight when set to 'double' to hide network transmission behind GPU compute
    flow_par = getattr(parent().par, 'Flowmode', None)
    is_double = (flow_par is None or flow_par.eval() == 'double')
    max_in_flight = 2 if is_double else 1

    in_flight = parent().fetch('in_flight', 0)
    last_send_time = parent().fetch('last_send_time', 0.0)

    # Watchdog: if nothing received for 300ms, reset in_flight counter to recover from drops
    if in_flight > 0 and (now - last_send_time > 0.30):
        in_flight = 0
        parent().store('in_flight', 0)
        parent().store('can_send', True)

    if in_flight >= max_in_flight:
        return

    # Throttle: Cap maximum transmission at ~45 FPS (~22ms)
    min_interval = 0.022 if is_double else 0.033
    if (now - last_send_time < min_interval):
        return

    cam = op('null_cam')
    if not cam:
        return
        
    try:
        cam.cook(force=True)
        arr = cam.numpyArray()
        if arr is not None:
            # Float32 texture (0.0..1.0) to uint8 (0..255)
            img_bgr = (np.clip(arr[:, :, :3], 0.0, 1.0) * 255).astype(np.uint8)
            img_bgr = cv2.flip(img_bgr, 0)
            
            # Downsample to 256x256 before JPEG compression
            # Payload stays tiny (~5KB-8KB) while preserving full visual structures
            img_small = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)
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
                
                parent().store('in_flight', in_flight + 1)
                parent().store('can_send', (in_flight + 1 < max_in_flight))
                parent().store('last_send_time', now)
                frame_count += 1
                if frame_count % 30 == 1:
                    print(f"[StreamDiffusion] Streaming input frame #{frame_count} ({len(data)} bytes in {total_chunks} chunk(s), in_flight={in_flight+1})")
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

# me - this DAT
# par - the parameter that changed
# prev - the previous value

def onValueChange(par, prev):
    try:
        if par.name in ['Prompta', 'Promptb', 'Promptblend', 'Negprompt', 'Prompt', 'Mode', 'Steps', 'Source', 'Strength', 'Seed']:
            for name in ['websocket1_callbacks', 'websocket1_callbacks1']:
                cb = op(name)
                if cb and hasattr(cb.module, 'send_control'):
                    cb.module.send_control(force=True)
                    break
        elif par.name == 'Displaymode':
            rec = op('script_receiver')
            if rec:
                rec.cook(force=True)
        elif par.name == 'Connect':
            if not par.eval():
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
        
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        base_fn = f"sd_capture_{ts}"
        img_path = os.path.join(dir_name, f"{base_fn}.png")
        meta_path = os.path.join(dir_name, f"{base_fn}.json")
        
        cv2.imwrite(img_path, img)
        
        p = parent().par
        meta = {
            'timestamp': datetime.datetime.now().isoformat(),
            'prompt_a': str(getattr(p, 'Prompta', getattr(p, 'Prompt', ''))),
            'prompt_b': str(getattr(p, 'Promptb', '')),
            'prompt_blend': float(getattr(p, 'Promptblend', 0.0)),
            'neg_prompt': str(getattr(p, 'Negprompt', '')),
            'steps': str(getattr(p, 'Steps', '2')),
            'flow_mode': str(getattr(p, 'Flowmode', 'double')),
            'strength': float(getattr(p, 'Strength', 0.6)),
            'seed': int(getattr(p, 'Seed', 42)),
            'mode': str(getattr(p, 'Mode', 'txt2img')),
            'fps': str(getattr(p, 'Fps', '')),
            'inference_time': str(getattr(p, 'Gputime', '')),
            'network_ping': str(getattr(p, 'Ping', '')),
            'engine': str(getattr(p, 'Engine', '')),
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

CPARM_CONTENT = '''?
pages 1 StreamDiffusion
772804868 Serverurl "Colab WebSocket URL" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 0
772804867 Connect "Connect to Colab" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 1
772804879 Mode "Generation Mode" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 4097 2 txt2img "Text-to-Video" img2img "Image / Noise to Image" 2
772804879 Steps "Diffusion Steps" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 4097 3 1 "1 Step (Ultra Turbo ~35-50 FPS)" 2 "2 Steps (Balanced ~25 FPS)" 4 "4 Steps (High Quality ~14 FPS)" 3
772804879 Flowmode "Flow Control" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 4097 2 double "Double Buffered (High FPS)" single "Single Token (Strict Latency)" 4
772804879 Source "Input Source" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 4097 3 in1 "External Input (in1)" cam "Webcam (cam_in)" noise "Internal Noise" 5
772804879 Displaymode "Display Mode" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 4097 4 ai "AI Output Only" sidebyside "Side-by-Side (Input | Output)" wipe "Split Wipe (50/50)" input "Input Only" 6
772804868 Prompta "Prompt A" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 7
772804868 Promptb "Prompt B" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 8
772804865 Promptblend "Prompt Blend (A <-> B)" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 9
772804868 Negprompt "Negative Prompt" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 10
772804869 Submitprompt "Submit Prompts" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 11
772804865 Strength "Denoise Strength" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 12
772804866 Seed Seed 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 13
772804869 Randomseed "Randomize Seed" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 14
772804869 Snapshot "Take Snapshot (PNG+JSON)" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 15
772804868 Snapdir "Snapshot Directory" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 16
772804867 Audiorx "Audio Reactivity" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 17
772804868 Audiochop "Audio CHOP Source" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 18
772804865 Audstrength "Audio -> Denoise Gain" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 19
772804865 Audseedthresh "Audio Beat Seed Threshold" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 20
772804868 Status Status 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 21
772804868 Engine "Acceleration Engine" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 22
772804868 Fps "Effective FPS" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 23
772804868 Gputime "Colab Inference" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 24
772804868 Ping "Network Ping (RTT)" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 25
772804868 Vram "Colab GPU VRAM" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 26
772804868 Gputemp "Colab GPU Temp" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 27
?
'''

PARM_CONTENT = '''?
pageindex 0 3
Serverurl 67108928 wss://samples-null-chance-beverage.trycloudflare.com/ws/stream
Connect 67108928 off
Mode 67108928 txt2img
Steps 67108928 2
Flowmode 67108928 double
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

def build_toe_and_tox():
    base_dir = r'colab-sd.toe.dir\project1\StreamDiffusion_Remote'
    proj_dir = r'colab-sd.toe.dir\project1'
    collapse_exe = r'C:\Program Files\Derivative\TouchDesigner\bin\toecollapse.exe'
    
    # 1. Update Custom Parameters (.cparm and .parm)
    with open(os.path.join(proj_dir, 'StreamDiffusion_Remote.cparm'), 'w', newline='\n') as f:
        f.write(CPARM_CONTENT)
    with open(os.path.join(proj_dir, 'StreamDiffusion_Remote.parm'), 'w', newline='\n') as f:
        f.write(PARM_CONTENT)
        
    # 2. Add in_audio CHOP node
    with open(os.path.join(base_dir, 'in_audio.n'), 'w', newline='\n') as f:
        f.write(IN_AUDIO_N)
    with open(os.path.join(base_dir, 'in_audio.parm'), 'w', newline='\n') as f:
        f.write(IN_AUDIO_PARM)

    # 3. Update parexec_controls.parm to monitor all control parameters
    parexec_parm = (
        "?\n"
        "fromop 201326609 \"\" parent()\n"
        "op 67108864 ..\n"
        "pars 67108864 \"Serverurl Connect Mode Steps Flowmode Source Displaymode Prompta Promptb Promptblend Negprompt Submitprompt Strength Seed Randomseed Snapshot Snapdir Audiorx Audiochop Audstrength Audseedthresh\"\n"
        "language 67108864 python\n"
        "?\n"
    )
    with open(os.path.join(base_dir, 'parexec_controls.parm'), 'w', newline='\n') as f:
        f.write(parexec_parm)

    # 4. Update text callbacks
    write_text_file(os.path.join(base_dir, 'script_receiver_callbacks.text'), SCRIPT_RECEIVER_CODE)
    write_text_file(os.path.join(base_dir, 'script_receiver_callbacks1.text'), SCRIPT_RECEIVER_CODE)
    write_text_file(os.path.join(base_dir, 'websocket1_callbacks.text'), WEBSOCKET_CODE)
    write_text_file(os.path.join(base_dir, 'websocket1_callbacks1.text'), WEBSOCKET_CODE)
    write_text_file(os.path.join(base_dir, 'parexec_controls.text'), PAREXEC_CODE)
    write_text_file(os.path.join(base_dir, 'exec_send_frame.text'), EXEC_SEND_FRAME_CODE)
    
    # 5. Update colab-sd.toe.toc to include in_audio files if not present
    toe_toc_path = 'colab-sd.toe.toc'
    with open(toe_toc_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    for item in ['project1/StreamDiffusion_Remote/in_audio.n', 'project1/StreamDiffusion_Remote/in_audio.parm']:
        if item not in lines:
            lines.append(item)
    with open(toe_toc_path, 'w', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')

    # 6. Re-collapse colab-sd.toe
    target_toe = 'colab-sd.toe'
    print(f"\nCollapsing {target_toe}...")
    res = subprocess.run([collapse_exe, target_toe], capture_output=True, text=True)
    if os.path.exists(target_toe):
        sz = os.path.getsize(target_toe)
        print(f"SUCCESS: {target_toe} generated successfully! Size: {sz} bytes.")
    else:
        print("ERROR: colab-sd.toe was not created.")

    # 7. Build standalone StreamDiffusion_Remote.tox
    tox_dir = r"StreamDiffusion_Remote.tox.dir"
    toc_path = r"StreamDiffusion_Remote.tox.toc"
    target_tox = r"StreamDiffusion_Remote.tox"
    
    os.makedirs(tox_dir, exist_ok=True)
    for fname in os.listdir(tox_dir):
        fp = os.path.join(tox_dir, fname)
        if os.path.isfile(fp):
            try:
                os.remove(fp)
            except Exception:
                pass
    
    for ext in ['.cparm', '.n', '.parm']:
        src = os.path.join(proj_dir, f"StreamDiffusion_Remote{ext}")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(tox_dir, f".{ext[1:]}"))
            
    toc_lines = ["# 4 0 0 0 1", ".build"]
    with open(os.path.join(tox_dir, ".build"), "w") as f:
        f.write("2025.32820\n")
        
    for item in os.listdir(base_dir):
        s = os.path.join(base_dir, item)
        d = os.path.join(tox_dir, item)
        if os.path.isfile(s):
            shutil.copy2(s, d)
            toc_lines.append(item)
            
    with open(toc_path, "w", newline="\n") as f:
        f.write("\n".join(toc_lines) + "\n")
        
    res_tox = subprocess.run([collapse_exe, target_tox], capture_output=True, text=True)
    if os.path.exists(target_tox):
        print(f"SUCCESS: {target_tox} generated successfully! Size: {os.path.getsize(target_tox)} bytes.")

if __name__ == '__main__':
    build_toe_and_tox()
