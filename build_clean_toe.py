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
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            h, w = img.shape[:2]
            # Ensure Script TOP output resolution matches the incoming frame
            if scriptOp.par.resolutionw.eval() != w:
                scriptOp.par.resolutionw = w
            if scriptOp.par.resolutionh.eval() != h:
                scriptOp.par.resolutionh = h
                
            rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            rgba_flipped = np.ascontiguousarray(np.flipud(rgba))
            scriptOp.copyNumpyArray(rgba_flipped)
    except Exception as e:
        print('[StreamDiffusion] Error in onCook:', e)
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
    parent().store('can_send', True)
    print('[StreamDiffusion] WebSocket Connected to Colab!')
    send_control(force=True)

def onDisconnect(dat):
    p_status = getattr(parent().par, 'Status', None)
    if p_status is not None:
        p_status.val = 'Disconnected'
    p_fps = getattr(parent().par, 'Fps', None)
    if p_fps is not None:
        p_fps.val = '0.0 FPS'
    p_gpu = getattr(parent().par, 'Gputime', None)
    if p_gpu is not None:
        p_gpu.val = '-- ms'
    p_ping = getattr(parent().par, 'Ping', None)
    if p_ping is not None:
        p_ping.val = '-- ms'
    parent().store('can_send', True)
    print('[StreamDiffusion] WebSocket Disconnected.')
    global current_frame_id, frame_chunks
    current_frame_id = -1
    frame_chunks.clear()

def onReceiveText(dat, rowIndex, message):
    try:
        data = json.loads(message)
        if 'status' in data:
            p_status = getattr(parent().par, 'Status', None)
            if p_status is not None:
                p_status.val = data['status']
        if 'inf_ms' in data:
            p_gpu = getattr(parent().par, 'Gputime', None)
            if p_gpu is not None:
                p_gpu.val = f"{data['inf_ms']:.1f} ms"
    except Exception:
        pass

def onReceiveBinary(dat, contents):
    global current_frame_id, frame_chunks, total_received, last_frame_bytes
    global last_fps_time, fps_frame_count
    
    if not contents or len(contents) < 4:
        return
        
    # StreamDiffusion Binary Chunk Protocol:
    # byte 0: frame_id (0-255)
    # byte 1: chunk_idx (0..total_chunks-1)
    # byte 2: total_chunks (1-255)
    # bytes 3+: JPEG slice bytes
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
        
        # Adaptive Flow Control: unlock sending immediately upon frame receipt
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

def send_control(force=False):
    p = parent().par
    mode = 'img2img' if getattr(p, 'Mode', None) and p.Mode.eval() == 'img2img' else 'txt2img'
    prompt_val = str(p.Prompt.eval()) if getattr(p, 'Prompt', None) else ''
    strength_val = float(p.Strength.eval()) if getattr(p, 'Strength', None) else 0.6
    seed_val = int(p.Seed.eval()) if getattr(p, 'Seed', None) else 42
    msg = {
        'type': 'control',
        'mode': mode,
        'prompt': prompt_val,
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
            print(f"[StreamDiffusion] Sent control: mode={mode}, prompt='{prompt_val}', strength={strength_val:.2f}, seed={seed_val}")
        except Exception as e:
            print('[StreamDiffusion] Error sending control:', e)
    else:
        print(f"[StreamDiffusion] WebSocket not active, queued prompt: '{prompt_val}'")
'''

EXEC_SEND_FRAME_CODE = '''import cv2
import numpy as np
import time

# me - this DAT
# frame - current timeline frame number

frame_count = 0
frame_id_seq = 0

def onStart():
    return

def onCreate():
    return

def onExit():
    return

def onFrameStart(frame):
    global frame_count, frame_id_seq
    
    # Only active when in img2img mode
    mode_par = getattr(parent().par, 'Mode', None)
    if not mode_par or mode_par.eval() != 'img2img':
        return
        
    ws = op('websocket1')
    if not ws or not ws.par.active.eval():
        return

    # Adaptive Flow Control:
    # Only send when server is ready (replied to previous frame)
    # or after 250ms watchdog timeout to recover from any dropped packet
    now = time.time()
    can_send = parent().fetch('can_send', True)
    last_send_time = parent().fetch('last_send_time', 0.0)
    
    if not can_send and (now - last_send_time < 0.25):
        return

    # Throttle: Cap maximum transmission at 30 FPS (~33ms)
    if (now - last_send_time < 0.033):
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

PAREXEC_CODE = '''import random

# me - this DAT
# par - the parameter that changed
# prev - the previous value

def onValueChange(par, prev):
    try:
        if par.name in ['Prompt', 'Mode', 'Source', 'Strength', 'Seed']:
            for name in ['websocket1_callbacks', 'websocket1_callbacks1']:
                cb = op(name)
                if cb and hasattr(cb.module, 'send_control'):
                    cb.module.send_control(force=True)
                    break
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
    except Exception as e:
        print('[StreamDiffusion] parexec onPulse error:', e)
'''

CPARM_CONTENT = '''?
pages 1 StreamDiffusion
772804868 Serverurl "Colab WebSocket URL" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 0
772804867 Connect "Connect to Colab" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 1
772804879 Mode "Generation Mode" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 4097 2 txt2img "Text-to-Video" img2img "Image / Noise to Image" 2
772804879 Source "Input Source" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 4097 3 in1 "External Input (in1)" cam "Webcam (cam_in)" noise "Internal Noise" 3
772804868 Prompt Prompt 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 4
772804869 Sendprompt "Submit Prompt" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 5
772804865 Strength "Denoise Strength" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 6
772804866 Seed Seed 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 7
772804869 Randomseed "Randomize Seed" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 8
772804868 Status Status 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 9
772804868 Fps "Effective FPS" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 10
772804868 Gputime "Colab GPU Time" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 11
772804868 Ping "Network Ping (RTT)" 1 1 0 0 1 1 1 2 0 "" "" StreamDiffusion 12
?
'''

PARM_CONTENT = '''?
pageindex 0 3
Serverurl 67108928 wss://samples-null-chance-beverage.trycloudflare.com/ws/stream
Connect 67108928 off
Prompt 67108928 "cyberpunk cat, neon lights, 8k, detailed"
Strength 67108928 0.6
Seed 67108928 42
Status 67109184 Disconnected
Fps 67109184 "0.0 FPS"
Gputime 67109184 "-- ms"
Ping 67109184 "-- ms"
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
        
    # 2. Update parexec_controls.parm to monitor all control parameters
    parexec_parm = (
        "?\n"
        "fromop 201326609 \"\" parent()\n"
        "op 67108864 ..\n"
        "pars 67108864 \"Prompt Mode Source Strength Seed Randomseed Sendprompt Connect\"\n"
        "language 67108864 python\n"
        "?\n"
    )
    with open(os.path.join(base_dir, 'parexec_controls.parm'), 'w', newline='\n') as f:
        f.write(parexec_parm)

    # 3. Update text callbacks
    write_text_file(os.path.join(base_dir, 'script_receiver_callbacks.text'), SCRIPT_RECEIVER_CODE)
    write_text_file(os.path.join(base_dir, 'script_receiver_callbacks1.text'), SCRIPT_RECEIVER_CODE)
    write_text_file(os.path.join(base_dir, 'websocket1_callbacks.text'), WEBSOCKET_CODE)
    write_text_file(os.path.join(base_dir, 'websocket1_callbacks1.text'), WEBSOCKET_CODE)
    write_text_file(os.path.join(base_dir, 'parexec_controls.text'), PAREXEC_CODE)
    write_text_file(os.path.join(base_dir, 'exec_send_frame.text'), EXEC_SEND_FRAME_CODE)
    
    # 4. Re-collapse colab-sd.toe
    target_toe = 'colab-sd.toe'
    print(f"\nCollapsing {target_toe}...")
    res = subprocess.run([collapse_exe, target_toe], capture_output=True, text=True)
    if os.path.exists(target_toe):
        sz = os.path.getsize(target_toe)
        print(f"SUCCESS: {target_toe} generated successfully! Size: {sz} bytes.")
    else:
        print("ERROR: colab-sd.toe was not created.")

    # 5. Build standalone StreamDiffusion_Remote.tox
    tox_dir = r"StreamDiffusion_Remote.tox.dir"
    toc_path = r"StreamDiffusion_Remote.tox.toc"
    target_tox = r"StreamDiffusion_Remote.tox"
    
    if os.path.exists(tox_dir):
        shutil.rmtree(tox_dir)
    os.makedirs(tox_dir, exist_ok=True)
    
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
