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
    parent().par.Status = 'Connected (Streaming)'
    parent().store('ready_to_send', True)
    print('[StreamDiffusion] WebSocket Connected to Colab!')
    send_control(force=True)

def onDisconnect(dat):
    parent().par.Status = 'Disconnected'
    parent().store('ready_to_send', True)
    print('[StreamDiffusion] WebSocket Disconnected.')
    global current_frame_id, frame_chunks
    current_frame_id = -1
    frame_chunks.clear()

def onReceiveText(dat, rowIndex, message):
    try:
        data = json.loads(message)
        if 'status' in data:
            parent().par.Status = data['status']
    except:
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
        
        # Adaptive Flow Control: signal ready to send next input frame
        parent().store('ready_to_send', True)
        total_received += 1
        fps_frame_count += 1
        
        now = time.time()
        elapsed = now - last_fps_time
        if elapsed >= 1.0:
            current_fps = fps_frame_count / elapsed
            parent().par.Status = f"Connected ({current_fps:.1f} FPS)"
            last_fps_time = now
            fps_frame_count = 0
        
        receiver = op('script_receiver')
        if receiver:
            receiver.cook(force=True)
            
        frame_chunks.clear()

def send_control(force=False):
    p = parent().par
    mode = 'img2img' if p.Mode.eval() == 'img2img' else 'txt2img'
    prompt_val = str(p.Prompt.eval() or '')
    msg = {
        'type': 'control',
        'mode': mode,
        'prompt': prompt_val,
        'seed': int(p.Seed.eval()),
        'force': force,
        'width': 512,
        'height': 512,
    }
    ws = op('websocket1')
    if ws and ws.par.active.eval():
        try:
            ws.sendText(json.dumps(msg))
            print(f"[StreamDiffusion] Sent control: mode={mode}, prompt='{prompt_val}'")
        except Exception as e:
            print('[StreamDiffusion] Error sending control:', e)
    else:
        print(f"[StreamDiffusion] WebSocket not active, queued prompt: '{prompt_val}'")
'''

EXEC_SEND_FRAME_CODE = '''import cv2
import numpy as np

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
    if parent().par.Mode.eval() != 'img2img':
        return
        
    ws = op('websocket1')
    if not ws or not ws.par.active.eval():
        return

    # Adaptive Flow Control:
    # Only send when the server is ready (replied to previous frame)
    # or after 12 frames timeout (~200ms) as a watchdog safety
    ready = parent().fetch('ready_to_send', True)
    last_send_frame = parent().fetch('last_send_frame', 0)
    
    if not ready and (frame - last_send_frame < 12):
        return

    # Cap maximum transmission at 30 FPS (every 2nd frame minimum)
    if frame - last_send_frame < 2:
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
                parent().store('ready_to_send', False)
                parent().store('last_send_frame', frame)
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
    if par.name in ['Prompt', 'Mode', 'Source', 'Seed']:
        for name in ['websocket1_callbacks', 'websocket1_callbacks1']:
            cb = op(name)
            if cb and hasattr(cb.module, 'send_control'):
                cb.module.send_control(force=True)
                break
    elif par.name == 'Connect':
        if not par.eval():
            parent().par.Status = 'Disconnected'

def onPulse(par):
    if par.name == 'Randomseed':
        parent().par.Seed = random.randint(1, 999999)
    elif par.name == 'Sendprompt':
        for name in ['websocket1_callbacks', 'websocket1_callbacks1']:
            cb = op(name)
            if cb and hasattr(cb.module, 'send_control'):
                cb.module.send_control(force=True)
                break
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
    collapse_exe = r'C:\Program Files\Derivative\TouchDesigner\bin\toecollapse.exe'
    
    # 1. Update text callbacks
    write_text_file(os.path.join(base_dir, 'script_receiver_callbacks.text'), SCRIPT_RECEIVER_CODE)
    write_text_file(os.path.join(base_dir, 'script_receiver_callbacks1.text'), SCRIPT_RECEIVER_CODE)
    write_text_file(os.path.join(base_dir, 'websocket1_callbacks.text'), WEBSOCKET_CODE)
    write_text_file(os.path.join(base_dir, 'websocket1_callbacks1.text'), WEBSOCKET_CODE)
    write_text_file(os.path.join(base_dir, 'parexec_controls.text'), PAREXEC_CODE)
    write_text_file(os.path.join(base_dir, 'exec_send_frame.text'), EXEC_SEND_FRAME_CODE)
    
    # 2. Re-collapse colab-sd.toe
    target_toe = 'colab-sd.toe'
    print(f"\nCollapsing {target_toe}...")
    res = subprocess.run([collapse_exe, target_toe], capture_output=True, text=True)
    if os.path.exists(target_toe):
        sz = os.path.getsize(target_toe)
        print(f"SUCCESS: {target_toe} generated successfully! Size: {sz} bytes.")
    else:
        print("ERROR: colab-sd.toe was not created.")

    # 3. Build standalone StreamDiffusion_Remote.tox
    tox_dir = r"StreamDiffusion_Remote.tox.dir"
    toc_path = r"StreamDiffusion_Remote.tox.toc"
    target_tox = r"StreamDiffusion_Remote.tox"
    
    if os.path.exists(tox_dir):
        shutil.rmtree(tox_dir)
    os.makedirs(tox_dir, exist_ok=True)
    
    for ext in ['.cparm', '.n', '.parm']:
        src = os.path.join(r"colab-sd.toe.dir\project1", f"StreamDiffusion_Remote{ext}")
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
