import os
import struct
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
import cv2
import numpy as np

# me - this DAT
# dat - the OP which is cooking

current_frame_id = -1
frame_chunks = {}
total_received = 0
last_frame_bytes = None

def onConnect(dat):
    parent().par.Status = 'Connected (Streaming)'
    print('[StreamDiffusion] WebSocket Connected to Colab!')
    send_control(force=True)

def onDisconnect(dat):
    parent().par.Status = 'Disconnected'
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
        total_received += 1
        
        receiver = op('script_receiver')
        if receiver:
            receiver.cook(force=True)
            
        frame_chunks.clear()

def send_control(force=False):
    p = parent().par
    mode = 'v2v' if p.Mode.eval() == 'img2img' else 'txt2img'
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

PAREXEC_CODE = '''import random

# me - this DAT
# par - the parameter that changed
# prev - the previous value

def onValueChange(par, prev):
    if par.name in ['Prompt', 'Mode', 'Seed']:
        for name in ['websocket1_callbacks', 'websocket1_callbacks1']:
            cb = op(name)
            if cb and hasattr(cb.module, 'send_control'):
                cb.module.send_control()
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

TIMER_CODE = '''import numpy as np
import cv2

# me - this DAT
# timerOp - the OP which is cooking

def onCycle(timerOp, segment, cycle):
    if parent().par.Mode.eval() != 'img2img':
        return
    ws = op('websocket1')
    if not ws or not ws.par.active or parent().par.Status == 'Disconnected':
        return
    cam = op('null_cam')
    if cam:
        try:
            arr = cam.numpyArray(delayed=True)
            if arr is not None:
                img_bgr = (arr[:, :, :3] * 255).astype(np.uint8)
                img_bgr = cv2.flip(img_bgr, 0)
                _, buffer = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                ws.sendBinary(buffer.tobytes())
        except Exception as e:
            pass
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

def main():
    base_dir = r'colab-sd.toe.dir\project1\StreamDiffusion_Remote'
    
    # 1. Update text callbacks
    write_text_file(os.path.join(base_dir, 'script_receiver_callbacks.text'), SCRIPT_RECEIVER_CODE)
    write_text_file(os.path.join(base_dir, 'script_receiver_callbacks1.text'), SCRIPT_RECEIVER_CODE)
    write_text_file(os.path.join(base_dir, 'websocket1_callbacks.text'), WEBSOCKET_CODE)
    write_text_file(os.path.join(base_dir, 'websocket1_callbacks1.text'), WEBSOCKET_CODE)
    write_text_file(os.path.join(base_dir, 'parexec_controls.text'), PAREXEC_CODE)
    write_text_file(os.path.join(base_dir, 'timer_callbacks.text'), TIMER_CODE)
    write_text_file(os.path.join(base_dir, 'timer_v2v_callbacks.text'), TIMER_CODE)
    
    # 2. Update parexec_controls.parm
    parexec_parm = (
        "?\n"
        "fromop 201326609 \"\" parent()\n"
        "op 67108864 ..\n"
        "pars 67108864 \"Prompt Mode Seed Randomseed Sendprompt Connect\"\n"
        "language 67108864 python\n"
        "?\n"
    )
    with open(os.path.join(base_dir, 'parexec_controls.parm'), 'w', newline='\n') as f:
        f.write(parexec_parm)
    print("Updated parexec_controls.parm")

    # 3. Update script_receiver.parm
    script_receiver_parm = (
        "?\n"
        "outputresolution 0 custom\n"
        "resolutionw 0 512\n"
        "resolutionh 0 512\n"
        "outputaspect 0 resolution\n"
        "callbacks 0 script_receiver_callbacks\n"
        "?\n"
    )
    with open(os.path.join(base_dir, 'script_receiver.parm'), 'w', newline='\n') as f:
        f.write(script_receiver_parm)
    print("Updated script_receiver.parm")

    # 4. Update websocket1.parm
    websocket1_parm = (
        "?\n"
        "active 17 on parent().par.Connect\n"
        "netaddress 17 \"\" parent().par.Serverurl\n"
        "port 0 443\n"
        "callbacks 0 websocket1_callbacks\n"
        "language 0 text\n"
        "?\n"
    )
    with open(os.path.join(base_dir, 'websocket1.parm'), 'w', newline='\n') as f:
        f.write(websocket1_parm)
    print("Updated websocket1.parm")

    # 5. Run toecollapse.exe
    collapse_exe = r'C:\Program Files\Derivative\TouchDesigner\bin\toecollapse.exe'
    target_toe = 'colab-sd.toe'
    print(f"\nCollapsing {target_toe}...")
    res = subprocess.run([collapse_exe, target_toe], capture_output=True, text=True)
    print("toecollapse exit code:", res.returncode)
    if res.stdout:
        print("stdout:", res.stdout)
    if res.stderr:
        print("stderr:", res.stderr)
        
    if os.path.exists(target_toe):
        sz = os.path.getsize(target_toe)
        print(f"SUCCESS: {target_toe} generated successfully! Size: {sz} bytes.")
    else:
        print("ERROR: colab-sd.toe was not created.")

if __name__ == '__main__':
    main()
