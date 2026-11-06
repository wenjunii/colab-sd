"""Opt-in Krea GPU validation; reuses an existing isolated runtime/model cache."""
import argparse
import json
import logging
import os
from pathlib import Path
import time


def check_int8():
    import bitsandbytes as bnb
    import torch
    from krea_quant import prepare_int8_linears
    torch.manual_seed(7)
    report = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'bitsandbytes': bnb.__version__}
    layer = bnb.nn.Linear8bitLt(64, 32, has_fp16_weights=False, threshold=6.0).eval().cuda()
    value = torch.randn(1, 3, 16, 64, device='cuda', dtype=torch.bfloat16) * 2
    value[..., 0] = 20
    with torch.inference_mode():
        try:
            layer(value)
            report['unadapted_rank4'] = 'accepted'
        except RuntimeError as exc:
            report['unadapted_rank4'] = str(exc)
        assert prepare_int8_linears(layer) == 1
        for candidate in (value, value.transpose(1, 2), value.unsqueeze(0)):
            reference = layer(candidate.reshape(-1, candidate.shape[-1]))
            reference = reference.reshape(*candidate.shape[:-1], reference.shape[-1])
            actual = layer(candidate)
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            assert torch.isfinite(actual).all()
        assert prepare_int8_linears(layer) == 0
    report['adapted_rank4_rank5_noncontiguous'] = 'identical to explicit flattened INT8 reference'
    print('INT8_CHECK ' + json.dumps(report), flush=True)
    return report


def check_stream(root, output, report, timeout):
    import cv2
    import numpy as np
    import torch
    from fastapi.testclient import TestClient
    # Choose the actual server backend before importing its application.
    os.environ.update(BRIDGE_BACKEND='krea', ENABLE_TENSORRT='0', KREA_CLIP_BLOCKS='64',
                      HF_HUB_DISABLE_SYMLINKS='1', HF_HUB_DISABLE_SHARED_BLOBS='1')
    import colab_server as server
    assert server.is_krea_backend and Path(server.__file__).resolve().parent == root
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    print('Loading the existing Krea runtime/cache for real server/WebSocket validation...', flush=True)
    with TestClient(server.app) as client:
        health = client.get('/health').json()
        assert health['ready'] and health['backend'] == 'krea'
        report['load_seconds'] = round(time.perf_counter() - start, 3)
        print('MODEL_READY ' + json.dumps({'load_seconds': report['load_seconds'], 'engine': health['engine']}), flush=True)
        with client.websocket_connect('/ws/stream') as ws:
            welcome = ws.receive_json()
            assert welcome['type'] == 'welcome' and welcome['supported_modes'] == ['krea']
            ws.send_json({'type': 'client_options', 'frame_ack': True, 'frame_window': 3, 'video_quality': 'fast'})
            partial, frame_times, stages = {}, [], []
            deadline = time.monotonic() + timeout

            def receive():
                if time.monotonic() > deadline:
                    raise TimeoutError('Krea stream validation exceeded its frame deadline')
                message = ws.receive()
                if message.get('type') == 'websocket.close':
                    raise RuntimeError('Krea WebSocket closed during validation')
                if message.get('text') is not None:
                    data = json.loads(message['text'])
                    if data.get('type') == 'error' or data.get('world_phase') == 'error':
                        raise RuntimeError('Krea server reported generation failure: ' + json.dumps(data))
                    return data, None
                payload = message.get('bytes')
                if not payload or len(payload) < 4:
                    raise ValueError('Invalid video protocol packet')
                frame_id, index, total = payload[:3]
                assert 0 <= index < total
                count, chunks = partial.setdefault(frame_id, (total, {}))
                assert count == total
                chunks[index] = payload[3:]
                if len(chunks) != total:
                    return {}, None
                pixels = cv2.imdecode(np.frombuffer(b''.join(chunks[i] for i in range(total)), dtype=np.uint8), cv2.IMREAD_COLOR)
                assert pixels is not None and pixels.shape == (480, 832, 3)
                assert pixels.dtype == np.uint8 and pixels.std() > 2
                del partial[frame_id]
                ws.send_json({'type': 'frame_ack', 'frame_id': frame_id})
                frame_times.append(time.perf_counter())
                return {}, pixels

            def frames(count, label):
                began, last = time.perf_counter(), None
                for _ in range(count):
                    while True:
                        data, pixels = receive()
                        if pixels is not None:
                            last = pixels
                            break
                elapsed = time.perf_counter() - began
                stages.append({'stage': label, 'frames': count, 'seconds': round(elapsed, 3),
                               'last_batch_ms': round(server.krea_backend.batch_ms, 1)})
                assert cv2.imwrite(str(output / (label + '.jpg')), last)
                print('FRAMES_OK ' + json.dumps(stages[-1]), flush=True)
                return last

            ws.send_json({'type': 'control', 'mode': 'krea', 'width': 832, 'height': 480,
                          'prompt_a': 'A cinematic shot of a red toy car driving along a mountain road at sunrise',
                          'seed': 42, 'krea_paused': False})
            frames(9, 'first')
            ws.send_json({'type': 'control', 'prompt_a': 'A cinematic aerial shot of a green forest and a river'})
            frames(12, 'prompt-change')
            ws.send_json({'type': 'control', 'krea_paused': True})
            while True:
                data, _ = receive()
                if data.get('type') == 'configured' and data.get('krea_paused'):
                    break
            report['pause_confirmed'] = True
            ws.send_json({'type': 'control', 'krea_reset': True, 'krea_paused': False, 'seed': 43})
            while True:
                data, _ = receive()
                if data.get('type') == 'configured' and not data.get('krea_paused'):
                    break
            frames(9, 'reset')
            report.update(reset_confirmed=True, decoded_frames=len(frame_times), stages=stages,
                          peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 1024**3, 3),
                          peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 1024**3, 3),
                          frame_dimensions=[832, 480])
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('KREA_GPU_PASS ' + json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--int8-only', action='store_true', help='Small CUDA quantization check, without pretrained models')
    parser.add_argument('--output', type=Path, help='Keep validation JPEGs/report in this explicit output directory')
    parser.add_argument('--timeout', type=int, default=600, help='Frame validation deadline after model loading')
    arguments = parser.parse_args()
    if not arguments.int8_only and arguments.output is None:
        parser.error('--output is required for full model validation')
    # Avoid thousands of identical cast notices while keeping all other warnings.
    class CastOnce(logging.Filter):
        seen = False
        def filter(self, record):
            if 'inputs will be cast' in record.getMessage():
                previous, self.seen = self.seen, True
                return not previous
            return True
    logging.getLogger('bitsandbytes.autograd._functions').addFilter(CastOnce())
    report = check_int8()
    if not arguments.int8_only:
        output = arguments.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        check_stream(Path(__file__).resolve().parent, output, report, arguments.timeout)


if __name__ == '__main__':
    main()
