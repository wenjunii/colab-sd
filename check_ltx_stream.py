"""Opt-in LTX delivery/prompt probe against an already running server.

Disconnect TouchDesigner first (the server has one controller). This does not
install packages, load another model, or start a listener. Output is retained.
"""
import argparse
import asyncio
import json
from pathlib import Path
import time


async def probe(url, output, clips=6, timeout=1200):
    import cv2
    import numpy as np
    import websockets
    report = {'url_kind': 'TLS tunnel' if url.startswith('wss://') else 'local WebSocket',
              'clips_per_prompt': clips, 'segments': []}
    prompts = ('A cinematic close-up of a red toy car driving on a mountain road in daylight',
               'A wide shot of a white sailboat on a blue tropical ocean in daylight')
    async with websockets.connect(url, max_size=2**20, ping_interval=20, open_timeout=20) as ws:
        welcome = json.loads(await asyncio.wait_for(ws.recv(), 30))
        if welcome.get('backend') != 'ltx' or welcome.get('type') != 'welcome':
            raise RuntimeError('Expected an idle LTX server; disconnect its other controller first')
        size = (welcome['width'], welcome['height'])
        frames = welcome['inference_batch_frames']
        report.update(size=list(size), frames_per_clip=frames, engine=welcome.get('engine'))
        await ws.send(json.dumps({'type': 'client_options', 'frame_ack': True,
                                  'frame_window': 1, 'video_quality': 'fast'}))
        chunks, current, total = {}, None, 0
        for segment, prompt in enumerate(prompts):
            started, times, progress = time.monotonic(), [], {}
            await ws.send(json.dumps({'type': 'control', 'mode': 'ltx', 'prompt_a': prompt,
                'ltx_continuous': True, 'ltx_continue_scene': True, 'ltx_refresh_every': 4,
                **({'seed': 42, 'ltx_generate': True} if segment == 0 else {})}))
            configured = False
            while len(times) < frames * clips:
                raw = await asyncio.wait_for(ws.recv(), timeout)
                if time.monotonic() - started > timeout:
                    raise TimeoutError('LTX stream probe exceeded its bounded segment deadline')
                if isinstance(raw, str):
                    data = json.loads(raw)
                    if data.get('type') == 'error' or data.get('world_phase') == 'error':
                        raise RuntimeError(data.get('status', 'Server rejected probe controls'))
                    if data.get('type') == 'configured':
                        configured = True
                    if data.get('type') in ('stats', 'configured'):
                        progress = {k: v for k, v in data.items() if k.startswith('ltx_')}
                    continue
                if len(raw) < 4:
                    raise ValueError('Invalid output chunk')
                frame_id, index, count = raw[:3]
                if not count or index >= count:
                    raise ValueError('Invalid output chunk index')
                if current != frame_id:
                    chunks, current, total = {}, frame_id, 0
                if index not in chunks:
                    total += len(raw) - 3
                chunks[index] = raw[3:]
                if total > 2**20:
                    raise ValueError('Probe frame exceeded 1 MiB')
                if len(chunks) != count:
                    continue
                image = cv2.imdecode(np.frombuffer(b''.join(chunks[n] for n in range(count)), np.uint8),
                                     cv2.IMREAD_COLOR)
                if image is None or (image.shape[1], image.shape[0]) != size:
                    raise ValueError('Expected native output resolution')
                await ws.send(json.dumps({'type': 'frame_ack', 'frame_id': frame_id}))
                chunks, current, total = {}, None, 0
                if not configured:
                    continue  # at most an already-sent old frame during a prompt switch
                times.append(time.monotonic())
                if len(times) == 1 or len(times) % frames == 0:
                    cv2.imwrite(str(output / f'prompt-{segment}-frame-{len(times)}.jpg'), image)
                if len(times) % frames == 0:
                    print('LTX_STREAM_CLIP ' + json.dumps({'segment': segment, 'frames': len(times),
                          'elapsed_seconds': round(times[-1] - started, 3), **progress}), flush=True)
            intervals = np.diff(times)
            boundaries = [float(intervals[n - 1]) for n in range(frames, len(times), frames)]
            item = {'prompt': prompt, 'frames': len(times), 'first_frame_seconds': round(times[0] - started, 3),
                'delivery_fps': round((len(times) - 1) / (times[-1] - times[0]), 3),
                'interval_median_seconds': round(float(np.median(intervals)), 4),
                'interval_max_seconds': round(float(max(intervals)), 4),
                'clip_boundary_intervals_seconds': [round(v, 4) for v in boundaries], 'last_stats': progress}
            report['segments'].append(item)
            (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        await ws.send(json.dumps({'type': 'control', 'ltx_paused': True, 'ltx_continuous': False}))
    print('LTX_STREAM_PASS ' + json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--url', default='ws://127.0.0.1:8000/ws/stream',
                        help='Default is localhost; pass the printed wss URL to measure tunnel delivery')
    parser.add_argument('--clips', type=int, choices=(3, 4, 6, 8), default=6)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'report.json').exists():
        parser.error('Choose a new output folder; an existing report is preserved')
    asyncio.run(probe(args.url, output, args.clips))


if __name__ == '__main__':
    main()
