"""Opt-in LTX native clip benchmark in the existing isolated CUDA environment."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--frames', type=int, choices=(17, 33, 65, 121), default=17)
    parser.add_argument('--size', choices=('512x512', '768x512', '1024x576', '1024x768'), default='512x512')
    parser.add_argument('--image', default='')
    parser.add_argument('--profile', choices=('resident', 'low_memory'), default='resident')
    parser.add_argument('--continuous', action='store_true', help='Validate automatic seed advancement and scene continuation')
    args = parser.parse_args()
    from krea_setup import require_unused_port
    from krea_quant import configure_allocator
    require_unused_port()
    configure_allocator(os.environ)
    width, height = map(int, args.size.split('x'))
    os.environ.update(LTX_CLIP_FRAMES=str(args.frames), LTX_INPUT_IMAGE=args.image,
                      LTX_PERFORMANCE_PROFILE=args.profile, LTX_CONTINUOUS='0',
                      BRIDGE_INITIAL_WIDTH=str(width), BRIDGE_INITIAL_HEIGHT=str(height),
                      HF_HUB_DISABLE_SYMLINKS='1', HF_HUB_DISABLE_SHARED_BLOBS='1')
    import cv2
    import torch
    from ltx_backend import LTXBackend
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'report.json').exists():
        parser.error('Choose a new output folder; an existing report is preserved')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    started = time.perf_counter()
    print('LTX_LOADING existing runtime and model cache', flush=True)
    backend = LTXBackend.load(Path(__file__).resolve().parent)
    report = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'engine': backend.engine,
              'size': [width, height], 'frames_per_clip': args.frames, 'image_conditioning': bool(args.image),
              'continuous': args.continuous, 'scene_continuation': backend.continue_scene,
              'construction_seconds': round(time.perf_counter() - started, 3), 'clips': []}
    print('LTX_READY ' + json.dumps(report), flush=True)
    backend.configure({'ltx_continuous': args.continuous})
    for seed in (42, 43):
        if seed == 42 or not args.continuous:
            backend.configure({'prompt_a': 'A cinematic shot of a red toy car driving along a mountain road at sunrise',
                               'seed': seed, 'ltx_generate': True})
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        first, _ = backend.next_frame()
        first_seconds = time.perf_counter() - start
        count = 1
        last = first
        while count < args.frames:
            last, _ = backend.next_frame()
            count += 1
        assert count == args.frames and first.std() > 2 and last.std() > 2
        assert backend.last_seed == seed
        assert cv2.imwrite(str(output / f'clip-{seed}-first.jpg'), first)
        assert cv2.imwrite(str(output / f'clip-{seed}-last.jpg'), last)
        item = {'seed': seed, 'decoded_frames': count, 'first_frame_seconds': round(first_seconds, 3),
                'generation_seconds': round(backend.batch_ms / 1000, 3),
                'generated_fps': round(count * 1000 / backend.batch_ms, 3),
                'profile': backend.profile, 'actual_seed': backend.last_seed,
                'prompt_cache_hit': backend.progress()['ltx_prompt_cache_hit'],
                'prompt_seconds': backend.progress()['ltx_prompt_ms'] / 1000,
                'first_last_mean_difference': round(float(cv2.norm(first, last, cv2.NORM_L1) / first.size), 3),
                'peak_allocated_gib': round(torch.cuda.max_memory_allocated() / 1024**3, 3),
                'peak_reserved_gib': round(torch.cuda.max_memory_reserved() / 1024**3, 3)}
        report['clips'].append(item)
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print('LTX_CLIP_PASS ' + json.dumps(item), flush=True)
    backend.close()
    print('LTX_GPU_PASS ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
