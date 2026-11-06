"""Opt-in native-resolution Krea comparison, reusing one loaded model."""
import argparse
import json
import logging
import math
import os
from pathlib import Path
import statistics
import time
from functools import wraps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--chunks', default=4, type=int)
    parser.add_argument('--thresholds', nargs='+', type=float, default=[6.0, 0.0, 6.0])
    parser.add_argument('--vae-formats', nargs='+', choices=['contiguous', 'channels_last_3d'],
                        help='One VAE convolution layout for each outlier threshold')
    args = parser.parse_args()
    if not 3 <= args.chunks <= 8:
        parser.error('--chunks must be between 3 and 8 (first two are warmup)')
    if any(not math.isfinite(x) or not 0 <= x <= 64 for x in args.thresholds):
        parser.error('--thresholds must be finite values between 0 and 64')
    formats = args.vae_formats or ['contiguous'] * len(args.thresholds)
    if len(formats) != len(args.thresholds):
        parser.error('--vae-formats must match the number of --thresholds')
    from krea_setup import require_unused_port
    require_unused_port()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Explicit output is retained; weights and reusable kernel caches are untouched.
    if (output / 'benchmark.json').exists():
        parser.error('Use a new output folder; an existing benchmark report is preserved')
    from krea_quant import configure_allocator, limit_cast_warnings
    configure_allocator(os.environ)
    import bitsandbytes as bnb
    import cv2
    import torch
    from krea_backend import KreaBackend
    # Match colab_server's inference policy when comparing kernel timings.
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    limit_cast_warnings()
    logging.basicConfig(level=logging.INFO)
    report = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'bitsandbytes': bnb.__version__, 'resolution': [832, 480],
              'steps': 4, 'seed': 42, 'warmup_chunks': 2, 'variants': [],
              'allocator': os.environ.get('PYTORCH_ALLOC_CONF', os.environ.get('PYTORCH_CUDA_ALLOC_CONF'))}
    start = time.perf_counter()
    print('BENCH_LOADING cached Krea model', flush=True)
    backend = KreaBackend.load(Path(__file__).resolve().parent, clip_blocks=64)
    report['load_seconds'] = round(time.perf_counter() - start, 3)
    layers = [layer for layer in backend.pipe.transformer.modules()
              if isinstance(layer, bnb.nn.Linear8bitLt)]
    assert layers, 'INT8 transformer layers were not found'
    events = []
    def timed(name, forward):
        @wraps(forward)
        def call(*values, **kwargs):
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin.record()
            result = forward(*values, **kwargs)
            end.record()
            events.append((name, begin, end))
            return result
        return call
    for component, method, label in ((backend.pipe.transformer, 'forward', 'transformer'),
            (backend.pipe.text_encoder, 'forward', 'text_encoder'),
            (backend.pipe.vae, 'encode', 'vae_encode'), (backend.pipe.vae, 'decode', 'vae_decode')):
        setattr(component, method, timed(label, getattr(component, method)))
    report['int8_layers'] = len(layers)
    print('BENCH_READY ' + json.dumps({k: v for k, v in report.items() if k != 'variants'}), flush=True)
    originals = [layer.state.threshold for layer in layers]
    try:
        for index, (threshold, layout) in enumerate(zip(args.thresholds, formats)):
            name = f'{index:02d}-threshold-{threshold:g}-{layout}'
            variant = {'name': name, 'threshold': threshold, 'vae_format': layout, 'chunks': []}
            report['variants'].append(variant)
            for layer in layers:
                layer.state.threshold = threshold
            backend.configure({'prompt_a': 'A cinematic shot of a red toy car driving along a mountain road at sunrise',
                               'seed': 42, 'krea_paused': False, 'krea_reset': True})
            torch.nn.utils.convert_conv3d_weight_memory_format(backend.pipe.vae,
                torch.channels_last_3d if layout == 'channels_last_3d' else torch.contiguous_format)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print('BENCH_START ' + name, flush=True)
            try:
                for chunk in range(args.chunks):
                    events.clear()
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    frame, _ = backend.next_frame()
                    count = 1 + len(backend.frames)
                    while backend.frames:
                        frame, _ = backend.next_frame()
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - begin
                    assert frame.shape == (480, 832, 3) and frame.std() > 2
                    assert cv2.imwrite(str(output / f'{name}-chunk-{chunk}.jpg'), frame)
                    item = {'chunk': chunk, 'frames': count, 'seconds': round(seconds, 4),
                            'fps': round(count / seconds, 3)}
                    item['gpu_ms'] = {label: round(sum(begin.elapsed_time(end) for name, begin, end in events
                                                       if name == label), 2)
                                      for label in {name for name, _, _ in events}}
                    variant['chunks'].append(item)
                    print('BENCH_CHUNK ' + json.dumps(dict(name=name, **item)), flush=True)
                steady = variant['chunks'][2:]
                variant.update(steady_fps=round(sum(x['frames'] for x in steady) / sum(x['seconds'] for x in steady), 3),
                               median_chunk_seconds=round(statistics.median(x['seconds'] for x in steady), 4),
                               peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 1024**3, 3),
                               peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 1024**3, 3))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                variant['error'] = str(exc)
            (output / 'benchmark.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
            print('BENCH_RESULT ' + json.dumps(variant), flush=True)
    finally:
        for layer, threshold in zip(layers, originals):
            layer.state.threshold = threshold
    print('BENCH_DONE ' + json.dumps(report), flush=True)
    if any('error' in variant for variant in report['variants']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
