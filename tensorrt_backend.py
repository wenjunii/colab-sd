"""Torch-TensorRT acceleration for current Colab Python/CUDA runtimes."""
from pathlib import Path
import logging
import re
import subprocess
import sys


def package_requirement(torch_version, python_version=None):
    python_version = python_version or sys.version_info[:2]
    if tuple(python_version) < (3, 11):
        raise RuntimeError('TensorRT acceleration needs Python 3.11 or newer. Use a current Colab runtime or disable TensorRT.')
    match = re.match(r'^(\d+)\.(\d+)\.', torch_version)
    if match is None or tuple(map(int, match.groups())) < (2, 11):
        raise RuntimeError('This TensorRT backend requires PyTorch 2.11 or newer. Disable TensorRT on older runtimes.')
    return f'torch-tensorrt=={match[1]}.{match[2]}.*'


def ensure_dependencies(root):
    """Install matching wheels and verify imports; never silently request FP16."""
    import torch
    requirement = package_requirement(torch.__version__)
    if not torch.cuda.is_available() or not torch.version.cuda:
        raise RuntimeError('TensorRT requires a CUDA GPU runtime.')
    probe = [sys.executable, '-c', 'import torch_tensorrt, tensorrt; print("Torch-TensorRT", torch_tensorrt.__version__, "TensorRT", tensorrt.__version__)']
    result = subprocess.run(probe, cwd=root, capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout.strip(), flush=True)
        return
    logs = Path(root) / 'bridge_logs'
    logs.mkdir(exist_ok=True)
    log_path = logs / 'tensorrt-setup.log'
    cuda_tag = 'cu' + torch.version.cuda.replace('.', '')
    command = [sys.executable, '-m', 'pip', 'install', requirement, f'torch=={torch.__version__}',
               '--extra-index-url', f'https://download.pytorch.org/whl/{cuda_tag}']
    print(f'Installing {requirement} for PyTorch {torch.__version__} / CUDA {torch.version.cuda}.', flush=True)
    print(f'Installation log: {log_path}', flush=True)
    with log_path.open('w', encoding='utf-8') as log:
        installed = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=900)
    if installed.returncode:
        print(log_path.read_text(encoding='utf-8', errors='replace')[-10000:], flush=True)
        raise RuntimeError('TensorRT installation failed. See bridge_logs/tensorrt-setup.log; uncheck TensorRT to use PyTorch.')
    result = subprocess.run(probe, cwd=root, capture_output=True, text=True)
    if result.returncode:
        print(result.stderr, flush=True)
        raise RuntimeError('TensorRT imports failed after installation. See the startup output; acceleration was not enabled.')
    print(result.stdout.strip(), flush=True)


def cache_directory(root, width, height, batch_size, torch_version, trt_version, capability):
    tag = re.sub(r'[^A-Za-z0-9._-]', '_', f'torch-{torch_version}-trt-{trt_version}-sm{capability[0]}{capability[1]}')
    return Path(root) / tag / f'{width}x{height}' / f'batch-{batch_size}'


class TensorRTUNet:
    """Adapt the compiled tensor output to StreamDiffusion's UNet call contract."""
    backend = 'TensorRT (UNet)'

    def __init__(self, compiled, source, batch_size):
        self.compiled = compiled
        self.config = source.config
        self.dtype = source.dtype
        self.device = source.device
        self.batch_size = batch_size

    def __call__(self, sample, timestep, encoder_hidden_states, **kwargs):
        import torch
        if sample.shape[0] != self.batch_size:
            raise ValueError('TensorRT UNet batch size does not match the compiled mode')
        result = self.compiled(sample, timestep.to(dtype=torch.float32), encoder_hidden_states)
        return (result,)

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self


def compile_unet(unet, width, height, batch_size, cache_root='engines/torch-tensorrt'):
    import torch
    import torch_tensorrt
    import tensorrt
    logging.getLogger('torch_tensorrt.dynamo.conversion').setLevel(logging.WARNING)
    logging.getLogger('torch_tensorrt.dynamo.conversion._conversion').setLevel(logging.INFO)
    package_requirement(torch.__version__)
    if not torch.cuda.is_available():
        raise RuntimeError('TensorRT compilation requires a CUDA GPU.')

    class UNetGraph(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, sample, timestep, encoder_hidden_states):
            return self.model(sample, timestep, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]

    graph = UNetGraph(unet).eval()
    inputs = [
        torch.randn((batch_size, unet.config.in_channels, height // 8, width // 8), device=unet.device, dtype=unet.dtype),
        torch.full((batch_size,), 999.0, device=unet.device, dtype=torch.float32),
        torch.randn((batch_size, 77, unet.config.cross_attention_dim), device=unet.device, dtype=unet.dtype),
    ]
    cache = cache_directory(cache_root, width, height, batch_size, torch.__version__,
                            f'{torch_tensorrt.__version__}-sdk-{tensorrt.__version__}',
                            torch.cuda.get_device_capability())
    cache.mkdir(parents=True, exist_ok=True)
    print(f'>>> Compiling TensorRT UNet: {width} x {height}, batch {batch_size}. First use can take several minutes.', flush=True)
    with torch.no_grad():
        expected = graph(*inputs).detach()
        # Export standard attention operations: FlashAttention cross-attention
        # lacks converter coverage, and its self-attention converter requires
        # strong typing that triggers a TensorRT 10.15 builder error here.
        # Compare the compiled result against the original attention processor.
        from diffusers.models.attention_processor import AttnProcessor
        unet.set_attn_processor({
            name: AttnProcessor() for name in unet.attn_processors
        })
        compiled = torch_tensorrt.compile(
            # This FP16 builder path avoids the TensorRT 10.15 strong-typing
            # compiler failure while retaining full TensorRT graph coverage.
            graph, ir='dynamo', inputs=inputs, use_explicit_typing=False,
            enabled_precisions={torch.float16}, optimization_level=0,
            # Decompose Diffusers' baddbmm into supported primitive operations.
            enable_experimental_decompositions=True,
            require_full_compilation=True, workspace_size=2 * 1024**3,
            # Torch-TensorRT only stores/reuses refittable engines in its cache.
            immutable_weights=False,
            cache_built_engines=True, reuse_cached_engines=True, engine_cache_dir=str(cache),
        )
        engine_count = sum(type(module).__name__ in ('TorchTensorRTModule', 'PythonTorchTensorRTModule')
                           for module in compiled.modules())
        if engine_count == 0:
            raise RuntimeError('Compilation produced no TensorRT engines; acceleration was not enabled.')
        actual = compiled(*inputs)
        torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.05)
        torch.cuda.synchronize()
    print(f'>>> TensorRT UNet batch {batch_size} compiled and output validation passed ({engine_count} engine(s)).', flush=True)
    return TensorRTUNet(compiled, unet, batch_size)
