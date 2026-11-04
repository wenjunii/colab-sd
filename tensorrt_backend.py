"""Torch-TensorRT acceleration for current Colab Python/CUDA runtimes."""
from pathlib import Path
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import weakref

_weight_hashes = weakref.WeakKeyDictionary()
ENGINE_CACHE_SCHEMA = 1


def weight_fingerprint(unet):
    signature = tuple((name, value.data_ptr(), value._version, str(value.dtype), tuple(value.shape))
                      for name, value in list(unet.named_parameters()) + list(unet.named_buffers()))
    previous = _weight_hashes.get(unet)
    if previous is not None and previous[0] == signature:
        return previous[1]
    digest = hashlib.sha256()
    for name, value in sorted(unet.state_dict().items()):
        digest.update(f'{name}:{value.dtype}:{tuple(value.shape)}'.encode())
        import torch
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    result = digest.hexdigest()
    _weight_hashes[unet] = (signature, result)
    return result


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def engine_metadata(unet, width, height, batch_size, torch_version, cuda_version,
                    torchtrt_version, trt_version, gpu_name, capability):
    try:
        diffusers_version = version('diffusers')
    except PackageNotFoundError:
        diffusers_version = 'unavailable'
    config = dict(unet.config) if hasattr(unet.config, 'items') else vars(unet.config)
    config = {key: value for key, value in config.items() if not key.startswith('_')}
    return dict(schema=ENGINE_CACHE_SCHEMA, weights=weight_fingerprint(unet), width=width, height=height,
                batch=batch_size, dtype=str(unet.dtype), torch=torch_version, cuda=cuda_version,
                torchtrt=torchtrt_version, tensorrt=trt_version, gpu=gpu_name, sm=list(capability),
                platform=platform.system(), machine=platform.machine(), python=platform.python_version(),
                diffusers=diffusers_version, config=hashlib.sha256(json.dumps(
                    config, sort_keys=True, default=str).encode()).hexdigest(),
                compiler='immutable-fp16-standard-attention-opt0')


def alternate_inputs(inputs):
    values = [inputs[0] * 0.5, inputs[1] * 0.5, inputs[2] * 0.5]
    if len(inputs) == 5:
        values.extend([inputs[3] * 0.5, inputs[4]])
    return values


def load_engine_archive(directory, metadata, inputs, expected, reference=None):
    """Load a full-weight immutable archive; never refit cached engines."""
    import torch
    archive, manifest = directory / 'unet.ts', directory / 'manifest.json'
    if not archive.exists() or not manifest.exists():
        return None
    try:
        record = json.loads(manifest.read_text(encoding='utf-8'))
        if record['metadata'] != metadata:
            return None
        # Stage a single file on runtime disk so Drive is read only once.
        with tempfile.TemporaryDirectory(prefix='bridge-trt-load-') as name:
            local = Path(name) / 'unet.ts'
            shutil.copyfile(archive, local)
            if file_digest(local) != record['sha256']:
                raise ValueError('TensorRT archive checksum mismatch')
            module = torch.jit.load(str(local)).eval()
        if 'tensorrt::execute_engine' not in str(module.inlined_graph):
            raise ValueError('Archive does not contain a TensorRT execution graph')
        torch.testing.assert_close(module(*inputs), expected, rtol=0.05, atol=0.05)
        if reference is not None:
            alternate = alternate_inputs(inputs)
            torch.testing.assert_close(module(*alternate), reference(*alternate), rtol=0.05, atol=0.05)
        torch.cuda.synchronize()
        print(f'>>> Loaded and validated full-weight TensorRT archive: {directory}', flush=True)
        return module
    except Exception as error:
        logging.getLogger(__name__).warning('TensorRT archive rejected; rebuilding: %s', error)
        return None


def save_engine_archive(module, directory, metadata, inputs, expected, reference=None):
    import torch
    import torch_tensorrt
    try:
        with tempfile.TemporaryDirectory(prefix='bridge-trt-save-') as name:
            local = Path(name) / 'unet.ts'
            torch_tensorrt.save(module, str(local), output_format='torchscript', inputs=inputs)
            # A successful export alone does not establish a usable cache.
            loaded = torch.jit.load(str(local)).eval()
            # FX can retain unused native parameters after full conversion.
            # Freeze removes those duplicate weights from the deployment file.
            try:
                compact = torch.jit.freeze(loaded)
                compact_path = Path(name) / 'compact.ts'
                torch.jit.save(compact, str(compact_path))
                compact_path.replace(local)
                loaded = compact
            except RuntimeError as error:
                logging.getLogger(__name__).warning('Archive compaction unavailable; keeping full export: %s', error)
            if 'tensorrt::execute_engine' not in str(loaded.inlined_graph):
                raise ValueError('Saved graph has no TensorRT execution')
            torch.testing.assert_close(loaded(*inputs), expected, rtol=0.05, atol=0.05)
            if reference is not None:
                alternate = alternate_inputs(inputs)
                torch.testing.assert_close(loaded(*alternate), reference(*alternate), rtol=0.05, atol=0.05)
            torch.cuda.synchronize()
            record = dict(metadata=metadata, sha256=file_digest(local))
            temporary = directory / ('unet-' + os.urandom(8).hex() + '.tmp')
            try:
                shutil.copyfile(local, temporary)
                temporary.replace(directory / 'unet.ts')
            finally:
                temporary.unlink(missing_ok=True)
            manifest = directory / ('manifest-' + os.urandom(8).hex() + '.tmp')
            try:
                manifest.write_text(json.dumps(record, sort_keys=True), encoding='utf-8')
                manifest.replace(directory / 'manifest.json')
            finally:
                manifest.unlink(missing_ok=True)
        print(f'>>> Saved validated full-weight TensorRT archive: {directory}', flush=True)
    except Exception as error:
        logging.getLogger(__name__).warning('This engine works, but could not be cached; next launch will rebuild: %s', error)


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
        inputs = [sample, timestep.to(dtype=torch.float32), encoder_hidden_states]
        if getattr(self.config, 'addition_embed_type', None) == 'text_time':
            conditioning = kwargs.get('added_cond_kwargs') or {}
            if any(key not in conditioning for key in ('text_embeds', 'time_ids')):
                raise ValueError('SDXL TensorRT requires live pooled text and image-size conditioning')
            inputs.extend([conditioning['text_embeds'], conditioning['time_ids']])
        result = self.compiled(*inputs)
        return (result,)

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self


def compile_unet(unet, width, height, batch_size, cache_root=None):
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

    class XLUNetGraph(UNetGraph):
        def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
            return self.model(sample, timestep, encoder_hidden_states=encoder_hidden_states,
                              added_cond_kwargs={'text_embeds': text_embeds, 'time_ids': time_ids},
                              return_dict=False)[0]

    is_xl = getattr(unet.config, 'addition_embed_type', None) == 'text_time'
    graph = (XLUNetGraph(unet) if is_xl else UNetGraph(unet)).eval()
    reuse_engines = os.environ.get('BRIDGE_TRT_ENGINE_REUSE', '1') == '1'
    cache_root = cache_root or os.environ.get('BRIDGE_TRT_CACHE_DIR', 'engines/torch-tensorrt')
    inputs = [
        torch.randn((batch_size, unet.config.in_channels, height // 8, width // 8), device=unet.device, dtype=unet.dtype),
        torch.full((batch_size,), 999.0, device=unet.device, dtype=torch.float32),
        torch.randn((batch_size, 77, unet.config.cross_attention_dim), device=unet.device, dtype=unet.dtype),
    ]
    if is_xl:
        pooled_dim = unet.config.projection_class_embeddings_input_dim - 6 * unet.config.addition_time_embed_dim
        if pooled_dim <= 0:
            raise ValueError('Invalid SDXL pooled text dimension')
        inputs.extend([
            torch.randn((batch_size, pooled_dim), device=unet.device, dtype=unet.dtype),
            torch.tensor([[height, width, 0, 0, height, width]], device=unet.device,
                         dtype=unet.dtype).repeat(batch_size, 1),
        ])
    cache = cache_directory(cache_root, width, height, batch_size, torch.__version__,
                            f'{torch_tensorrt.__version__}-sdk-{tensorrt.__version__}',
                            torch.cuda.get_device_capability())
    metadata = engine_metadata(unet, width, height, batch_size, torch.__version__, torch.version.cuda,
                               torch_tensorrt.__version__, tensorrt.__version__,
                               torch.cuda.get_device_name(), torch.cuda.get_device_capability())
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:24]
    cache = cache / key
    cache.mkdir(parents=True, exist_ok=True)
    print(f'>>> Checking TensorRT profile: {width} x {height}, batch {batch_size}.', flush=True)
    with torch.no_grad():
        expected = graph(*inputs).detach()
        if reuse_engines:
            loaded = load_engine_archive(cache, metadata, inputs, expected, graph)
            if loaded is not None:
                return TensorRTUNet(loaded, unet, batch_size)
        print(f'>>> Compiling TensorRT UNet: {width} x {height}, batch {batch_size}. First use can take several minutes.', flush=True)
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
            use_python_runtime=False,
            require_full_compilation=True, workspace_size=2 * 1024**3,
            # TensorRT 10.15 refit returned NaNs after a real cache hit on
            # SD-Turbo. Keep immutable weights in full deployment archives;
            # the compiler's stripped/refit cache must remain disabled.
            immutable_weights=True,
            cache_built_engines=False, reuse_cached_engines=False,
            timing_cache_path=str(cache / 'timing-cache.bin'),
        )
        engine_count = sum(type(module).__name__ in ('TorchTensorRTModule', 'PythonTorchTensorRTModule')
                           for module in compiled.modules())
        if engine_count == 0:
            raise RuntimeError('Compilation produced no TensorRT engines; acceleration was not enabled.')
        actual = compiled(*inputs)
        torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.05)
        torch.cuda.synchronize()
        # Check another timestep/input before exporting or reporting readiness.
        # Prompt conditioning stays a graph input, never frozen in a cache.
        check_inputs = alternate_inputs(inputs)
        torch.testing.assert_close(compiled(*check_inputs), graph(*check_inputs), rtol=0.05, atol=0.05)
        if reuse_engines:
            save_engine_archive(compiled, cache, metadata, inputs, expected, graph)
    print(f'>>> TensorRT UNet batch {batch_size} compiled and output validation passed ({engine_count} engine(s)).', flush=True)
    return TensorRTUNet(compiled, unet, batch_size)
