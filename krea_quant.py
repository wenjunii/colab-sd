"""Keep n-dimensional Linear semantics around bitsandbytes INT8 matmul."""
from functools import wraps
import logging


INT8_THRESHOLDS = {'fast': 0.0, 'balanced': 6.0}


def configure_allocator(environment):
    # Configure the owned child before importing Torch. Respect either user alias.
    if not any(name in environment for name in ('PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF')):
        environment['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
    return environment


def validate_int8_profile(value):
    if not isinstance(value, str) or value not in INT8_THRESHOLDS:
        raise ValueError('KREA_INT8_PROFILE must be fast or balanced')
    return value


def int8_engine_label(profile):
    return f'Krea 14B ({validate_int8_profile(profile).title()} INT8)'


def prepare_vae_layout(vae, profile):
    if validate_int8_profile(profile) == 'fast':
        import torch
        # Convert Conv3d weights only; retain tensor dimension/weight semantics.
        torch.nn.utils.convert_conv3d_weight_memory_format(vae, torch.channels_last_3d)


class _CastWarningOnce(logging.Filter):
    def __init__(self):
        super().__init__()
        self.seen = set()

    def filter(self, record):
        # Exact pinned bitsandbytes warning only; preserve other diagnostics.
        if (record.levelno != logging.WARNING or record.msg !=
                'MatMul8bitLt: inputs will be cast from %s to float16 during quantization'):
            return True
        message = record.getMessage()
        if message in self.seen:
            return False
        self.seen.add(message)
        return True


def limit_cast_warnings():
    target = logging.getLogger('bitsandbytes.autograd._functions')
    if not any(isinstance(item, _CastWarningOnce) for item in target.filters):
        target.addFilter(_CastWarningOnce())


def _linear_forward(forward):
    @wraps(forward)
    def wrapped(value, *args, **kwargs):
        if value.ndim <= 3:
            return forward(value, *args, **kwargs)
        # bitsandbytes 0.50.2 flattens rank 3 only. Krea's output head passes
        # [batch, frames, tokens, features], which breaks its outlier indexing.
        result = forward(value.reshape(-1, value.shape[-1]), *args, **kwargs)
        return result.reshape(*value.shape[:-1], result.shape[-1])
    return wrapped


def prepare_int8_linears(model, linear_type=None):
    if linear_type is None:
        from bitsandbytes.nn import Linear8bitLt
        linear_type = Linear8bitLt
    count = 0
    for layer in model.modules():
        if isinstance(layer, linear_type) and not getattr(layer, '_krea_flatten_nd', False):
            layer.forward = _linear_forward(layer.forward)
            layer._krea_flatten_nd = True
            count += 1
    return count
