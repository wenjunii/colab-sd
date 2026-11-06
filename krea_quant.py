"""Keep n-dimensional Linear semantics around bitsandbytes INT8 matmul."""
from functools import wraps


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
