"""flash-linear-attention 参考实现：fused_recurrent_rwkv7。
"""
from fla.ops import fused_recurrent_rwkv7 as _fla_fused_recurrent_rwkv7


def _baseline_fused_recurrent_rwkv7(r, w, k, v, a, b):
    o, _ = _fla_fused_recurrent_rwkv7(r, w, k, v, a, b, output_final_state=False)
    return o


def fused_recurrent_rwkv7(*args, **kwargs):
    return _baseline_fused_recurrent_rwkv7(*args, **kwargs)
