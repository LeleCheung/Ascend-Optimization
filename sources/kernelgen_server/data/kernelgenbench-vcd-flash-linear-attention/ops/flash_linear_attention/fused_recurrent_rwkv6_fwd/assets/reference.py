"""flash-linear-attention 参考实现：fused_recurrent_rwkv6_fwd。
"""
from fla.ops.rwkv6 import fused_recurrent_rwkv6


def _baseline_fused_recurrent_rwkv6_fwd(r, k, v, w, u):
    o, _ = fused_recurrent_rwkv6(r, k, v, w, u, output_final_state=False)
    return o


def fused_recurrent_rwkv6_fwd(*args, **kwargs):
    return _baseline_fused_recurrent_rwkv6_fwd(*args, **kwargs)
