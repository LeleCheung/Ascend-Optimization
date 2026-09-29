"""flash-linear-attention 参考实现：chunk_rwkv6_fwd。
"""
from fla.ops.rwkv6 import chunk_rwkv6


def _baseline_chunk_rwkv6_fwd(r, k, v, w, u):
    o, _ = chunk_rwkv6(r, k, v, w, u, output_final_state=False)
    return o


def chunk_rwkv6_fwd(*args, **kwargs):
    return _baseline_chunk_rwkv6_fwd(*args, **kwargs)
