"""flash-linear-attention 参考实现：chunk_rwkv7_fwd。
"""
from fla.ops.rwkv7 import chunk_rwkv7


def _baseline_chunk_rwkv7_fwd(r, w, k, v, a, b):
    o, _ = chunk_rwkv7(r, w, k, v, a, b, output_final_state=False)
    return o


def chunk_rwkv7_fwd(*args, **kwargs):
    return _baseline_chunk_rwkv7_fwd(*args, **kwargs)
