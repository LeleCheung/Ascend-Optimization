"""flash-linear-attention 参考实现：chunk_abc_fwd。
"""
from fla.ops.abc import chunk_abc


def _baseline_chunk_abc_fwd(q, k, v, s):
    o, _ = chunk_abc(q, k, v, s, output_final_state=False)
    return o


def chunk_abc_fwd(*args, **kwargs):
    return _baseline_chunk_abc_fwd(*args, **kwargs)
