"""flash-linear-attention 参考实现：chunk_simple_gla_fwd。
"""
from fla.ops.simple_gla import chunk_simple_gla


def _baseline_chunk_simple_gla_fwd(q, k, v, g):
    o, _ = chunk_simple_gla(q, k, v, g, output_final_state=False)
    return o


def chunk_simple_gla_fwd(*args, **kwargs):
    return _baseline_chunk_simple_gla_fwd(*args, **kwargs)
