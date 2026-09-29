"""flash-linear-attention 参考实现：chunk_retention_fwd。
"""
from fla.ops.retention import chunk_retention


def _baseline_chunk_retention_fwd(q, k, v):
    o, _ = chunk_retention(q, k, v, output_final_state=False)
    return o


def chunk_retention_fwd(*args, **kwargs):
    return _baseline_chunk_retention_fwd(*args, **kwargs)
