"""flash-linear-attention 参考实现：fused_chunk_based_fwd。
"""
from fla.ops.based import fused_chunk_based


def _baseline_fused_chunk_based_fwd(q, k, v, scale):
    return fused_chunk_based(q, k, v, scale=scale, use_norm=True)


def fused_chunk_based_fwd(*args, **kwargs):
    return _baseline_fused_chunk_based_fwd(*args, **kwargs)
