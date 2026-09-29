"""flash-linear-attention 参考实现：parallel_based_fwd。
"""
from fla.ops.based import parallel_based


def _baseline_parallel_based_fwd(q, k, v, scale):
    return parallel_based(q, k, v, scale=scale, use_norm=True)


def parallel_based_fwd(*args, **kwargs):
    return _baseline_parallel_based_fwd(*args, **kwargs)
