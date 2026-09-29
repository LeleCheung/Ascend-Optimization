"""flash-linear-attention 参考实现：parallel_retention_fwd。
"""
from fla.ops.retention import parallel_retention


def _baseline_parallel_retention_fwd(q, k, v):
    o, _ = parallel_retention(q, k, v)
    return o


def parallel_retention_fwd(*args, **kwargs):
    return _baseline_parallel_retention_fwd(*args, **kwargs)
