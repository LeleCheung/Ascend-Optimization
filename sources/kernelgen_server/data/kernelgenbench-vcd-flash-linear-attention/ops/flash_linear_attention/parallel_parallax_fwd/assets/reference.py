"""flash-linear-attention 参考实现：parallel_parallax_fwd。
"""
from fla.ops.parallax import parallel_parallax


def _baseline_parallel_parallax_fwd(q, r, k, v, scale):
    return parallel_parallax(q, r, k, v, scale=scale)


def parallel_parallax_fwd(*args, **kwargs):
    return _baseline_parallel_parallax_fwd(*args, **kwargs)
