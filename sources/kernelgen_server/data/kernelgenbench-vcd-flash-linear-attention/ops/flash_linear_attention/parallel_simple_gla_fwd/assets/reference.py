"""flash-linear-attention 参考实现：parallel_simple_gla_fwd。
"""
from fla.ops.simple_gla import parallel_simple_gla


def _baseline_parallel_simple_gla_fwd(q, k, v, g):
    o, _ = parallel_simple_gla(q, k, v, g)
    return o


def parallel_simple_gla_fwd(*args, **kwargs):
    return _baseline_parallel_simple_gla_fwd(*args, **kwargs)
