"""flash-linear-attention 参考实现：parallel_wall_attn_fwd。
"""
from fla.ops.wall_attn import parallel_wall_attn


def _baseline_parallel_wall_attn_fwd(q, k, v, g, scale):
    return parallel_wall_attn(q, k, v, g, scale=scale)


def parallel_wall_attn_fwd(*args, **kwargs):
    return _baseline_parallel_wall_attn_fwd(*args, **kwargs)
