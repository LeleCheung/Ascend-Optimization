"""flash-linear-attention 参考实现：fused_attnres_fwd。
"""
from fla.ops.attnres import fused_attnres


def _baseline_fused_attnres_fwd(query, residuals, rms_weight, scale):
    return fused_attnres(query, residuals, rms_weight, scale=scale)


def fused_attnres_fwd(*args, **kwargs):
    return _baseline_fused_attnres_fwd(*args, **kwargs)
