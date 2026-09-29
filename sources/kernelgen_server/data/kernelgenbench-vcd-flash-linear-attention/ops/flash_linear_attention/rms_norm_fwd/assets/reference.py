"""flash-linear-attention 参考实现：rms_norm_fwd。
"""
from fla.modules.layernorm import rms_norm


def _baseline_rms_norm_fwd(x, weight, bias, eps):
    return rms_norm(x, weight, bias, eps=eps)


def rms_norm_fwd(*args, **kwargs):
    return _baseline_rms_norm_fwd(*args, **kwargs)
