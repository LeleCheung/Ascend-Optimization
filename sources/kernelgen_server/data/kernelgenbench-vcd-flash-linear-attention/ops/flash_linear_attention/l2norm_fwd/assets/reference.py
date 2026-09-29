"""flash-linear-attention 参考实现：l2norm_fwd。
"""
from fla.modules.l2norm import l2norm


def _baseline_l2norm_fwd(x, eps):
    return l2norm(x, eps)


def l2norm_fwd(*args, **kwargs):
    return _baseline_l2norm_fwd(*args, **kwargs)
