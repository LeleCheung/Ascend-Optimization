"""flash-linear-attention 参考实现：token_shift_fwd。
"""
from fla.modules import TokenShift


def _baseline_token_shift_fwd(x):
    out = TokenShift.apply(x, None, None, False, None)
    if isinstance(out, tuple):
        return out[0]
    return out


def token_shift_fwd(*args, **kwargs):
    return _baseline_token_shift_fwd(*args, **kwargs)
