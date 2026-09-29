"""flash-linear-attention 参考实现：token_shift_bwd。
"""
from fla.modules import TokenShift


def _grad(fwd, x, do):
    x = x.clone().detach().requires_grad_()
    o = fwd(x)
    if isinstance(o, tuple):
        o = o[0]
    (o * do).sum().backward()
    return {"dx": x.grad}


def _baseline_token_shift_bwd(x, do):
    return _grad(lambda x: TokenShift.apply(x, None, None, False, None), x, do)


def token_shift_bwd(*args, **kwargs):
    return _baseline_token_shift_bwd(*args, **kwargs)
