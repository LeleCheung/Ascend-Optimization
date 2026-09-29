"""flash-linear-attention 参考实现：l2norm_bwd。
"""
from fla.modules.l2norm import l2norm


def _grad(fwd, x, eps, do):
    x = x.clone().detach().requires_grad_()
    o = fwd(x, eps)
    (o * do).sum().backward()
    return {"dx": x.grad}


def _baseline_l2norm_bwd(x, eps, do):
    return _grad(lambda x, eps: l2norm(x, eps), x, eps, do)


def l2norm_bwd(*args, **kwargs):
    return _baseline_l2norm_bwd(*args, **kwargs)
