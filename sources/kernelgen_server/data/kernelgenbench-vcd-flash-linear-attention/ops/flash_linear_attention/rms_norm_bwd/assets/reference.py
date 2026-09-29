"""flash-linear-attention 参考实现：rms_norm_bwd。
"""
from fla.modules.layernorm import rms_norm


def _grad(fwd, x, weight, bias, eps, do):
    x = x.clone().detach().requires_grad_()
    weight = weight.clone().detach().requires_grad_()
    bias = bias.clone().detach().requires_grad_() if bias is not None else None
    o = fwd(x, weight, bias, eps)
    (o * do).sum().backward()
    result = {"dx": x.grad, "dweight": weight.grad}
    if bias is not None:
        result["dbias"] = bias.grad
    return result


def _baseline_rms_norm_bwd(x, weight, bias, eps, do):
    return _grad(lambda x, w, b, e: rms_norm(x, w, b, eps=e), x, weight, bias, eps, do)


def rms_norm_bwd(*args, **kwargs):
    return _baseline_rms_norm_bwd(*args, **kwargs)
