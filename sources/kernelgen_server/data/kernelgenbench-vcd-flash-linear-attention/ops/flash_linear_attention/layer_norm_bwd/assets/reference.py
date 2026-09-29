"""flash-linear-attention 参考实现：layer_norm_bwd。
"""
from fla.modules import LayerNorm


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


def _baseline_layer_norm_bwd(x, weight, bias, eps, do):
    def _module_fwd(x, weight, bias, eps):
        module = LayerNorm(hidden_size=x.shape[-1], elementwise_affine=True, bias=bias is not None, eps=eps)
        module = module.to(x.device)
        module.weight.data.copy_(weight)
        if bias is not None:
            module.bias.data.copy_(bias)
        return module(x)
    return _grad(_module_fwd, x, weight, bias, eps, do)


def layer_norm_bwd(*args, **kwargs):
    return _baseline_layer_norm_bwd(*args, **kwargs)
