"""flash-linear-attention 参考实现：fused_layernorm_swishgate_bwd。
"""
from fla.modules import FusedLayerNormSwishGate


def _grad(fwd, x, g, weight, bias, eps, do):
    x = x.clone().detach().requires_grad_()
    g = g.clone().detach().requires_grad_()
    weight = weight.clone().detach().requires_grad_()
    bias = bias.clone().detach().requires_grad_() if bias is not None else None
    o = fwd(x, g, weight, bias, eps)
    (o * do).sum().backward()
    result = {"dx": x.grad, "dg": g.grad, "dweight": weight.grad}
    if bias is not None:
        result["dbias"] = bias.grad
    return result


def _baseline_fused_layernorm_swishgate_bwd(x, g, weight, bias, eps, do):
    def _module_fwd(x, g, weight, bias, eps):
        module = FusedLayerNormSwishGate(hidden_size=x.shape[-1], elementwise_affine=True,
                                        bias=bias is not None, eps=eps)
        module = module.to(x.device)
        module.weight.data.copy_(weight)
        if bias is not None:
            module.bias.data.copy_(bias)
        return module(x, g)
    return _grad(_module_fwd, x, g, weight, bias, eps, do)


def fused_layernorm_swishgate_bwd(*args, **kwargs):
    return _baseline_fused_layernorm_swishgate_bwd(*args, **kwargs)
