"""flash-linear-attention 参考实现：fused_rmsnorm_swishgate_linear_bwd。
"""
from fla.modules import FusedRMSNormSwishGateLinear


def _grad(fwd, x, g, norm_weight, linear_weight, linear_bias, eps, do):
    x = x.clone().detach().requires_grad_()
    g = g.clone().detach().requires_grad_()
    norm_weight = norm_weight.clone().detach().requires_grad_()
    linear_weight = linear_weight.clone().detach().requires_grad_()
    linear_bias = linear_bias.clone().detach().requires_grad_() if linear_bias is not None else None
    o = fwd(x, g, norm_weight, linear_weight, linear_bias, eps)
    (o * do).sum().backward()
    result = {"dx": x.grad, "dg": g.grad, "dnorm_weight": norm_weight.grad, "dlinear_weight": linear_weight.grad}
    if linear_bias is not None:
        result["dlinear_bias"] = linear_bias.grad
    return result


def _baseline_fused_rmsnorm_swishgate_linear_bwd(x, g, norm_weight, linear_weight, linear_bias, eps, do):
    def _module_fwd(x, g, norm_weight, linear_weight, linear_bias, eps):
        module = FusedRMSNormSwishGateLinear(hidden_size=x.shape[-1], elementwise_affine=True, eps=eps)
        module = module.to(x.device)
        module.weight.data.copy_(norm_weight)
        return module(x, g, linear_weight, linear_bias)
    return _grad(_module_fwd, x, g, norm_weight, linear_weight, linear_bias, eps, do)


def fused_rmsnorm_swishgate_linear_bwd(*args, **kwargs):
    return _baseline_fused_rmsnorm_swishgate_linear_bwd(*args, **kwargs)
