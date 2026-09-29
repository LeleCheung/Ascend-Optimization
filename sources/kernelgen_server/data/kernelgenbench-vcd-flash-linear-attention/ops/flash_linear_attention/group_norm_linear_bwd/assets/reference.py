"""flash-linear-attention 参考实现：group_norm_linear_bwd。
"""
from fla.modules import GroupNormLinear


def _grad(fwd, x, norm_weight, norm_bias, num_groups, linear_weight, linear_bias, eps, do):
    x = x.clone().detach().requires_grad_()
    norm_weight = norm_weight.clone().detach().requires_grad_()
    norm_bias = norm_bias.clone().detach().requires_grad_() if norm_bias is not None else None
    linear_weight = linear_weight.clone().detach().requires_grad_()
    linear_bias = linear_bias.clone().detach().requires_grad_() if linear_bias is not None else None
    o = fwd(x, norm_weight, norm_bias, num_groups, linear_weight, linear_bias, eps)
    (o * do).sum().backward()
    result = {"dx": x.grad, "dnorm_weight": norm_weight.grad, "dlinear_weight": linear_weight.grad}
    if norm_bias is not None:
        result["dnorm_bias"] = norm_bias.grad
    if linear_bias is not None:
        result["dlinear_bias"] = linear_bias.grad
    return result


def _baseline_group_norm_linear_bwd(x, norm_weight, norm_bias, num_groups, linear_weight, linear_bias, eps, do):
    def _module_fwd(x, norm_weight, norm_bias, num_groups, linear_weight, linear_bias, eps):
        module = GroupNormLinear(num_groups=num_groups, hidden_size=x.shape[-1],
                                elementwise_affine=True, bias=norm_bias is not None, eps=eps)
        module = module.to(x.device)
        module.weight.data.copy_(norm_weight)
        if norm_bias is not None:
            module.bias.data.copy_(norm_bias)
        return module(x, linear_weight, linear_bias)
    return _grad(_module_fwd, x, norm_weight, norm_bias, num_groups, linear_weight, linear_bias, eps, do)


def group_norm_linear_bwd(*args, **kwargs):
    return _baseline_group_norm_linear_bwd(*args, **kwargs)
