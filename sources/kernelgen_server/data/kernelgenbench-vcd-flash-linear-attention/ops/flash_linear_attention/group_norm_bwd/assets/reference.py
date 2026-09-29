"""flash-linear-attention 参考实现：group_norm_bwd。
"""
from fla.modules import GroupNorm


def _grad(fwd, x, weight, bias, num_groups, eps, do):
    x = x.clone().detach().requires_grad_()
    weight = weight.clone().detach().requires_grad_()
    bias = bias.clone().detach().requires_grad_() if bias is not None else None
    o = fwd(x, weight, bias, num_groups, eps)
    (o * do).sum().backward()
    result = {"dx": x.grad, "dweight": weight.grad}
    if bias is not None:
        result["dbias"] = bias.grad
    return result


def _baseline_group_norm_bwd(x, weight, bias, num_groups, eps, do):
    def _module_fwd(x, weight, bias, num_groups, eps):
        module = GroupNorm(num_groups=num_groups, hidden_size=x.shape[-1],
                          elementwise_affine=True, bias=bias is not None, eps=eps)
        module = module.to(x.device)
        module.weight.data.copy_(weight)
        if bias is not None:
            module.bias.data.copy_(bias)
        return module(x)
    return _grad(_module_fwd, x, weight, bias, num_groups, eps, do)


def group_norm_bwd(*args, **kwargs):
    return _baseline_group_norm_bwd(*args, **kwargs)
