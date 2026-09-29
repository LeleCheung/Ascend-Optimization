"""flash-linear-attention 参考实现：group_norm_fwd。
"""
from fla.modules import GroupNorm


def _baseline_group_norm_fwd(x, weight, bias, num_groups, eps):
    module = GroupNorm(num_groups=num_groups, hidden_size=x.shape[-1],
                      elementwise_affine=True, bias=bias is not None, eps=eps)
    module = module.to(x.device)
    module.weight.data.copy_(weight)
    if bias is not None:
        module.bias.data.copy_(bias)
    return module(x)


def group_norm_fwd(*args, **kwargs):
    return _baseline_group_norm_fwd(*args, **kwargs)
