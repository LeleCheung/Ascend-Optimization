"""flash-linear-attention 参考实现：fused_rmsnorm_gated_fwd。
"""
from fla.modules import FusedRMSNormGated


def _baseline_fused_rmsnorm_gated_fwd(x, g, weight, eps):
    module = FusedRMSNormGated(hidden_size=x.shape[-1], elementwise_affine=True,
                              activation="swish", eps=eps)
    module = module.to(x.device)
    module.weight.data.copy_(weight)
    return module(x, g)


def fused_rmsnorm_gated_fwd(*args, **kwargs):
    return _baseline_fused_rmsnorm_gated_fwd(*args, **kwargs)
