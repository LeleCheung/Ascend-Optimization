"""flash-linear-attention 参考实现：fused_rmsnorm_swishgate_fwd。
"""
from fla.modules import FusedRMSNormSwishGate


def _baseline_fused_rmsnorm_swishgate_fwd(x, g, weight, eps):
    module = FusedRMSNormSwishGate(hidden_size=x.shape[-1], elementwise_affine=True, eps=eps)
    module = module.to(x.device)
    module.weight.data.copy_(weight)
    return module(x, g)


def fused_rmsnorm_swishgate_fwd(*args, **kwargs):
    return _baseline_fused_rmsnorm_swishgate_fwd(*args, **kwargs)
