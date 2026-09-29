"""flash-linear-attention 参考实现：fused_layernorm_swishgate_linear_fwd。
"""
from fla.modules import FusedLayerNormSwishGateLinear


def _baseline_fused_layernorm_swishgate_linear_fwd(x, g, norm_weight, norm_bias, linear_weight, linear_bias, eps):
    module = FusedLayerNormSwishGateLinear(hidden_size=x.shape[-1], elementwise_affine=True, eps=eps)
    module = module.to(x.device)
    module.weight.data.copy_(norm_weight)
    return module(x, g, linear_weight, linear_bias)


def fused_layernorm_swishgate_linear_fwd(*args, **kwargs):
    return _baseline_fused_layernorm_swishgate_linear_fwd(*args, **kwargs)
