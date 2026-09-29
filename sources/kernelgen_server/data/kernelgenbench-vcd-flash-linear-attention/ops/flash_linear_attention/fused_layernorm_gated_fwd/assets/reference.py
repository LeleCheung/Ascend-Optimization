"""flash-linear-attention 参考实现：fused_layernorm_gated_fwd。
"""
from fla.modules import FusedLayerNormGated


def _baseline_fused_layernorm_gated_fwd(x, g, weight, bias, eps):
    module = FusedLayerNormGated(hidden_size=x.shape[-1], elementwise_affine=True,
                                bias=bias is not None, activation="swish", eps=eps)
    module = module.to(x.device)
    module.weight.data.copy_(weight)
    if bias is not None:
        module.bias.data.copy_(bias)
    return module(x, g)


def fused_layernorm_gated_fwd(*args, **kwargs):
    return _baseline_fused_layernorm_gated_fwd(*args, **kwargs)
