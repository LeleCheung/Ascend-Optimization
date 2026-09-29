"""flash-linear-attention 参考实现：fused_layernorm_swishgate_fwd。
"""
from fla.modules import FusedLayerNormSwishGate


def _baseline_fused_layernorm_swishgate_fwd(x, g, weight, bias, eps):
    module = FusedLayerNormSwishGate(hidden_size=x.shape[-1], elementwise_affine=True,
                                    bias=bias is not None, eps=eps)
    module = module.to(x.device)
    module.weight.data.copy_(weight)
    if bias is not None:
        module.bias.data.copy_(bias)
    return module(x, g)


def fused_layernorm_swishgate_fwd(*args, **kwargs):
    return _baseline_fused_layernorm_swishgate_fwd(*args, **kwargs)
