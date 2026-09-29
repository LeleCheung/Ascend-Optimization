"""flash-linear-attention 参考实现：layer_norm_linear_fwd。
"""
from fla.modules import LayerNormLinear


def _baseline_layer_norm_linear_fwd(x, norm_weight, norm_bias, linear_weight, linear_bias, eps):
    module = LayerNormLinear(hidden_size=x.shape[-1], elementwise_affine=True,
                            bias=norm_bias is not None, eps=eps)
    module = module.to(x.device)
    module.weight.data.copy_(norm_weight)
    if norm_bias is not None:
        module.bias.data.copy_(norm_bias)
    return module(x, linear_weight, linear_bias)


def layer_norm_linear_fwd(*args, **kwargs):
    return _baseline_layer_norm_linear_fwd(*args, **kwargs)
