"""flash-linear-attention 参考实现：fused_bit_linear_bwd。

注意：直接对张量参数（x / norm_weight / weight）求梯度，不再依赖 module 的
.parameters()，以便 input 可跨机传输（见 accuracy 侧 input_build）。梯度 key
dx/dweight/dnorm_weight 分别对应 x / linear weight / rms-norm weight。
"""
from fla.modules.fused_bitlinear import layer_norm_linear_quant_fn


def _baseline_fused_bit_linear_bwd(x, do, norm_weight, weight):
    x = x.clone().detach().requires_grad_()
    weight = weight.clone().detach().requires_grad_()
    norm_weight = norm_weight.clone().detach().requires_grad_()
    o = layer_norm_linear_quant_fn(x, norm_weight, None, weight, None, is_rms_norm=True)
    (o * do).sum().backward()
    return {"dx": x.grad, "dweight": weight.grad, "dnorm_weight": norm_weight.grad}


def fused_bit_linear_bwd(*args, **kwargs):
    return _baseline_fused_bit_linear_bwd(*args, **kwargs)
