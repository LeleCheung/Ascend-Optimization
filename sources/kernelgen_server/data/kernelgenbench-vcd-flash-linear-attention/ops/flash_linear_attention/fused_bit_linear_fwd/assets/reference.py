"""flash-linear-attention 参考实现：fused_bit_linear_fwd。

注意：直接调用底层 forward 函数 layer_norm_linear_quant_fn，输入全部为张量
（x / norm_weight / weight），不再经由 FusedBitLinear module 间接调用，以便
input 可跨机传输（见 accuracy 侧 input_build）。
"""
from fla.modules.fused_bitlinear import layer_norm_linear_quant_fn


def _baseline_fused_bit_linear_fwd(x, norm_weight, weight):
    # bias=False -> linear_bias=None；RMSNorm(bias=False) -> norm_bias=None。
    # 不传 eps，与 FusedBitLinear.forward 一致（默认 eps=1e-6）。
    return layer_norm_linear_quant_fn(x, norm_weight, None, weight, None, is_rms_norm=True)


def fused_bit_linear_fwd(*args, **kwargs):
    return _baseline_fused_bit_linear_fwd(*args, **kwargs)
