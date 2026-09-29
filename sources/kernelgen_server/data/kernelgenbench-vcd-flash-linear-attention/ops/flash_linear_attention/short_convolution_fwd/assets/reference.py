"""flash-linear-attention 参考实现：short_convolution_fwd。

注意：直接调用底层 causal_conv1d 函数，activation='silu'/backend='triton'/bias=None
与 ShortConvolution 的构造配置一致并固化为常量；input 只含张量 (x/weight)，可跨机
传输（见 accuracy 侧 input_build）。weight 为 nn.Conv1d 权重，shape [D,1,W]，底层
函数需要 [D,W]，故 squeeze(1)（与 module.forward 的 rearrange "d 1 w -> d w" 等价）。
"""
from fla.modules.conv.causal_conv1d import causal_conv1d


def _baseline_short_convolution_fwd(x, weight):
    y, _ = causal_conv1d(
        x=x, weight=weight.squeeze(1), bias=None,
        activation='silu', backend='triton',
    )
    return y


def short_convolution_fwd(*args, **kwargs):
    return _baseline_short_convolution_fwd(*args, **kwargs)
