"""flash-linear-attention 参考实现：long_convolution。

注意：直接调用底层 fft_conv 函数（含 LongConvolution.forward 的 transpose 逻辑），
filter 作为纯张量随 input 传入，不再依赖 module；input 可跨机传输。gelu=False 与
LongConvolution.forward 一致。
"""
from fla.modules.conv.long_conv import fft_conv


def _baseline_long_convolution(x, filter):
    # 复刻 LongConvolution.forward: transpose -> fft_conv -> transpose
    x_t = x.transpose(1, 2)
    y = fft_conv(x_t, filter, dropout_mask=None, gelu=False)
    y = y.transpose(1, 2)
    return y.to(dtype=x.dtype)


def long_convolution(*args, **kwargs):
    return _baseline_long_convolution(*args, **kwargs)
