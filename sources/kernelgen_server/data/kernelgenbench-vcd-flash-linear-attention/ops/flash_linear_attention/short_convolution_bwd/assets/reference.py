"""flash-linear-attention 参考实现：short_convolution_bwd。

注意：对张量参数 x/weight 直接求梯度，不再依赖 module；input 只含张量。dweight 的
梯度形状与 nn.Conv1d 权重一致 [D,1,W]（对 squeeze 前的 weight 求导）。
"""
from fla.modules.conv.causal_conv1d import causal_conv1d


def _baseline_short_convolution_bwd(x, do, weight):
    x = x.clone().detach().requires_grad_()
    weight = weight.clone().detach().requires_grad_()
    y, _ = causal_conv1d(
        x=x, weight=weight.squeeze(1), bias=None,
        activation='silu', backend='triton',
    )
    (y * do).sum().backward()
    return {"dx": x.grad, "dweight": weight.grad}


def short_convolution_bwd(*args, **kwargs):
    return _baseline_short_convolution_bwd(*args, **kwargs)
