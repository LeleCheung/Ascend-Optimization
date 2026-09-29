"""flash-linear-attention 参考实现：fused_linear_cross_entropy_bwd。

注意：对 hidden/weight 直接求梯度，不再依赖 module；input 只含张量。
dhidden/dweight 语义不变。
"""
from fla.modules.fused_linear_cross_entropy import fused_linear_cross_entropy_loss


def _baseline_fused_linear_cross_entropy_bwd(hidden, weight, target, do):
    hidden = hidden.clone().detach().requires_grad_()
    weight = weight.clone().detach().requires_grad_()
    loss = fused_linear_cross_entropy_loss(
        hidden.view(-1, hidden.shape[-1]), target.view(-1),
        weight=weight, bias=None,
        ignore_index=-100, label_smoothing=0.0, logit_scale=1.0,
        logit_softcapping=None, num_chunks=8, reduction='mean',
        use_l2warp=False, l2_penalty_factor=1e-4, accumulate_grad_in_fp32=True,
    )
    (loss * do).backward()
    return {"dhidden": hidden.grad, "dweight": weight.grad}


def fused_linear_cross_entropy_bwd(*args, **kwargs):
    return _baseline_fused_linear_cross_entropy_bwd(*args, **kwargs)
