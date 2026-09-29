"""flash-linear-attention 参考实现：fused_cross_entropy_bwd。

注意：对 logits 直接求梯度，不再依赖 module；input 只含张量。dlogits 语义不变。
"""
from fla.modules.fused_cross_entropy import cross_entropy_loss

_IGNORE_INDEX = -100


def _fwd(logits, target):
    loss, _ = cross_entropy_loss(
        logits, target,
        label_smoothing=0.0, logit_scale=1.0, lse_square_scale=0.0,
        logit_softcapping=None, ignore_index=_IGNORE_INDEX,
        inplace_backward=False, process_group=None,
    )
    # reduction='mean'（与 FusedCrossEntropyLoss.forward 一致）
    return loss.sum() / (target != _IGNORE_INDEX).sum()


def _baseline_fused_cross_entropy_bwd(logits, target, do):
    logits = logits.clone().detach().requires_grad_()
    loss = _fwd(logits, target)
    (loss * do).backward()
    return {"dlogits": logits.grad}


def fused_cross_entropy_bwd(*args, **kwargs):
    return _baseline_fused_cross_entropy_bwd(*args, **kwargs)
