"""flash-linear-attention 参考实现：fused_cross_entropy_fwd。

注意：直接调用底层 cross_entropy_loss 函数，reduction 等超参与
FusedCrossEntropyLoss(reduction='mean') 的构造配置一致并固化为常量；input 只含
张量 (logits/target)，可跨机传输（见 accuracy 侧 input_build）。
"""
from fla.modules.fused_cross_entropy import cross_entropy_loss

_IGNORE_INDEX = -100


def _baseline_fused_cross_entropy_fwd(logits, target):
    loss, _ = cross_entropy_loss(
        logits, target,
        label_smoothing=0.0, logit_scale=1.0, lse_square_scale=0.0,
        logit_softcapping=None, ignore_index=_IGNORE_INDEX,
        inplace_backward=False, process_group=None,
    )
    # reduction='mean'（与 FusedCrossEntropyLoss.forward 一致）
    return loss.sum() / (target != _IGNORE_INDEX).sum()


def fused_cross_entropy_fwd(*args, **kwargs):
    return _baseline_fused_cross_entropy_fwd(*args, **kwargs)
