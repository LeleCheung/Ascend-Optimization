"""flash-linear-attention 参考实现：fused_linear_cross_entropy_fwd。

注意：直接调用底层 fused_linear_cross_entropy_loss 函数，超参与
FusedLinearCrossEntropyLoss(reduction='mean') 的构造配置一致并固化为常量；
input 只含张量 (hidden/weight/target)，可跨机传输。
"""
from fla.modules.fused_linear_cross_entropy import fused_linear_cross_entropy_loss


def _baseline_fused_linear_cross_entropy_fwd(hidden, weight, target):
    return fused_linear_cross_entropy_loss(
        hidden.view(-1, hidden.shape[-1]), target.view(-1),
        weight=weight, bias=None,
        ignore_index=-100, label_smoothing=0.0, logit_scale=1.0,
        logit_softcapping=None, num_chunks=8, reduction='mean',
        use_l2warp=False, l2_penalty_factor=1e-4, accumulate_grad_in_fp32=True,
    )


def fused_linear_cross_entropy_fwd(*args, **kwargs):
    return _baseline_fused_linear_cross_entropy_fwd(*args, **kwargs)
