"""flash-linear-attention 参考实现：fused_kl_div_fwd。

注意：直接调用底层 fused_kl_div_loss 函数，reduction='batchmean' 与
FusedKLDivLoss 的构造配置一致并固化为常量；input 只含张量，可跨机传输。
"""
from fla.modules.fused_kl_div import fused_kl_div_loss


def _baseline_fused_kl_div_fwd(x, target_x, weight, target_weight):
    return fused_kl_div_loss(
        x=x, target_x=target_x, weight=weight, target_weight=target_weight,
        reduction='batchmean', accumulate_grad_in_fp32=True,
    )


def fused_kl_div_fwd(*args, **kwargs):
    return _baseline_fused_kl_div_fwd(*args, **kwargs)
