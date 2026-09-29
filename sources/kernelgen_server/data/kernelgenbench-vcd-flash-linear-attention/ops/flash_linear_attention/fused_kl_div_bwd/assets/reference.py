"""flash-linear-attention 参考实现：fused_kl_div_bwd。

注意：只对学生侧 x/weight 求梯度，teacher 侧 (target_x/target_weight) detach；
不再依赖 module；input 只含张量。dx/dweight 语义不变。
"""
from fla.modules.fused_kl_div import fused_kl_div_loss


def _baseline_fused_kl_div_bwd(x, target_x, weight, target_weight, do):
    x = x.clone().detach().requires_grad_()
    weight = weight.clone().detach().requires_grad_()
    target_x = target_x.clone().detach()
    target_weight = target_weight.clone().detach()
    loss = fused_kl_div_loss(
        x=x, target_x=target_x, weight=weight, target_weight=target_weight,
        reduction='batchmean', accumulate_grad_in_fp32=True,
    )
    (loss * do).backward()
    return {"dx": x.grad, "dweight": weight.grad}


def fused_kl_div_bwd(*args, **kwargs):
    return _baseline_fused_kl_div_bwd(*args, **kwargs)
