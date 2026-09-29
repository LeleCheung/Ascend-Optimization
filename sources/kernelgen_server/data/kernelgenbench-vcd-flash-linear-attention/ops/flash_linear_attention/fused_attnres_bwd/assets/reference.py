"""flash-linear-attention 参考实现：fused_attnres_bwd。
"""
from fla.ops.attnres import fused_attnres


def _grads(fwd, query, residuals, rms_weight, scale, do):
    # 对每个残差源求梯度
    res_leaves = [r.clone().detach().requires_grad_() for r in residuals]
    o = fwd(query, res_leaves, rms_weight, scale)
    (o * do).sum().backward()
    return {f"dres{i}": res_leaves[i].grad for i in range(len(res_leaves))}


def _baseline_fused_attnres_bwd(query, residuals, rms_weight, scale, do):
    def fwd(query, residuals, rms_weight, scale):
        return fused_attnres(query, residuals, rms_weight, scale=scale)
    return _grads(fwd, query, residuals, rms_weight, scale, do)


def fused_attnres_bwd(*args, **kwargs):
    return _baseline_fused_attnres_bwd(*args, **kwargs)
