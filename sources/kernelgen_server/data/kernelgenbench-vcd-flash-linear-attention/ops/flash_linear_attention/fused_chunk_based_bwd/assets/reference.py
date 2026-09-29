"""flash-linear-attention 参考实现：fused_chunk_based_bwd。
"""
from fla.ops.based import fused_chunk_based


def _grads(fwd, q, k, v, scale, do):
    q = q.clone().detach().requires_grad_()
    k = k.clone().detach().requires_grad_()
    v = v.clone().detach().requires_grad_()
    o = fwd(q, k, v, scale)
    (o * do).sum().backward()
    return {"dq": q.grad, "dk": k.grad, "dv": v.grad}


def _baseline_fused_chunk_based_bwd(q, k, v, scale, do):
    def fwd(q, k, v, scale):
        return fused_chunk_based(q, k, v, scale=scale, use_norm=True)
    return _grads(fwd, q, k, v, scale, do)


def fused_chunk_based_bwd(*args, **kwargs):
    return _baseline_fused_chunk_based_bwd(*args, **kwargs)
