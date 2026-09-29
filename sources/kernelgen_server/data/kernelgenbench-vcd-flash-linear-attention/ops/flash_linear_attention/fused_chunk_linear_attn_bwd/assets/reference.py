"""flash-linear-attention 参考实现：fused_chunk_linear_attn_bwd。
"""
from fla.ops.linear_attn import fused_chunk_linear_attn


def _grads(fwd, q, k, v, do):
    q = q.clone().detach().requires_grad_()
    k = k.clone().detach().requires_grad_()
    v = v.clone().detach().requires_grad_()
    o = fwd(q, k, v)
    (o * do).sum().backward()
    return {"dq": q.grad, "dk": k.grad, "dv": v.grad}


def _baseline_fused_chunk_linear_attn_bwd(q, k, v, do):
    def fwd(q, k, v):
        o, _ = fused_chunk_linear_attn(q, k, v, normalize=False, output_final_state=False)
        return o
    return _grads(fwd, q, k, v, do)


def fused_chunk_linear_attn_bwd(*args, **kwargs):
    return _baseline_fused_chunk_linear_attn_bwd(*args, **kwargs)
