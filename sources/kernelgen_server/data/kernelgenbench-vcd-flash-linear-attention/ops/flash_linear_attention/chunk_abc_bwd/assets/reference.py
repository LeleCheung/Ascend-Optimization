"""flash-linear-attention 参考实现：chunk_abc_bwd。
"""
from fla.ops.abc import chunk_abc


def _grads(fwd, q, k, v, s, do):
    q = q.clone().detach().requires_grad_()
    k = k.clone().detach().requires_grad_()
    v = v.clone().detach().requires_grad_()
    s = s.clone().detach().requires_grad_()
    o = fwd(q, k, v, s)
    (o * do).sum().backward()
    return {"dq": q.grad, "dk": k.grad, "dv": v.grad, "ds": s.grad}


def _baseline_chunk_abc_bwd(q, k, v, s, do):
    def fwd(q, k, v, s):
        o, _ = chunk_abc(q, k, v, s, output_final_state=False)
        return o
    return _grads(fwd, q, k, v, s, do)


def chunk_abc_bwd(*args, **kwargs):
    return _baseline_chunk_abc_bwd(*args, **kwargs)
