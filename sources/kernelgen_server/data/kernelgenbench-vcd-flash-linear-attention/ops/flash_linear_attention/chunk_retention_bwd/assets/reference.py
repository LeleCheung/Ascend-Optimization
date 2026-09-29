"""flash-linear-attention 参考实现：chunk_retention_bwd。
"""
from fla.ops.retention import chunk_retention


def _grads(fwd, q, k, v, do):
    q = q.clone().detach().requires_grad_()
    k = k.clone().detach().requires_grad_()
    v = v.clone().detach().requires_grad_()
    o = fwd(q, k, v)
    (o * do).sum().backward()
    return {"dq": q.grad, "dk": k.grad, "dv": v.grad}


def _baseline_chunk_retention_bwd(q, k, v, do):
    def fwd(q, k, v):
        o, _ = chunk_retention(q, k, v, output_final_state=False)
        return o
    return _grads(fwd, q, k, v, do)


def chunk_retention_bwd(*args, **kwargs):
    return _baseline_chunk_retention_bwd(*args, **kwargs)
