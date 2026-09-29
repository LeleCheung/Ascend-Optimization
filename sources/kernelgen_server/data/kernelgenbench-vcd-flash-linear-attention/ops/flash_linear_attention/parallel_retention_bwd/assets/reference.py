"""flash-linear-attention 参考实现：parallel_retention_bwd。
"""
from fla.ops.retention import parallel_retention


def _grads(fwd, q, k, v, do):
    q = q.clone().detach().requires_grad_()
    k = k.clone().detach().requires_grad_()
    v = v.clone().detach().requires_grad_()
    o = fwd(q, k, v)
    (o * do).sum().backward()
    return {"dq": q.grad, "dk": k.grad, "dv": v.grad}


def _baseline_parallel_retention_bwd(q, k, v, do):
    def fwd(q, k, v):
        o, _ = parallel_retention(q, k, v)
        return o
    return _grads(fwd, q, k, v, do)


def parallel_retention_bwd(*args, **kwargs):
    return _baseline_parallel_retention_bwd(*args, **kwargs)
