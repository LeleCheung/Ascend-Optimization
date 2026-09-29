"""flash-linear-attention 参考实现：parallel_parallax_bwd。
"""
from fla.ops.parallax import parallel_parallax


def _grads(fwd, q, r, k, v, scale, do):
    q = q.clone().detach().requires_grad_()
    r = r.clone().detach().requires_grad_()
    k = k.clone().detach().requires_grad_()
    v = v.clone().detach().requires_grad_()
    o = fwd(q, r, k, v, scale)
    (o * do).sum().backward()
    return {"dq": q.grad, "dr": r.grad, "dk": k.grad, "dv": v.grad}


def _baseline_parallel_parallax_bwd(q, r, k, v, scale, do):
    def fwd(q, r, k, v, scale):
        return parallel_parallax(q, r, k, v, scale=scale)
    return _grads(fwd, q, r, k, v, scale, do)


def parallel_parallax_bwd(*args, **kwargs):
    return _baseline_parallel_parallax_bwd(*args, **kwargs)
