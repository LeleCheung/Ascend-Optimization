"""flash-linear-attention 参考实现：parallel_simple_gla_bwd。
"""
from fla.ops.simple_gla import parallel_simple_gla


def _grads(fwd, q, k, v, g, do):
    q = q.clone().detach().requires_grad_()
    k = k.clone().detach().requires_grad_()
    v = v.clone().detach().requires_grad_()
    g = g.clone().detach().requires_grad_()
    o = fwd(q, k, v, g)
    (o * do).sum().backward()
    return {"dq": q.grad, "dk": k.grad, "dv": v.grad, "dg": g.grad}


def _baseline_parallel_simple_gla_bwd(q, k, v, g, do):
    def fwd(q, k, v, g):
        o, _ = parallel_simple_gla(q, k, v, g)
        return o
    return _grads(fwd, q, k, v, g, do)


def parallel_simple_gla_bwd(*args, **kwargs):
    return _baseline_parallel_simple_gla_bwd(*args, **kwargs)
