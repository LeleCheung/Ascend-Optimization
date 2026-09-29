"""flash-linear-attention 参考实现：fused_recurrent_simple_gla_bwd。
"""
from fla.ops.simple_gla import fused_recurrent_simple_gla


def _grads(fwd, q, k, v, g, do):
    q = q.clone().detach().requires_grad_()
    k = k.clone().detach().requires_grad_()
    v = v.clone().detach().requires_grad_()
    g = g.clone().detach().requires_grad_()
    o = fwd(q, k, v, g)
    (o * do).sum().backward()
    return {"dq": q.grad, "dk": k.grad, "dv": v.grad, "dg": g.grad}


def _baseline_fused_recurrent_simple_gla_bwd(q, k, v, g, do):
    def fwd(q, k, v, g):
        o, _ = fused_recurrent_simple_gla(q, k, v, g, output_final_state=False)
        return o
    return _grads(fwd, q, k, v, g, do)


def fused_recurrent_simple_gla_bwd(*args, **kwargs):
    return _baseline_fused_recurrent_simple_gla_bwd(*args, **kwargs)
