"""flash-linear-attention 参考实现：fused_recurrent_gsa_bwd。
"""
from fla.ops.gsa import fused_recurrent_gsa


def _grads(fwd, q, k, v, s, g, do):
    q = q.clone().detach().requires_grad_()
    k = k.clone().detach().requires_grad_()
    v = v.clone().detach().requires_grad_()
    s = s.clone().detach().requires_grad_()
    g = g.clone().detach().requires_grad_()
    o = fwd(q, k, v, s, g)
    (o * do).sum().backward()
    return {"dq": q.grad, "dk": k.grad, "dv": v.grad, "ds": s.grad, "dg": g.grad}


def _baseline_fused_recurrent_gsa_bwd(q, k, v, s, g, do):
    def fwd(q, k, v, s, g):
        o, _ = fused_recurrent_gsa(q, k, v, s, g, output_final_state=False)
        return o
    return _grads(fwd, q, k, v, s, g, do)


def fused_recurrent_gsa_bwd(*args, **kwargs):
    return _baseline_fused_recurrent_gsa_bwd(*args, **kwargs)
