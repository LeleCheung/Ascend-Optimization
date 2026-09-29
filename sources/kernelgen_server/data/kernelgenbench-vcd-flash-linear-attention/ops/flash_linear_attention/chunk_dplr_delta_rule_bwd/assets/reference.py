"""flash-linear-attention 参考实现：chunk_dplr_delta_rule_bwd。
"""
from fla.ops.generalized_delta_rule.dplr import chunk_dplr_delta_rule


def _grads(fwd, q, k, v, a, b, gk, do):
    q = q.clone().detach().requires_grad_()
    k = k.clone().detach().requires_grad_()
    v = v.clone().detach().requires_grad_()
    a = a.clone().detach().requires_grad_()
    b = b.clone().detach().requires_grad_()
    gk = gk.clone().detach().requires_grad_()
    o = fwd(q, k, v, a, b, gk)
    (o * do).sum().backward()
    return {"dq": q.grad, "dk": k.grad, "dv": v.grad, "da": a.grad, "db": b.grad, "dgk": gk.grad}


def _baseline_chunk_dplr_delta_rule_bwd(q, k, v, a, b, gk, do):
    def fwd(q, k, v, a, b, gk):
        o, _ = chunk_dplr_delta_rule(q, k, v, a, b, gk, output_final_state=False)
        return o
    return _grads(fwd, q, k, v, a, b, gk, do)


def chunk_dplr_delta_rule_bwd(*args, **kwargs):
    return _baseline_chunk_dplr_delta_rule_bwd(*args, **kwargs)
