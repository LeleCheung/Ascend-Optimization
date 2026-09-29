"""flash-linear-attention 参考实现：fused_recurrent_iplr_delta_rule_bwd。
"""
from fla.ops.generalized_delta_rule.iplr import fused_recurrent_iplr_delta_rule


def _grads(fwd, q, k, v, a, b, do):
    q = q.clone().detach().requires_grad_()
    k = k.clone().detach().requires_grad_()
    v = v.clone().detach().requires_grad_()
    a = a.clone().detach().requires_grad_()
    b = b.clone().detach().requires_grad_()
    o = fwd(q, k, v, a, b)
    (o * do).sum().backward()
    return {"dq": q.grad, "dk": k.grad, "dv": v.grad, "da": a.grad, "db": b.grad}


def _baseline_fused_recurrent_iplr_delta_rule_bwd(q, k, v, a, b, do):
    def fwd(q, k, v, a, b):
        o, _ = fused_recurrent_iplr_delta_rule(q, k, v, a, b, output_final_state=False)
        return o
    return _grads(fwd, q, k, v, a, b, do)


def fused_recurrent_iplr_delta_rule_bwd(*args, **kwargs):
    return _baseline_fused_recurrent_iplr_delta_rule_bwd(*args, **kwargs)
