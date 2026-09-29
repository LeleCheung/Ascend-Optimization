"""flash-linear-attention 参考实现：fused_recurrent_delta_rule_bwd。
"""
from fla.ops.delta_rule import fused_recurrent_delta_rule


def _grads(fwd, q, k, v, beta, do):
    q = q.clone().detach().requires_grad_()
    k = k.clone().detach().requires_grad_()
    v = v.clone().detach().requires_grad_()
    beta = beta.clone().detach().requires_grad_()
    o = fwd(q, k, v, beta)
    (o * do).sum().backward()
    return {"dq": q.grad, "dk": k.grad, "dv": v.grad, "dbeta": beta.grad}


def _baseline_fused_recurrent_delta_rule_bwd(q, k, v, beta, do):
    def fwd(q, k, v, beta):
        o, _ = fused_recurrent_delta_rule(q, k, v, beta, output_final_state=False)
        return o
    return _grads(fwd, q, k, v, beta, do)


def fused_recurrent_delta_rule_bwd(*args, **kwargs):
    return _baseline_fused_recurrent_delta_rule_bwd(*args, **kwargs)
