"""flash-linear-attention 参考实现：chunk_precond_kda_bwd。
"""
from fla.ops.precond_kda import chunk_precond_kda


def _grads(fwd, q, k, v, g, g_atk, beta_atk, beta, do):
    q = q.clone().detach().requires_grad_()
    k = k.clone().detach().requires_grad_()
    v = v.clone().detach().requires_grad_()
    g = g.clone().detach().requires_grad_()
    g_atk = g_atk.clone().detach().requires_grad_()
    beta_atk = beta_atk.clone().detach().requires_grad_()
    beta = beta.clone().detach().requires_grad_()
    o = fwd(q, k, v, g, g_atk, beta_atk, beta)
    (o * do).sum().backward()
    return {"dq": q.grad, "dk": k.grad, "dv": v.grad, "dg": g.grad,
            "dg_atk": g_atk.grad, "dbeta_atk": beta_atk.grad, "dbeta": beta.grad}


def _baseline_chunk_precond_kda_bwd(q, k, v, g, g_atk, beta_atk, beta, do):
    def fwd(q, k, v, g, g_atk, beta_atk, beta):
        o, _, _ = chunk_precond_kda(q, k, v, g, g_atk, beta_atk, beta, output_final_state=False)
        return o
    return _grads(fwd, q, k, v, g, g_atk, beta_atk, beta, do)


def chunk_precond_kda_bwd(*args, **kwargs):
    return _baseline_chunk_precond_kda_bwd(*args, **kwargs)
