"""flash-linear-attention 参考实现：chunk_mesa_net_bwd。
"""
from fla.ops import chunk_mesa_net as _fla_chunk_mesa_net


def _grads(fwd, q, k, v, g, beta, lamb, do):
    q = q.clone().detach().requires_grad_()
    k = k.clone().detach().requires_grad_()
    v = v.clone().detach().requires_grad_()
    g = g.clone().detach().requires_grad_()
    beta = beta.clone().detach().requires_grad_()
    lamb = lamb.clone().detach().requires_grad_()
    o = fwd(q, k, v, g, beta, lamb)
    (o * do).sum().backward()
    return {"dq": q.grad, "dk": k.grad, "dv": v.grad,
            "dg": g.grad, "dbeta": beta.grad, "dlamb": lamb.grad}


def _baseline_chunk_mesa_net_bwd(q, k, v, g, beta, lamb, do):
    def fwd(q, k, v, g, beta, lamb):
        o = _fla_chunk_mesa_net(q, k, v, g, beta, lamb, max_CG_iteration=30, output_final_state=False)[0]
        return o
    return _grads(fwd, q, k, v, g, beta, lamb, do)


def chunk_mesa_net_bwd(*args, **kwargs):
    return _baseline_chunk_mesa_net_bwd(*args, **kwargs)
