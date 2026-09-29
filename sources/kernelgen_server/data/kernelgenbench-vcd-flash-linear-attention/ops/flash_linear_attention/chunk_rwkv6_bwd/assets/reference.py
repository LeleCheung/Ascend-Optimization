"""flash-linear-attention 参考实现：chunk_rwkv6_bwd。
"""
from fla.ops.rwkv6 import chunk_rwkv6


def _grads(fwd, r, k, v, w, u, do):
    r = r.clone().detach().requires_grad_()
    k = k.clone().detach().requires_grad_()
    v = v.clone().detach().requires_grad_()
    w = w.clone().detach().requires_grad_()
    u = u.clone().detach().requires_grad_()
    o = fwd(r, k, v, w, u)
    (o * do).sum().backward()
    return {"dr": r.grad, "dk": k.grad, "dv": v.grad, "dw": w.grad, "du": u.grad}


def _baseline_chunk_rwkv6_bwd(r, k, v, w, u, do):
    def fwd(r, k, v, w, u):
        o, _ = chunk_rwkv6(r, k, v, w, u, output_final_state=False)
        return o
    return _grads(fwd, r, k, v, w, u, do)


def chunk_rwkv6_bwd(*args, **kwargs):
    return _baseline_chunk_rwkv6_bwd(*args, **kwargs)
