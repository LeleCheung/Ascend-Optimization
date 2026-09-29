"""flash-linear-attention 参考实现：fused_recurrent_rwkv6_bwd。
"""
from fla.ops.rwkv6 import fused_recurrent_rwkv6


def _grads(fwd, r, k, v, w, u, do):
    r = r.clone().detach().requires_grad_()
    k = k.clone().detach().requires_grad_()
    v = v.clone().detach().requires_grad_()
    w = w.clone().detach().requires_grad_()
    u = u.clone().detach().requires_grad_()
    o = fwd(r, k, v, w, u)
    (o * do).sum().backward()
    return {"dr": r.grad, "dk": k.grad, "dv": v.grad, "dw": w.grad, "du": u.grad}


def _baseline_fused_recurrent_rwkv6_bwd(r, k, v, w, u, do):
    def fwd(r, k, v, w, u):
        o, _ = fused_recurrent_rwkv6(r, k, v, w, u, output_final_state=False)
        return o
    return _grads(fwd, r, k, v, w, u, do)


def fused_recurrent_rwkv6_bwd(*args, **kwargs):
    return _baseline_fused_recurrent_rwkv6_bwd(*args, **kwargs)
