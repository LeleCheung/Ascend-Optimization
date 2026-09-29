"""flash-linear-attention 参考实现：chunk_rwkv7_bwd。
"""
from fla.ops.rwkv7 import chunk_rwkv7


def _grads(fwd, r, w, k, v, a, b, do):
    r = r.clone().detach().requires_grad_()
    w = w.clone().detach().requires_grad_()
    k = k.clone().detach().requires_grad_()
    v = v.clone().detach().requires_grad_()
    a = a.clone().detach().requires_grad_()
    b = b.clone().detach().requires_grad_()
    o = fwd(r, w, k, v, a, b)
    (o * do).sum().backward()
    return {"dr": r.grad, "dw": w.grad, "dk": k.grad, "dv": v.grad, "da": a.grad, "db": b.grad}


def _baseline_chunk_rwkv7_bwd(r, w, k, v, a, b, do):
    def fwd(r, w, k, v, a, b):
        o, _ = chunk_rwkv7(r, w, k, v, a, b, output_final_state=False)
        return o
    return _grads(fwd, r, w, k, v, a, b, do)


def chunk_rwkv7_bwd(*args, **kwargs):
    return _baseline_chunk_rwkv7_bwd(*args, **kwargs)
