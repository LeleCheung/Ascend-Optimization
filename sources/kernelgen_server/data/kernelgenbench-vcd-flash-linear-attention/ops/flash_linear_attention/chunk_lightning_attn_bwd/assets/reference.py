"""flash-linear-attention 参考实现：chunk_lightning_attn_bwd。
"""
from fla.ops.lightning_attn import chunk_lightning_attn


def _grads(fwd, q, k, v, do):
    q = q.clone().detach().requires_grad_()
    k = k.clone().detach().requires_grad_()
    v = v.clone().detach().requires_grad_()
    o = fwd(q, k, v)
    (o * do).sum().backward()
    return {"dq": q.grad, "dk": k.grad, "dv": v.grad}


def _baseline_chunk_lightning_attn_bwd(q, k, v, g, layer_idx, num_layers, do):
    def fwd(q, k, v):
        o, _ = chunk_lightning_attn(q, k, v, layer_idx, num_layers, output_final_state=False)
        return o
    return _grads(fwd, q, k, v, do)


def chunk_lightning_attn_bwd(*args, **kwargs):
    return _baseline_chunk_lightning_attn_bwd(*args, **kwargs)
