"""flash-linear-attention 参考实现：fused_recurrent_hgrn_bwd。
"""
from fla.ops.hgrn import fused_recurrent_hgrn


def _grads(fwd, x, g, do):
    x = x.clone().detach().requires_grad_()
    g = g.clone().detach().requires_grad_()
    o = fwd(x, g)
    (o * do).sum().backward()
    return {"dx": x.grad, "dg": g.grad}


def _baseline_fused_recurrent_hgrn_bwd(x, g, do):
    def fwd(x, g):
        o, _ = fused_recurrent_hgrn(x, g, output_final_state=False)
        return o
    return _grads(fwd, x, g, do)


def fused_recurrent_hgrn_bwd(*args, **kwargs):
    return _baseline_fused_recurrent_hgrn_bwd(*args, **kwargs)
