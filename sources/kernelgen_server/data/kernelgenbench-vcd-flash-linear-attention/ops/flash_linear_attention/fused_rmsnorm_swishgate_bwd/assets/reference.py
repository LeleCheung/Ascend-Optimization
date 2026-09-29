"""flash-linear-attention 参考实现：fused_rmsnorm_swishgate_bwd。
"""
from fla.modules import FusedRMSNormSwishGate


def _grad(fwd, x, g, weight, eps, do):
    x = x.clone().detach().requires_grad_()
    g = g.clone().detach().requires_grad_()
    weight = weight.clone().detach().requires_grad_()
    o = fwd(x, g, weight, eps)
    (o * do).sum().backward()
    return {"dx": x.grad, "dg": g.grad, "dweight": weight.grad}


def _baseline_fused_rmsnorm_swishgate_bwd(x, g, weight, eps, do):
    def _module_fwd(x, g, weight, eps):
        module = FusedRMSNormSwishGate(hidden_size=x.shape[-1], elementwise_affine=True, eps=eps)
        module = module.to(x.device)
        module.weight.data.copy_(weight)
        return module(x, g)
    return _grad(_module_fwd, x, g, weight, eps, do)


def fused_rmsnorm_swishgate_bwd(*args, **kwargs):
    return _baseline_fused_rmsnorm_swishgate_bwd(*args, **kwargs)
