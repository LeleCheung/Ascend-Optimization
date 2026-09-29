REFERENCE_DEVICE = 'target'

import torch
def _rmsnorm(x, weight, eps):
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
def run(inputs_embeds, previous_hidden, enorm_weight, hnorm_weight, eps):
    e = _rmsnorm(inputs_embeds, enorm_weight, eps)
    h = _rmsnorm(previous_hidden, hnorm_weight, eps)
    return torch.cat([e, h], dim=-1).to(inputs_embeds.dtype)
