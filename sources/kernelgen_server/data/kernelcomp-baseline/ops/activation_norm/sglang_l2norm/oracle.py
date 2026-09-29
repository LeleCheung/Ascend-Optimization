REFERENCE_DEVICE = 'target'

import torch
def run(x, eps=1e-6):
    xf = x.float()
    rstd = (xf.pow(2).sum(dim=-1, keepdim=True) + eps).rsqrt()
    return (xf * rstd).to(x.dtype)
