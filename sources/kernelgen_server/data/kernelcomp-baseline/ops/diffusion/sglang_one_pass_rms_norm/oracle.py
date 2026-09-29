REFERENCE_DEVICE = 'target'

import torch
def run(x, w, eps):
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * rstd * w.float()).to(x.dtype)
