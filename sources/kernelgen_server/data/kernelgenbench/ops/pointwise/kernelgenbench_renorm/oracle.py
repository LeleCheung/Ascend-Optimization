REFERENCE_DEVICE = 'target'

import torch

def run(x, p, dim, maxnorm):
    return torch.ops.aten.renorm(x, p, dim, maxnorm)
