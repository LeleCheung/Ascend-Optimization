REFERENCE_DEVICE = 'target'

import torch

def run(x, dim, keepdim):
    return torch.ops.aten.sum.dim_IntList(x, dim, keepdim)
