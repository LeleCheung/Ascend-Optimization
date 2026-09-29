REFERENCE_DEVICE = 'target'

import torch

def run(x, dim, keepdim):
    return torch.ops.aten.amin(x, dim, keepdim)
