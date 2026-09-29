REFERENCE_DEVICE = 'target'

import torch

def run(x, dim):
    return torch.ops.aten.cumsum(x, dim)
