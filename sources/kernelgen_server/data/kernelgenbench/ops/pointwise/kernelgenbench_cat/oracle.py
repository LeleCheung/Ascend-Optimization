REFERENCE_DEVICE = 'target'

import torch

def run(x, other, dim):
    return torch.ops.aten.cat([x, other], dim)
