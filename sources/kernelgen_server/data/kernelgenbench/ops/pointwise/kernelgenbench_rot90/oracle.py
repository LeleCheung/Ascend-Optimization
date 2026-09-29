REFERENCE_DEVICE = 'target'

import torch

def run(x, k, dims):
    return torch.ops.aten.rot90(x, k, dims)
