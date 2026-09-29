REFERENCE_DEVICE = 'target'

import torch

def run(x, dim, start, length):
    return torch.ops.aten.narrow(x, dim, start, length)
