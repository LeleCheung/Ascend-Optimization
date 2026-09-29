REFERENCE_DEVICE = 'target'

import torch

def run(x, dim, index):
    return torch.ops.aten.select.int(x, dim, index)
