REFERENCE_DEVICE = 'target'

import torch

def run(x, other, alpha):
    return torch.ops.aten.add(x, other, alpha=alpha)
