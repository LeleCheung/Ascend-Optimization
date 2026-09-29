REFERENCE_DEVICE = 'target'

import torch

def run(x, other, alpha):
    return torch.ops.aten.add_(x, other, alpha=alpha)
