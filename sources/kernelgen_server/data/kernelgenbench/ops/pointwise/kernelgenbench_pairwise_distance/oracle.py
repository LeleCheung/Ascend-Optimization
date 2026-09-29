REFERENCE_DEVICE = 'target'

import torch

def run(x1, x2, p, eps, keepdim):
    return torch.ops.aten.pairwise_distance(x1, x2, p, eps, keepdim)
