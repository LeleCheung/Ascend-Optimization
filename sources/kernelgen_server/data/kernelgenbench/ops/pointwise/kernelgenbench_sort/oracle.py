REFERENCE_DEVICE = 'target'

import torch
def run(x, dim, descending, stable):
    return torch.sort(x, dim=dim, descending=descending, stable=stable)
