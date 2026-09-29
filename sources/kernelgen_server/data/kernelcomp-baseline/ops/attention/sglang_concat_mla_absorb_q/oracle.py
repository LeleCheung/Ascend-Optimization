REFERENCE_DEVICE = 'target'

import torch
def run(a, b):
    return torch.cat([a, b], dim=-1)
