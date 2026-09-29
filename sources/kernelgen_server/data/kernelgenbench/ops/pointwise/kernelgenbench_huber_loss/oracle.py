REFERENCE_DEVICE = 'target'

import torch

def run(input, target, reduction, delta):
    return torch.ops.aten.huber_loss(input, target, reduction, delta)
