REFERENCE_DEVICE = 'target'

import torch

def run(grad_output, x, target, reduction, beta):
    return torch.ops.aten.smooth_l1_loss_backward(grad_output, x, target, reduction, beta)
