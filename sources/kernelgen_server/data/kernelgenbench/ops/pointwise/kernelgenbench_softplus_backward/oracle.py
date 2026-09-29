REFERENCE_DEVICE = 'target'

import torch

def run(grad_output, x, beta, threshold):
    return torch.ops.aten.softplus_backward(grad_output, x, beta, threshold)
