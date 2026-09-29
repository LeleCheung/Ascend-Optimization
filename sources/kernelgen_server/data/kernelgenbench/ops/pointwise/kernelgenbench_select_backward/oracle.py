REFERENCE_DEVICE = 'target'

import torch

def run(grad_output, input_sizes, dim, index):
    return torch.ops.aten.select_backward(grad_output, input_sizes, dim, index)
