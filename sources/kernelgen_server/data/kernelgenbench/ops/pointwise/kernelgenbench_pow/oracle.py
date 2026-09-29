REFERENCE_DEVICE = 'target'

import torch
def run(x, exponent):
    return torch.ops.aten.pow.Tensor_Scalar(x, exponent)
