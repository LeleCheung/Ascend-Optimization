REFERENCE_DEVICE = 'target'

import torch

def run(x, value):
    return torch.ops.aten.fill_.Scalar(x, value)
