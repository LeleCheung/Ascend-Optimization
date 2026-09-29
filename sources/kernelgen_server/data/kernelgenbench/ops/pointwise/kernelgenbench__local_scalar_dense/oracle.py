REFERENCE_DEVICE = 'target'

import torch

def run(x):
    return torch.ops.aten._local_scalar_dense(x)
