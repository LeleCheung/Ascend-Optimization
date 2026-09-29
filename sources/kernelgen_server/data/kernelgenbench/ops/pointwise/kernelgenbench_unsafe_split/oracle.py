REFERENCE_DEVICE = 'target'

import torch

def run(x, split_size, dim):
    return torch.ops.aten.unsafe_split.Tensor(x, split_size, dim)
