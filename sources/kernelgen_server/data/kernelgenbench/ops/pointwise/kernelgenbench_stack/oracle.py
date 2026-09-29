REFERENCE_DEVICE = 'target'

import torch
def run(x0, x1, x2, dim):
    values = [x0, x1] if x2 is None else [x0, x1, x2]
    return torch.ops.aten.stack(values, dim)
