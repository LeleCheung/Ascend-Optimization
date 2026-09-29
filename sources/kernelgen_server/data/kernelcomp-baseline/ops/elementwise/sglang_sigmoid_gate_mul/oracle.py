REFERENCE_DEVICE = 'target'

import torch
def run(x, gate):
    return (x.float() * torch.sigmoid(gate.float())).to(x.dtype)
