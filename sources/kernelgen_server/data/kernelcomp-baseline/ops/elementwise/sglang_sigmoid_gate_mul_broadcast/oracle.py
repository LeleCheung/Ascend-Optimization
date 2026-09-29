REFERENCE_DEVICE = 'target'

import torch
def run(x, gate):
    g = torch.sigmoid(gate.reshape(-1, 1).float())
    return (x.float() * g).to(x.dtype)
