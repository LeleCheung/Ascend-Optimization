REFERENCE_DEVICE = 'target'

import torch
def run(x, n, dim, prepend, append):
    return torch.diff(x, n=n, dim=dim, prepend=prepend, append=append)
