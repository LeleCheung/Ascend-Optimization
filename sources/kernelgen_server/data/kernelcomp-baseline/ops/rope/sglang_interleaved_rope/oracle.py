REFERENCE_DEVICE = 'target'

import torch
def run(x, mrope_section):
    _, S, D = x.shape
    d = torch.arange(D, device=x.device)
    cond_a = (d % 3 == 1) & (d < mrope_section[1] * 3)
    cond_b = (d % 3 == 2) & (d < mrope_section[2] * 3)

    out = x[0].clone()
    out[:, cond_a] = x[1][:, cond_a]
    out[:, cond_b] = x[2][:, cond_b]
    return out
