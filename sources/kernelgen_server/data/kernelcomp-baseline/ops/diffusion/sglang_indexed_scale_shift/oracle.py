REFERENCE_DEVICE = 'target'

import torch
def run(x, shift, scale, indices):
    idx = indices.long()
    sh = shift[idx].float()
    sc = scale[idx].float()
    one_plus = (1.0 + sc).to(torch.bfloat16).float()
    scaled = (x.float() * one_plus).to(torch.bfloat16).float()
    return (scaled + sh).to(x.dtype)
