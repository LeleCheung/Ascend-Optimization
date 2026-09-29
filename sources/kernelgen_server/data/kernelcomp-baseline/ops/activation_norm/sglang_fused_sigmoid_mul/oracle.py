REFERENCE_DEVICE = 'target'

import torch
def run(attn_output, gate):
    g = gate.reshape(attn_output.shape).float()
    out = attn_output.float() * torch.sigmoid(g)
    return out.to(attn_output.dtype)
