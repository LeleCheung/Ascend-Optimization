REFERENCE_DEVICE = 'target'

import torch
def run(x, softcap_const):
    return torch.tanh(x.to(torch.float32) / softcap_const) * softcap_const
