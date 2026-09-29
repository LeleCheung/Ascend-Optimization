REFERENCE_DEVICE = 'target'

import torch
import torch.nn.functional as F
def run(x, weight, bias, num_groups, eps):
    y = F.group_norm(x.float(), num_groups, weight.float(), bias.float(), eps)
    return F.silu(y).to(x.dtype)
