import torch

REFERENCE_DEVICE = "target"


def run(inp, k, dim=-1, keepdim=False):
    return torch.kthvalue(inp, k, dim=dim, keepdim=keepdim)
