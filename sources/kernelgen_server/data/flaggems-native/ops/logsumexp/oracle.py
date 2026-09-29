import torch

REFERENCE_DEVICE = "target"


def run(inp, dim, keepdim=False):
    return torch.logsumexp(inp, dim=dim, keepdim=keepdim)
