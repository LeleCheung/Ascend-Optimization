import torch

REFERENCE_DEVICE = "target"


def run(x, p, dim, maxnorm):
    return torch.ops.aten.renorm_(x, p, dim, maxnorm)
