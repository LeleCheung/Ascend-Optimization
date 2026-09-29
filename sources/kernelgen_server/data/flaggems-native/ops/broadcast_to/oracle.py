import torch

REFERENCE_DEVICE = "target"


def run(x, size):
    return torch.broadcast_to(x, size)
