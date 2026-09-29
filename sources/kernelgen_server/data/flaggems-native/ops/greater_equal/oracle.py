import torch

REFERENCE_DEVICE = "target"


def run(A, B):
    return torch.greater_equal(A, B)
