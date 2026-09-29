REFERENCE_DEVICE = "target"

import torch


def run(A, B):
    return A.greater_equal_(B)
