REFERENCE_DEVICE = 'target'

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs 4D tensor-matrix multiplication:
        C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]

    Args:
        A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
        B (torch.Tensor): Input matrix of shape (l, k)

    Returns:
        torch.Tensor: Output 4D tensor of shape (b, i, j, k)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A, B):
        """
        Performs the 4D tensor-matrix multiplication.

        Args:
            A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
            B (torch.Tensor): Input matrix of shape (l, k)

        Returns:
            torch.Tensor: Output 4D tensor of shape (b, i, j, k)
        """
        return torch.einsum("bijl,lk->bijk", A, B)

# Test code
b = 8
i = 256
j = 512
l = 256
k = 768


_reference_model = None


def run(A, B):
    global _reference_model
    if _reference_model is None:
        torch.manual_seed(42)
        _reference_model = Model(*([])).to(
            device=A.device, dtype=torch.float32
        )
    with torch.no_grad():
        return _reference_model(A, B)
