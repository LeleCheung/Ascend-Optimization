REFERENCE_DEVICE = 'target'

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Simple model that performs a HardTanh activation.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies HardTanh activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with HardTanh applied, same shape as input.
        """
        return F.hardtanh(x, min_val=-1., max_val=1.)

batch_size = 4096
dim = 393216


_reference_model = None


def run(x):
    global _reference_model
    if _reference_model is None:
        torch.manual_seed(42)
        _reference_model = Model(*([])).to(
            device=x.device, dtype=torch.float32
        )
    with torch.no_grad():
        return _reference_model(x)
