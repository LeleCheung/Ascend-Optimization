REFERENCE_DEVICE = 'target'

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Simple model that performs an ELU activation.
    """
    def __init__(self, alpha: float = 1.0):
        """
        Initializes the ELU model.

        Args:
            alpha (float, optional): The alpha parameter for the ELU function. Defaults to 1.0.
        """
        super(Model, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ELU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ELU applied, same shape as input.
        """
        return F.elu(x, alpha=self.alpha)

batch_size = 4096
dim = 393216


_reference_model = None


def run(x):
    global _reference_model
    if _reference_model is None:
        torch.manual_seed(42)
        _reference_model = Model(*([1.0])).to(
            device=x.device, dtype=torch.float32
        )
    with torch.no_grad():
        return _reference_model(x)
