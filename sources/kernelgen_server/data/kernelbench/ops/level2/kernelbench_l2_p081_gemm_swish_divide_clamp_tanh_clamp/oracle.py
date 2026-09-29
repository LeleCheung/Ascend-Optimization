REFERENCE_DEVICE = 'target'

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a gemm, swish, divide, clamp, tanh, and clamp operations.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.gemm(x)
        x = x * torch.sigmoid(x)  # Swish activation
        x = x / 2.0
        x = torch.clamp(x, min=-1.0, max=1.0)  # Clamp between -1 and 1
        x = torch.tanh(x)  # Tanh activation
        x = torch.clamp(x, min=-1.0, max=1.0)  # Clamp between -1 and 1
        return x

batch_size = 1024
in_features = 8192
out_features = 8192


_reference_model = None


def run(x):
    global _reference_model
    if _reference_model is None:
        torch.manual_seed(42)
        _reference_model = Model(*([in_features, out_features])).to(
            device=x.device, dtype=torch.float32
        )
    with torch.no_grad():
        return _reference_model(x)
