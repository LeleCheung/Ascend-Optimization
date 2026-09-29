REFERENCE_DEVICE = 'target'

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a matrix multiplication, group normalization, leaky ReLU activation, and element-wise sum.
    """
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x):
        """
        Performs the forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, hidden_size).
        """
        x = self.fc(x)
        x = self.gn(x)
        x = self.leaky_relu(x)
        x = x + x
        return x


batch_size = 1024
input_size = 8192
hidden_size = 8192
num_groups = 512


_reference_model = None


def run(x):
    global _reference_model
    if _reference_model is None:
        torch.manual_seed(42)
        _reference_model = Model(*([input_size, hidden_size, num_groups])).to(
            device=x.device, dtype=torch.float32
        )
    with torch.no_grad():
        return _reference_model(x)
