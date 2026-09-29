REFERENCE_DEVICE = 'target'

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a matrix multiplication, subtraction, multiplication, and ReLU activation.
    """
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

    def forward(self, x):
        x = self.linear(x)
        x = x - self.subtract_value
        x = x * self.multiply_value
        x = torch.relu(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
subtract_value = 2.0
multiply_value = 1.5


_reference_model = None


def run(x):
    global _reference_model
    if _reference_model is None:
        torch.manual_seed(42)
        _reference_model = Model(*([in_features, out_features, subtract_value, multiply_value])).to(
            device=x.device, dtype=torch.float32
        )
    with torch.no_grad():
        return _reference_model(x)
