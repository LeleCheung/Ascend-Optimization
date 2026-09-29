REFERENCE_DEVICE = 'target'

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a matrix multiplication (Gemm), followed by LogSumExp, LeakyReLU,
    LeakyReLU, GELU, and GELU activations.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        # Gemm
        x = self.linear(x)
        # LogSumExp
        x = torch.logsumexp(x, dim=1, keepdim=True)
        # LeakyReLU
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        # LeakyReLU
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        # GELU
        x = torch.nn.functional.gelu(x)
        # GELU
        x = torch.nn.functional.gelu(x)
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
