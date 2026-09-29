REFERENCE_DEVICE = 'target'

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a matrix multiplication (Gemm), applies Sigmoid,
    another Gemm, and computes LogSumExp over features.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.logsumexp(x, dim=1)  # compute LogSumExp over features per sample
        return x

batch_size = 16384
input_size = 2048
hidden_size = 4096
output_size = 1024


_reference_model = None


def run(x):
    global _reference_model
    if _reference_model is None:
        torch.manual_seed(42)
        _reference_model = Model(*([input_size, hidden_size, output_size])).to(
            device=x.device, dtype=torch.float32
        )
    with torch.no_grad():
        return _reference_model(x)
