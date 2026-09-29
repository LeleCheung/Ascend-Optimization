REFERENCE_DEVICE = 'target'

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, scales the result, and applies batch normalization.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x):
        x = self.gemm(x)
        x = x * self.scale
        x = self.bn(x)
        return x

batch_size = 16384
in_features = 4096
out_features = 4096
scale_shape = (out_features,)


_reference_model = None


def run(x):
    global _reference_model
    if _reference_model is None:
        torch.manual_seed(42)
        _reference_model = Model(*([in_features, out_features, scale_shape])).to(
            device=x.device, dtype=torch.float32
        )
    with torch.no_grad():
        return _reference_model(x)
