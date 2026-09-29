REFERENCE_DEVICE = 'target'

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs matrix multiplication, applies dropout, and then applies softmax.
    """
    def __init__(self, in_features, out_features, dropout_p):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.matmul(x)
        x = self.dropout(x)
        x = torch.softmax(x, dim=1)  # Softmax over features
        return x

batch_size = 128
in_features = 16384
out_features = 16384
dropout_p = 0.2


_reference_model = None


def run(x):
    global _reference_model
    if _reference_model is None:
        torch.manual_seed(42)
        _reference_model = Model(*([in_features, out_features, dropout_p])).to(
            device=x.device, dtype=torch.float32
        )
    with torch.no_grad():
        return _reference_model(x)
