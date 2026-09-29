REFERENCE_DEVICE = 'target'

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs an exclusive cumulative sum (does not include the current element).

    Parameters:
        dim (int): The dimension along which to perform the exclusive cumulative sum.
    """

    def __init__(self, dim):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x):
        cumsum = torch.cumsum(x.narrow(dim=self.dim, start=0, length=x.size(self.dim)-1), dim=self.dim)
        return torch.cat((torch.zeros_like(x.select(self.dim, 0).unsqueeze(self.dim)), cumsum), dim=self.dim)

batch_size = 32768
input_shape = (32768,)
dim = 1


_reference_model = None


def run(x):
    global _reference_model
    if _reference_model is None:
        torch.manual_seed(42)
        _reference_model = Model(*([dim])).to(
            device=x.device, dtype=torch.float32
        )
    with torch.no_grad():
        return _reference_model(x)
