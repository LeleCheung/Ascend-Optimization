REFERENCE_DEVICE = 'target'

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes Kullback-Leibler Divergence for comparing two distributions.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        return torch.nn.functional.kl_div(torch.log(predictions), targets, reduction='batchmean')

batch_size = 8192 * 2
input_shape = (8192 * 2,)
dim = 1


_reference_model = None


def run(predictions, targets):
    global _reference_model
    if _reference_model is None:
        torch.manual_seed(42)
        _reference_model = Model(*([])).to(
            device=predictions.device, dtype=torch.float32
        )
    with torch.no_grad():
        return _reference_model(predictions, targets)
