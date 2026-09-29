REFERENCE_DEVICE = 'target'

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes Smooth L1 (Huber) Loss for regression tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        return torch.nn.functional.smooth_l1_loss(predictions, targets)

batch_size = 32768
input_shape = (32768,)
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
