REFERENCE_DEVICE = 'target'

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes Cross Entropy Loss for multi-class classification tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        return torch.nn.functional.cross_entropy(predictions, targets)

batch_size = 32768
num_classes = 4096
input_shape = (num_classes,)
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
