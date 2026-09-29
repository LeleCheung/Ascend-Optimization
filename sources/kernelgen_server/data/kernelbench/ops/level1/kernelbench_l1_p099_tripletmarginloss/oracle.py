REFERENCE_DEVICE = 'target'

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes Triplet Margin Loss for metric learning tasks.

    Parameters:
        margin (float): The margin between the positive and negative samples.
    """
    def __init__(self, margin=1.0):
        super(Model, self).__init__()
        self.loss_fn = torch.nn.TripletMarginLoss(margin=margin)

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)

batch_size = 32768
input_shape = (8192,)
dim = 1


_reference_model = None


def run(anchor, positive, negative):
    global _reference_model
    if _reference_model is None:
        torch.manual_seed(42)
        _reference_model = Model(*([1.0])).to(
            device=anchor.device, dtype=torch.float32
        )
    with torch.no_grad():
        return _reference_model(anchor, positive, negative)
