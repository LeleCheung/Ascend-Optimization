REFERENCE_DEVICE = 'target'

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param hidden_layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(Model, self).__init__()

        layers = []
        current_input_size = input_size

        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(nn.ReLU())
            current_input_size = hidden_size

        layers.append(nn.Linear(current_input_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        return self.network(x)

# Test code
batch_size = 128
input_size = 16384
hidden_layer_sizes = [32768, 32768]
output_size = 16384


_reference_model = None


def run(x):
    global _reference_model
    if _reference_model is None:
        torch.manual_seed(42)
        _reference_model = Model(*([input_size, hidden_layer_sizes, output_size])).to(
            device=x.device, dtype=torch.float32
        )
    with torch.no_grad():
        return _reference_model(x)
