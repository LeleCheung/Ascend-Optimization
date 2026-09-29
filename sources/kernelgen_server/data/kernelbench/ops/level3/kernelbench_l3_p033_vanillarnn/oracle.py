REFERENCE_DEVICE = 'target'

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize the Vanilla RNN model.

        :param input_size: The number of input features (int).
        :param hidden_size: The size of the hidden state (int).
        :param output_size: The number of output features (int).
        """
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden = torch.randn((batch_size, hidden_size))

        # Define the RNN cell components (input to hidden, hidden to hidden, and hidden to output)
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)  # Input to hidden
        self.h2o = nn.Linear(hidden_size, output_size)  # Hidden to output
        self.tanh = nn.Tanh()  # Activation function for hidden state

    def forward(self, x: torch.Tensor, initial_hidden=None) -> torch.Tensor:
        """
        Forward pass of the Vanilla RNN.

        :param x: Input tensor of shape (batch_size, input_size).
        :param hidden: Hidden state tensor of shape (batch_size, hidden_size).
        :return: Output tensor of shape (batch_size, output_size), and the new hidden state.
        """
        if initial_hidden is not None:
            self.hidden.copy_(initial_hidden)
        self.hidden = self.hidden.to(x.device)
        combined = torch.cat((x, self.hidden), dim=1)  # Concatenate input and hidden state
        self.hidden = self.tanh(self.i2h(combined))  # Update hidden state
        output = self.h2o(self.hidden)  # Compute output
        return output

batch_size = 256
input_size = 16384
hidden_size = 16384
output_size = 8192
sequence_length = 256


_reference_model = None


def run(x, initial_hidden):
    global _reference_model
    if _reference_model is None:
        torch.manual_seed(42)
        _reference_model = Model(*([input_size, hidden_size, output_size])).to(
            device=x.device, dtype=torch.float32
        )
    with torch.no_grad():
        return _reference_model(x, initial_hidden)
