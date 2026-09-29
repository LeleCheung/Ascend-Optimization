REFERENCE_DEVICE = 'target'

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a transposed convolution, applies Mish activation, adds a value,
    applies Hardtanh activation, and scales the output.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale

    def forward(self, x):
        x = self.conv_transpose(x)
        x = torch.nn.functional.mish(x) # Mish activation
        x = x + self.add_value
        x = torch.nn.functional.hardtanh(x, min_val=-1, max_val=1) # Hardtanh activation
        x = x * self.scale # Scaling
        return x

batch_size = 128
in_channels  = 64
out_channels = 64
height = width = 128
kernel_size  = 3
stride       = 2
padding      = 1
output_padding = 1
add_value = 0.5
scale = 2


_reference_model = None


def run(x):
    global _reference_model
    if _reference_model is None:
        torch.manual_seed(42)
        _reference_model = Model(*([in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale])).to(
            device=x.device, dtype=torch.float32
        )
    with torch.no_grad():
        return _reference_model(x)
