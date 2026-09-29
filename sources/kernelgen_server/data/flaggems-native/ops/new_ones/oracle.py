import torch

REFERENCE_DEVICE = "target"


def run(self, size, *, dtype=None, layout=None, device=None, pin_memory=None):
    return self.new_ones(size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory)
