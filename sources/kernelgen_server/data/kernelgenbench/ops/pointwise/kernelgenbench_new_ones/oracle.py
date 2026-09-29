REFERENCE_DEVICE = 'target'

import torch
def run(x, size, dtype, layout, pin_memory, memory_format):
    dtype = getattr(torch, dtype) if dtype else None
    layout = getattr(torch, layout) if layout else None
    return x.new_ones(size, dtype=dtype, layout=layout, pin_memory=pin_memory)
