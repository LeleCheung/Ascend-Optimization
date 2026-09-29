REFERENCE_DEVICE = 'target'

import torch
def run(x, dtype, layout, pin_memory, memory_format):
    dtype = getattr(torch, dtype) if dtype else None
    layout = getattr(torch, layout) if layout else None
    memory_format = getattr(torch, memory_format) if memory_format else None
    return torch.zeros_like(x, dtype=dtype, layout=layout,
                      pin_memory=pin_memory, memory_format=memory_format)
