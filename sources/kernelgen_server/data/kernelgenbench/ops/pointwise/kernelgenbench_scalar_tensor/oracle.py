REFERENCE_DEVICE = 'target'

import torch

def _current_device():
    for attr, device_type in (("npu", "npu"), ("musa", "musa"), ("mlu", "mlu")):
        runtime = getattr(torch, attr, None)
        if runtime is not None and runtime.is_available():
            return torch.device(device_type, runtime.current_device())
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")

def run(value, dtype, layout, pin_memory):
    dtype = getattr(torch, dtype) if dtype else None
    layout = getattr(torch, layout) if layout else None
    return torch.scalar_tensor(value, dtype=dtype, layout=layout, pin_memory=pin_memory, device=_current_device())
