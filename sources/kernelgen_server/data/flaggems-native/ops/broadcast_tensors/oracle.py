import torch

REFERENCE_DEVICE = "target"


def run(*tensors):
    return torch.broadcast_tensors(*tensors)


def gen_inputs(ctx, device):
    case = ctx["inputs"]["case"]
    dtype = getattr(torch, case["dtype"])
    shapes = [tuple(shape) for shape in case["shape"]["inputs"]]
    seed = ctx["seed"]
    try:
        gen = torch.Generator(device=device).manual_seed(seed)
        gen_device = device
    except (RuntimeError, ValueError):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        gen_device = "cpu"
    tensors = tuple(
        torch.randn(shape, dtype=dtype, device=gen_device, generator=gen)
        for shape in shapes
    )
    if gen_device != device:
        tensors = tuple(tensor.to(device) for tensor in tensors)
    return {"tensors": tensors}
