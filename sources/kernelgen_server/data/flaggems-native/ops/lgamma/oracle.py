import torch

REFERENCE_DEVICE = "target"


def run(A):
    return torch.lgamma(A)


def gen_inputs(ctx, device):
    case = ctx["inputs"].get("case")
    if case is None:
        return None

    dtype = getattr(torch, case["dtype"])
    shape = tuple(case["shape"])

    try:
        gen = torch.Generator(device=device)
    except Exception:
        gen = torch.Generator(device="cpu")
    gen.manual_seed(ctx["seed"])
    factory_device = gen.device
    inp = torch.rand(shape, dtype=dtype, device=factory_device, generator=gen) + 0.1
    if factory_device != device:
        inp = inp.to(device)

    return {"A": inp}
