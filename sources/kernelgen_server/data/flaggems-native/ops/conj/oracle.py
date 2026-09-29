import torch

REFERENCE_DEVICE = "target"


def run(input):
    return torch._conj(input)


def gen_inputs(ctx, device):
    case = ctx["inputs"].get("case")
    if case is None or case.get("phase") != "correctness":
        return None
    shape = tuple(case["shape"])
    dtype = getattr(torch, case["dtype"])
    try:
        gen = torch.Generator(device=device)
        gen.manual_seed(ctx["seed"])
    except (RuntimeError, ValueError):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(ctx["seed"])
        real = torch.randn(shape, dtype=torch.float32, generator=gen, device="cpu")
        imag = torch.randn(shape, dtype=torch.float32, generator=gen, device="cpu")
        inp = torch.complex(real, imag).to(dtype)
        return {"input": inp.to(device)}
    real = torch.randn(shape, dtype=torch.float32, generator=gen, device=device)
    imag = torch.randn(shape, dtype=torch.float32, generator=gen, device=device)
    inp = torch.complex(real, imag).to(dtype)
    return {"input": inp}
