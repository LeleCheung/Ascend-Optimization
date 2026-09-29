import torch

REFERENCE_DEVICE = "target"


def run(x, n):
    return torch.special.chebyshev_polynomial_u(x, n)


def gen_inputs(ctx, device):
    case = ctx["inputs"].get("case")
    if case is None:
        return None
    shape = tuple(case["shape"])
    try:
        generator = torch.Generator(device=device)
        factory_device = device
    except Exception:
        generator = torch.Generator(device="cpu")
        factory_device = "cpu"
    generator.manual_seed(ctx.get("seed", 0))
    n = torch.randint(
        0, 6, shape, dtype=torch.int32, device=factory_device, generator=generator
    )
    x = torch.randn(
        shape, dtype=torch.float32, device=factory_device, generator=generator
    )
    if factory_device != device:
        n = n.to(device)
        x = x.to(device)
    return {"x": x, "n": n}
