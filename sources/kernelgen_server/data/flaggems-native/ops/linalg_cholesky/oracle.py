import torch

REFERENCE_DEVICE = "target"


def run(A, upper=False):
    return torch.linalg.cholesky(A, upper=upper)


def gen_inputs(ctx, device):
    case = ctx["inputs"].get("case")
    if case is None:
        return None

    dtype = getattr(torch, case["dtype"])
    shape = tuple(case["shape"])
    n = shape[-1]
    seed = ctx["seed"]

    try:
        generator = torch.Generator(device=device)
    except Exception:
        generator = torch.Generator(device="cpu")
    factory_device = generator.device
    generator.manual_seed(seed)

    B = torch.randn(shape, dtype=dtype, device=factory_device, generator=generator)
    if str(factory_device) != str(device):
        B = B.to(device)

    A = B @ B.transpose(-2, -1) + torch.eye(n, dtype=dtype, device=device) * 0.1

    return {"A": A}
