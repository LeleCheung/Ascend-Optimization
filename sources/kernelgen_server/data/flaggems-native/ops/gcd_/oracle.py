import torch

REFERENCE_DEVICE = "target"


def run(A, B):
    return A.gcd_(B)


def gen_inputs(ctx, device):
    case = ctx["inputs"]["case"]
    phase = case["phase"]
    dtype = getattr(torch, case["dtype"])
    shapes = case["shape"]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(ctx["seed"])
    if phase == "correctness":
        A = torch.randint(
            low=1,
            high=100,
            size=shapes[0],
            dtype=dtype,
            device="cpu",
            generator=generator,
        ).to(device)
        B = torch.randint(
            low=1,
            high=100,
            size=shapes[1],
            dtype=dtype,
            device="cpu",
            generator=generator,
        ).to(device)
        return {"A": A, "B": B}
    A = torch.randint(
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        size=shapes[0],
        dtype=dtype,
        device="cpu",
        generator=generator,
    ).to(device)
    B = torch.randint(
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        size=shapes[1],
        dtype=dtype,
        device="cpu",
        generator=generator,
    ).to(device)
    return {"A": A, "B": B}
