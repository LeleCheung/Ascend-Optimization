REFERENCE_DEVICE = "target"

import torch


def run(self, other):
    return torch.lcm(self, other)


def gen_inputs(ctx, device):
    case = ctx["inputs"]["case"]
    phase = case["phase"]
    dtype = getattr(torch, case["dtype"])
    seed = ctx["seed"]

    if phase == "correctness":
        shape = tuple(case["shape"])
        try:
            gen = torch.Generator(device=device)
        except Exception:
            gen = torch.Generator(device="cpu")
        factory_device = gen.device
        gen.manual_seed(seed)
        inp1 = torch.randint(1, 100, shape, dtype=dtype, generator=gen, device=factory_device)
        inp2 = torch.randint(1, 100, shape, dtype=dtype, generator=gen, device=factory_device)
        inp1 = inp1.to(device)
        inp2 = inp2.to(device)
        return {"self": inp1, "other": inp2}

    # timing phase: source uses full dtype range allocated on CPU then transferred to device
    shapes = case["shape"]
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    inp1 = torch.randint(
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        tuple(shapes[0]),
        dtype=dtype,
        generator=gen,
        device="cpu",
    ).to(device)
    inp2 = torch.randint(
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        tuple(shapes[1]),
        dtype=dtype,
        generator=gen,
        device="cpu",
    ).to(device)
    return {"self": inp1, "other": inp2}
