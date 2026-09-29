import torch

REFERENCE_DEVICE = "target"


def run(self, other):
    return self.lcm_(other)


def gen_inputs(ctx, device):
    case = ctx["inputs"]["case"]
    phase = case["phase"]
    dtype = getattr(torch, case["dtype"])
    seed = ctx["seed"]

    if phase == "correctness":
        shape = tuple(case["shape"])
        # Source: torch.randint(1, 100, shape, dtype=dtype, device=flag_gems.device)
        try:
            gen = torch.Generator(device=device)
        except Exception:
            gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        factory_device = gen.device
        inp1 = torch.randint(1, 100, shape, dtype=dtype, device=factory_device, generator=gen)
        inp2 = torch.randint(1, 100, shape, dtype=dtype, device=factory_device, generator=gen)
        if factory_device != device:
            inp1 = inp1.to(device)
            inp2 = inp2.to(device)
        return {"self": inp1, "other": inp2}

    # phase == "timing"
    # Source: torch.randint(torch.iinfo(dtype).min, torch.iinfo(dtype).max,
    #                       shape, dtype=dtype, device="cpu").to(device)
    shapes = case["shape"]
    shape0 = tuple(shapes[0])
    shape1 = tuple(shapes[1])
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    inp1 = torch.randint(
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        shape0,
        dtype=dtype,
        device="cpu",
        generator=gen,
    )
    inp2 = torch.randint(
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        shape1,
        dtype=dtype,
        device="cpu",
        generator=gen,
    )
    inp1 = inp1.to(device)
    inp2 = inp2.to(device)
    return {"self": inp1, "other": inp2}
