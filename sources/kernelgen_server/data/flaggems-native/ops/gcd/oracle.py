import torch

REFERENCE_DEVICE = "target"


def run(self, other, *, out=None):
    return torch.gcd(self, other, out=out)


def gen_inputs(ctx, device):
    case = ctx["inputs"]["case"]
    seed = ctx["seed"]
    dtype = getattr(torch, case["dtype"])
    phase = case["phase"]
    info = torch.iinfo(dtype)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    def _make_gcd_tensor(shape, values):
        t = torch.randint(
            info.min, info.max, shape,
            dtype=dtype, device="cpu", generator=gen,
        ).to(device)
        flat = t.reshape(-1)
        if flat.numel() > 0:
            bnd = torch.tensor(values, dtype=dtype, device=device)
            flat[: min(flat.numel(), bnd.numel())] = bnd[: flat.numel()]
        return t

    if phase == "timing":
        shapes = case["shape"]
        shape1 = tuple(shapes[0])
        shape2 = tuple(shapes[1])
        inp1 = torch.randint(
            info.min, info.max, shape1,
            dtype=dtype, device="cpu", generator=gen,
        ).to(device)
        inp2 = torch.randint(
            info.min, info.max, shape2,
            dtype=dtype, device="cpu", generator=gen,
        ).to(device)
        return {"self": inp1, "other": inp2}

    test_fn = case["test_fn"]

    if test_fn == "test_gcd":
        shape = tuple(case["shape"])
        inp1 = _make_gcd_tensor(shape, [0, -12, info.min, -27, 81])
        inp2 = _make_gcd_tensor(shape, [6, 18, 0, -9, info.min])
        return {"self": inp1, "other": inp2}

    if test_fn == "test_gcd_special_values":
        inp1 = torch.tensor(
            [0, 0, -12, -27, info.min, info.min, info.min, info.max],
            dtype=dtype, device=device,
        )
        inp2 = torch.tensor(
            [0, 6, 18, -9, 0, info.min, 2, info.min],
            dtype=dtype, device=device,
        )
        return {"self": inp1, "other": inp2}

    if test_fn == "test_gcd_empty":
        inp1 = torch.empty((2, 0, 3), dtype=dtype, device=device)
        inp2 = torch.empty((2, 0, 3), dtype=dtype, device=device)
        return {"self": inp1, "other": inp2}

    if test_fn == "test_gcd_noncontiguous_broadcast":
        lhs = _make_gcd_tensor((5, 7), [0, -12, info.min, -27, 81]).T
        rhs = _make_gcd_tensor((1, 5), [6, 18, 0, -9, info.min])
        return {"self": lhs, "other": rhs}

    return None
