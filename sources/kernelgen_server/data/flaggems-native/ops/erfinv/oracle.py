import torch

REFERENCE_DEVICE = "target"

_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
}


def run(x):
    return torch.erfinv(x)


def gen_inputs(ctx, device):
    case = ctx["inputs"].get("case")
    if case is None or case.get("phase") != "correctness":
        return None
    shape = tuple(case["shape"])
    dtype = _DTYPES[case["dtype"]]
    try:
        gen = torch.Generator(device=device).manual_seed(ctx["seed"])
        on_device = True
    except Exception:
        gen = torch.Generator(device="cpu").manual_seed(ctx["seed"])
        on_device = False
    if on_device:
        x = torch.empty(shape, dtype=dtype, device=device)
        x.uniform_(-0.99, 0.99, generator=gen)
    else:
        x = torch.empty(shape, dtype=dtype, device="cpu")
        x.uniform_(-0.99, 0.99, generator=gen)
        x = x.to(device)
    return {"x": x}
