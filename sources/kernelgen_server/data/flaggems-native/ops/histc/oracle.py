import torch

REFERENCE_DEVICE = "target"


def run(inp, bins=100, min=0, max=0):
    return torch.histc(inp, bins=bins, min=min, max=max)


def gen_inputs(ctx, device):
    case = ctx["inputs"]["case"]
    phase = case["phase"]
    dtype = getattr(torch, case["dtype"])

    if phase == "timing":
        shape = case["shape"]
        bins = case["bins"]
        min_val = case["min"]
        max_val = case["max"]

        try:
            generator = torch.Generator(device=device)
        except Exception:
            generator = torch.Generator(device="cpu")
        factory_device = generator.device
        generator.manual_seed(ctx["seed"])
        inp = torch.rand(shape, dtype=dtype, device=factory_device, generator=generator) * 10
        if factory_device != device:
            inp = inp.to(device)
        return {"inp": inp, "bins": bins, "min": min_val, "max": max_val}

    # correctness phase
    shape = tuple(case["shape"])
    bins = case["bins"]
    min_val = float(case["min_val"])
    max_val = float(case["max_val"])
    include_endpoints = case.get("include_endpoints", False)
    include_outliers = case.get("include_outliers", False)
    histc_min = case["histc_min"]
    histc_max = case["histc_max"]

    numel = 1
    for dim in shape:
        numel *= dim

    bucket_ids = torch.arange(numel, device=device) % 100
    inp = min_val + (bucket_ids.to(dtype) + 0.5) * ((max_val - min_val) / 100)
    inp = inp.reshape(shape)

    if include_endpoints and numel >= 2:
        flat = inp.reshape(-1)
        flat[0] = min_val
        flat[1] = max_val
    elif include_outliers and numel >= 2:
        flat = inp.reshape(-1)
        flat[0] = min_val - 0.5
        flat[1] = max_val + 0.5

    return {"inp": inp, "bins": bins, "min": histc_min, "max": histc_max}
