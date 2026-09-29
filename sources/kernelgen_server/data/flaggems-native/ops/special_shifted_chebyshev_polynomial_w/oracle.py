import torch

REFERENCE_DEVICE = "target"


def correctness_run(x, n):
    ref_x = x.to(torch.float64)
    ref_n = n.to(torch.float64)
    ref_out = torch.special.shifted_chebyshev_polynomial_w(ref_x, ref_n)
    return ref_out.to(torch.float32)


def timing_run(x, n):
    return torch.special.shifted_chebyshev_polynomial_w(x, n)


def gen_inputs(ctx, device):
    device_type = torch.device(device).type
    if device_type == "musa":
        torch.backends.mudnn.allow_tf32 = False
    elif device_type == "npu":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        try:
            getattr(torch.backends, device_type).matmul.allow_tf32 = False
        except Exception:
            pass

    case = ctx["inputs"].get("case")
    if case is None:
        return None

    seed = ctx["seed"]
    shape = tuple(case["shape"])
    dtype = torch.float32
    try:
        generator = torch.Generator(device=device)
    except Exception:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        factory_device = generator.device
        x = torch.randn(shape, dtype=dtype, device=factory_device, generator=generator).to(device)
        n = torch.randint(0, 11, shape, dtype=torch.int32, device=factory_device, generator=generator).to(device)
        return {"x": x, "n": n}
    generator.manual_seed(seed)
    factory_device = generator.device
    x = torch.randn(shape, dtype=dtype, device=factory_device, generator=generator)
    n = torch.randint(0, 11, shape, dtype=torch.int32, device=factory_device, generator=generator)
    return {"x": x, "n": n}
