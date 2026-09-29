REFERENCE_DEVICE = "target"

import torch


def gen_inputs(ctx, device):
    device_type = torch.device(device).type
    if device_type == "musa":
        torch.backends.mudnn.allow_tf32 = False
    elif device_type == "npu":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        try:
            torch_backend_device = getattr(torch.backends, device_type)
            torch_backend_device.matmul.allow_tf32 = False
        except Exception:
            pass

    if "case" not in ctx["inputs"]:
        return None

    case = ctx["inputs"]["case"]
    shape = tuple(case["shape"])
    dtype = getattr(torch, case["dtype"])
    p = case["params"]["p"]
    seed = ctx["seed"]

    try:
        gen = torch.Generator(device=device)
    except Exception:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        factory_device = gen.device
        inp = torch.rand(shape, dtype=dtype, generator=gen, device=factory_device)
        inp = inp.to(device)
        inp = inp + (p - 1) / 2 + 1.0
        return {"self": inp, "p": p}

    gen.manual_seed(seed)
    factory_device = gen.device
    inp = torch.rand(shape, dtype=dtype, generator=gen, device=factory_device)
    inp = inp + (p - 1) / 2 + 1.0
    return {"self": inp, "p": p}


def run(self, p):
    return self.mvlgamma_(p)
