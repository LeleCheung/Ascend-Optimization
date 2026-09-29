import torch

REFERENCE_DEVICE = "target"


def gen_inputs(ctx, device):
    case = ctx["inputs"]["case"]
    phase = case["phase"]
    dtype = getattr(torch, case["dtype"])
    inp_shape = case["inp_shape"]
    dim = case["params"]["dim"]

    try:
        gen = torch.Generator(device=device)
    except Exception:
        gen = torch.Generator(device="cpu")
    gen.manual_seed(ctx["seed"])
    factory_device = gen.device

    inp = torch.randn(inp_shape, dtype=dtype, device=factory_device, generator=gen)

    if phase == "correctness":
        index_len = inp_shape[dim]
    else:
        index_len = inp_shape[dim] // 2 if inp_shape[dim] >= 2 else 1

    index = torch.randperm(index_len, device=factory_device, generator=gen)

    src_shape = list(inp_shape)
    src_shape[dim] = index_len
    src = torch.randn(src_shape, dtype=dtype, device=factory_device, generator=gen)

    if factory_device != device:
        inp = inp.to(device)
        index = index.to(device)
        src = src.to(device)

    return {"inp": inp, "dim": dim, "index": index, "src": src}


def run(inp, dim, index, src):
    return torch.index_copy(inp, dim, index, src)
