import torch

REFERENCE_DEVICE = "target"


def run(inp, dim, index, src):
    inp.index_copy_(dim, index, src)
    return inp


def gen_inputs(ctx, device):
    case = ctx["inputs"]["case"]
    dtype = getattr(torch, case["dtype"])
    inp_shape = case["inp_shape"]
    dim = case["dim"]
    index_len = case["index_len"]
    seed = ctx["seed"]

    try:
        generator = torch.Generator(device=device)
    except Exception:
        generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    factory_device = generator.device

    inp = torch.randn(inp_shape, dtype=dtype, device=factory_device, generator=generator)
    index = torch.randperm(index_len, device=factory_device, generator=generator)
    src_shape = list(inp_shape)
    src_shape[dim] = index_len
    src = torch.randn(src_shape, dtype=dtype, device=factory_device, generator=generator)

    if factory_device.type == "cpu":
        inp = inp.to(device)
        index = index.to(device)
        src = src.to(device)

    return {
        "inp": inp,
        "dim": dim,
        "index": index,
        "src": src,
    }
