import torch


REFERENCE_DEVICE = "target"


def _flat_to_per_dim_indices(flat_indices, inp_shape):
    strides = []
    prod = 1
    for s in reversed(inp_shape):
        strides.append(prod)
        prod *= s
    strides = tuple(reversed(strides))
    result = []
    for i in range(len(inp_shape)):
        result.append((flat_indices // strides[i]) % inp_shape[i])
    return tuple(result)


def run(inp, mask, indices, values):
    return torch._unsafe_masked_index_put_accumulate(inp, mask, indices, values)


def gen_inputs(ctx, device):
    case = ctx["inputs"]["case"]
    dtype = getattr(torch, case["dtype"])
    inp_shape = tuple(case["shape"]["input"])
    mask_shape = tuple(case["shape"]["mask"])
    values_shape = tuple(case["shape"]["values"])

    numel = 1
    for s in inp_shape:
        numel *= s

    try:
        generator = torch.Generator(device=device)
    except Exception:
        generator = torch.Generator(device="cpu")
    factory_device = generator.device
    generator.manual_seed(ctx["seed"])

    inp = torch.randn(inp_shape, dtype=dtype, device=factory_device, generator=generator)
    mask = torch.randint(
        0, 2, mask_shape, dtype=torch.int32, device=factory_device, generator=generator
    )
    flat_indices = torch.randint(
        0, max(numel, 1), mask_shape, device=factory_device, generator=generator
    )
    values = torch.randn(
        values_shape, dtype=dtype, device=factory_device, generator=generator
    )

    if factory_device != device:
        inp = inp.to(device)
        mask = mask.to(device)
        flat_indices = flat_indices.to(device)
        values = values.to(device)

    indices = _flat_to_per_dim_indices(flat_indices, inp_shape)
    return {"inp": inp, "mask": mask, "indices": indices, "values": values}
