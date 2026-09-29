REFERENCE_DEVICE = 'target'

import torch
def run(x, dtype, memory_format):
    dtype = getattr(torch, dtype) if dtype else None
    memory_format = getattr(torch, memory_format) if memory_format else None
    return torch.ops.aten._to_copy(x, dtype=dtype, memory_format=memory_format)


def gen_inputs(ctx, device):
    result = {}
    for name, spec in ctx["inputs"].items():
        if not isinstance(spec, dict) or spec.get("type") != "custom":
            continue
        shape = tuple(spec["shape"])
        stride = tuple(spec["stride"])
        storage_offset = spec.get("storage_offset", 0)
        storage_size = storage_offset
        if all(dimension > 0 for dimension in shape):
            storage_size += 1 + sum(
                (dimension - 1) * step
                for dimension, step in zip(shape, stride)
            )
        dtype = getattr(torch, spec["dtype"])
        base = torch.empty((storage_size,), dtype=dtype, device=device)
        if dtype.is_floating_point or dtype.is_complex:
            base.normal_()
        elif dtype is torch.bool:
            base.random_(0, 2)
        elif dtype is torch.uint8:
            base.random_(0, 256)
        else:
            base.random_(-1024, 1024)
        result[name] = torch.as_strided(
            base, shape, stride, storage_offset
        )
    return result
