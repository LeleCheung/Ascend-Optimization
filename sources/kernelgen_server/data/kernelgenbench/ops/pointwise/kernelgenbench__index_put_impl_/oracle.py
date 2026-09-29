REFERENCE_DEVICE = 'target'

import torch

def _legacy_gen_inputs(ctx, device):
    shape = tuple(ctx["x__shape"])
    dtype = getattr(torch, ctx["x__dtype"])
    num_index_tensors, count = tuple(ctx["indices__shape"])
    accumulate = bool(ctx["accumulate"])
    if num_index_tensors != len(shape):
        raise ValueError("indices metadata must contain one tensor per input dimension")
    if accumulate:
        x = torch.zeros(shape, dtype=dtype, device=device)
        indices = [torch.zeros(count, dtype=torch.int64, device=device)
                   for _ in shape]
        values = torch.ones(count, dtype=dtype, device=device)
    else:
        x = torch.randn(shape, dtype=dtype, device=device)
        indices = [torch.randint(0, size, (count,), dtype=torch.int64, device=device)
                   for size in shape]
        values = torch.randn(count, dtype=dtype, device=device)
    return {"x": x, "indices": indices, "values": values}

def run(x, indices, values, accumulate):
    return torch.ops.aten._index_put_impl_(x, indices, values, accumulate, False)


def _legacy_context(ctx):
    result = {}
    for name, spec in ctx["inputs"].items():
        kind = spec.get("type") if isinstance(spec, dict) else None
        if kind in {"random", "custom"}:
            result[f"{name}__shape"] = spec["shape"]
            result[f"{name}__dtype"] = spec["dtype"]
            for key, value in spec.items():
                if key not in {"type", "shape", "dtype"}:
                    result[f"{name}__{key}"] = value
        elif kind in {"scalar", "literal"}:
            result[name] = spec["value"]
        else:
            raise ValueError(f"unsupported legacy input recipe: {name}")
    return result


def gen_inputs(ctx, device):
    return _legacy_gen_inputs(_legacy_context(ctx), device)
