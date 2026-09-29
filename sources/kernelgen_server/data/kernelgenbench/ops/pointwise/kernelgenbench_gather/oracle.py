REFERENCE_DEVICE = 'target'

import torch

def _legacy_gen_inputs(ctx, device):
    x_shape = tuple(ctx["x__shape"])
    dtype = getattr(torch, ctx["x__dtype"])
    index_shape = tuple(ctx["index__shape"])
    dim = int(ctx["dim"])
    x = torch.randn(x_shape, dtype=dtype, device=device)
    index = torch.randint(0, x_shape[dim], index_shape, dtype=torch.int64, device=device)
    return {"x": x, "index": index}

def run(x, dim, index):
    return torch.ops.aten.gather(x, dim, index)


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
