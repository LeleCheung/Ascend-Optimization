REFERENCE_DEVICE = 'target'

import torch
def _legacy_gen_inputs(ctx, device):
    result = {"x": torch.randn(tuple(ctx["x__shape"]), dtype=getattr(torch, ctx["x__dtype"]), device=device)}
    for position in range(3):
        name = f"index{position}"
        shape = ctx.get(f"{name}__shape")
        dtype = ctx.get(f"{name}__dtype")
        if shape is None:
            continue
        if dtype == "bool":
            result[name] = torch.randint(0, 2, tuple(shape), dtype=torch.bool, device=device)
        else:
            upper = ctx.get(f"{name}__upper", ctx["x__shape"][position])
            result[name] = (torch.empty(tuple(shape), dtype=torch.int64, device=device)
                            if upper == 0 else torch.randint(0, upper, tuple(shape), dtype=torch.int64, device=device))
    return result
def run(x, index0, index1, index2, index_count):
    return torch.ops.aten.index.Tensor(x, [index0, index1, index2][:index_count])


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
