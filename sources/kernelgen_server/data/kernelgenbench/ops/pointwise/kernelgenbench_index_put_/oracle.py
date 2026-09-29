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
    bool_index = next((result[f"index{i}"] for i in range(3)
                       if f"index{i}" in result and result[f"index{i}"].dtype == torch.bool), None)
    values_shape = (int(bool_index.sum().item()),) if bool_index is not None else tuple(ctx["values__shape"])
    result["values"] = torch.randn(values_shape, dtype=getattr(torch, ctx["values__dtype"]), device=device)
    return result
def run(x, index0, index1, index2, index_count, values, accumulate):
    return torch.ops.aten.index_put_(x, [index0, index1, index2][:index_count], values, accumulate)


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
