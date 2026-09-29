REFERENCE_DEVICE = 'target'

import torch
def _legacy_gen_inputs(ctx, device):
    if not ctx.get("other__column_major", False):
        return {}
    shape = tuple(ctx["other__shape"]); dtype = getattr(torch, ctx["other__dtype"])
    return {"other": torch.randn((shape[1], shape[0]), dtype=dtype, device=device).t()}
def run(x, other):
    return torch.ops.aten.mm(x, other)


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
