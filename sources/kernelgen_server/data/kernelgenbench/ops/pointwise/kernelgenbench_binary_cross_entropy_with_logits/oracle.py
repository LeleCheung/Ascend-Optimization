REFERENCE_DEVICE = 'target'

import torch
def _legacy_gen_inputs(ctx, device):
    result = {}
    for name in ("target", "weight", "pos_weight"):
        shape = ctx.get(f"{name}__shape")
        dtype = ctx.get(f"{name}__dtype")
        if shape is not None and dtype is not None:
            value = torch.rand(tuple(shape), dtype=getattr(torch, dtype), device=device)
            result[name] = value if name == "target" else value + 0.5
    return result
def run(input, target, weight, pos_weight, reduction):
    return torch.ops.aten.binary_cross_entropy_with_logits(
        input, target, weight, pos_weight, reduction
    )


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
