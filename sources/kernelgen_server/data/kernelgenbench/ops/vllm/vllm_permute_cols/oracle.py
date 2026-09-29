REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def _legacy_gen_inputs(ctx, device):
    cols = tuple(ctx["a__shape"])[1]
    pattern = ctx["perm__pattern"]
    if pattern == "identity":
        perm = torch.arange(cols, device=device, dtype=torch.int32)
    elif pattern == "reverse":
        perm = torch.arange(cols - 1, -1, -1, device=device, dtype=torch.int32)
    else:
        gen = torch.Generator(device=device)
        gen.manual_seed(42)
        perm = torch.randperm(cols, device=device, generator=gen).to(torch.int32)
    return {"perm": perm}


def run(a, perm):
    return _custom_ops.permute_cols(a, perm)


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
