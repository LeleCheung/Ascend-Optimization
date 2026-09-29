REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def _legacy_gen_inputs(ctx, device):
    src_rows = tuple(ctx["input_tensor__shape"])[0]
    dst_rows = tuple(ctx["dst2src_map__shape"])[0]
    if dst_rows == src_rows and src_rows > 1:
        dst2src_map = torch.randperm(src_rows, device=device).to(torch.int32).contiguous()
    else:
        dst2src_map = torch.randint(0, src_rows, (dst_rows,), device=device, dtype=torch.int32).contiguous()
    return {"dst2src_map": dst2src_map}


def run(input_tensor, dst2src_map):
    return _custom_ops.shuffle_rows(input_tensor, dst2src_map)


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
