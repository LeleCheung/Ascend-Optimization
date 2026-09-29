REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def _legacy_gen_inputs(ctx, device):
    shape = tuple(ctx["input__shape"])
    dtype = getattr(torch, ctx["input__dtype"])
    input_tensor = torch.randn(shape, device=device, dtype=dtype)
    output = torch.empty(shape, device=device, dtype=torch.uint8)
    return {"input": input_tensor, "output": output}


def run(output, input, scale):
    _custom_ops.convert_fp8(output, input, scale, "fp8")
    return output


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
