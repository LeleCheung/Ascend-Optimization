REFERENCE_DEVICE = 'target'

import torch

def _legacy_gen_inputs(ctx, device):
    shape = tuple(ctx["x__shape"]); dtype = getattr(torch, ctx["x__dtype"])
    x = torch.randn(shape, dtype=dtype, device=device)
    buffer = torch.sigmoid(x)
    return {"x": x, "buffer": buffer}

def run(grad_output, x, buffer):
    return torch.ops.aten.log_sigmoid_backward(grad_output, x, buffer)


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
