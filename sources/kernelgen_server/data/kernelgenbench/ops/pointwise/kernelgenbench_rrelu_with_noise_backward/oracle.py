REFERENCE_DEVICE = 'target'

import torch

def _legacy_gen_inputs(ctx, device):
    shape = tuple(ctx["noise__shape"]); dtype = getattr(torch, ctx["noise__dtype"])
    return {"noise": torch.ones(shape, dtype=dtype, device=device)}

def run(grad_output, x, noise, lower, upper, training, self_is_result):
    return torch.ops.aten.rrelu_with_noise_backward(grad_output, x, noise, lower, upper, training, self_is_result)


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
