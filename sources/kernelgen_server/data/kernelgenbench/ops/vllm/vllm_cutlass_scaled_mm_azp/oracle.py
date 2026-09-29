REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def _legacy_gen_inputs(ctx, device):
    k, n = ctx['b__shape']
    m = ctx['a__shape'][0]
    b = torch.randint(-128, 128, (k, n), device=device, dtype=torch.int8).t().contiguous().t()
    scale_a = torch.rand(m, 1, device=device, dtype=torch.float32) + 0.5
    scale_b = torch.rand(1, n, device=device, dtype=torch.float32) + 0.5
    azp_adj = torch.randint(-8, 9, (n,), device=device, dtype=torch.int32)
    azp = torch.randint(-8, 9, (m,), device=device, dtype=torch.int32)
    return {'b': b, 'scale_a': scale_a, 'scale_b': scale_b, 'azp_adj': azp_adj, 'azp': azp}


def run(a, b, scale_a, scale_b, out_dtype, azp_adj, azp, bias):
    dt = getattr(torch, out_dtype)
    return _custom_ops.cutlass_scaled_mm_azp(a, b, scale_a, scale_b, dt, azp_adj, azp, bias)


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
