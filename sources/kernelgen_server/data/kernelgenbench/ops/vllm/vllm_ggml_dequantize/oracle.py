REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def _legacy_gen_inputs(ctx, device):
    rows, nbytes = ctx['W__shape']
    n_blocks = nbytes // 34
    scales = (torch.randn(rows, n_blocks, device=device, dtype=torch.float32).abs() * 0.5 + 0.1).to(torch.float16)
    scale_bytes = scales.contiguous().view(torch.uint8).reshape(rows, n_blocks, 2)
    qs = torch.randint(0, 256, (rows, n_blocks, 32), device=device, dtype=torch.uint8)
    W = torch.cat([scale_bytes, qs], dim=2).reshape(rows, n_blocks * 34)
    return {'W': W}


def run(W, quant_type, m, n, dtype):
    dt = getattr(torch, dtype)
    return _custom_ops.ggml_dequantize(W, quant_type, m, n, dt)


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
