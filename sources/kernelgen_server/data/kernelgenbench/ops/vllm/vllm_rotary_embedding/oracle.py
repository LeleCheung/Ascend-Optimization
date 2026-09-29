REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def _legacy_gen_inputs(ctx, device):
    seq_len = tuple(ctx["positions__shape"])[0]
    cs_shape = tuple(ctx["cos_sin_cache__shape"])
    cache_len, head_size = cs_shape[0], cs_shape[2]
    cs_dtype = getattr(torch, ctx["cos_sin_cache__dtype"])
    positions = torch.randint(0, cache_len, (seq_len,), device=device, dtype=torch.long)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_size, 2, device=device, dtype=torch.float32) / head_size))
    t = torch.arange(cache_len, device=device, dtype=torch.float32).unsqueeze(1)
    freqs = t * inv_freq
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
    cos_sin_cache = torch.stack([cos, sin], dim=1).to(dtype=cs_dtype)
    return {"positions": positions, "cos_sin_cache": cos_sin_cache}


def run(positions, query, key, head_size, cos_sin_cache, is_neox):
    q = query.clone()
    k = key.clone() if key is not None else None
    _custom_ops.rotary_embedding(positions, q, k, head_size, cos_sin_cache, is_neox)
    return q, k


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
