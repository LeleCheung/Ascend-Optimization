REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def _legacy_gen_inputs(ctx, device):
    B = ctx["q_seqlens__shape"][0]
    q_len = ctx["context_size"]
    kv_len = ctx["context_size"]
    q_seqlens = torch.full((B,), q_len, device=device, dtype=torch.int32)
    kv_seqlens = torch.full((B,), kv_len, device=device, dtype=torch.int32)
    vi = torch.randint(0, kv_len, (B, q_len, q_len), device=device, dtype=torch.int32)
    si = torch.randint(0, kv_len, (B, q_len, q_len), device=device, dtype=torch.int32)
    return {"q_seqlens": q_seqlens, "kv_seqlens": kv_seqlens, "vertical_indexes": vi, "slash_indexes": si}


def run(q_seqlens, kv_seqlens, vertical_indexes, slash_indexes, context_size, block_size_M, block_size_N):
    return _custom_ops.convert_vertical_slash_indexes(
        q_seqlens, kv_seqlens, vertical_indexes, slash_indexes, context_size, block_size_M, block_size_N, True)


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
