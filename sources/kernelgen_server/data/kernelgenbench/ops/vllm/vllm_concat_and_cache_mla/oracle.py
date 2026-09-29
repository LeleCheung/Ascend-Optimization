REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def _legacy_gen_inputs(ctx, device):
    T = ctx["kv_c__shape"][0]
    kv_lora_rank = ctx["kv_c__shape"][1]
    pe_dim = ctx["k_pe__shape"][1]
    num_blocks = ctx["kv_cache__shape"][0]
    block_size = ctx["kv_cache__shape"][1]
    total_dim = kv_lora_rank + pe_dim
    dtype = getattr(torch, ctx["kv_c__dtype"])
    kv_c = torch.randn(T, kv_lora_rank, device=device, dtype=dtype)
    k_pe = torch.randn(T, pe_dim, device=device, dtype=dtype)
    kv_cache = torch.zeros(num_blocks, block_size, total_dim, device=device, dtype=dtype)
    slot_mapping = torch.randperm(num_blocks * block_size, device=device, dtype=torch.int64)[:T]
    return {"kv_c": kv_c, "k_pe": k_pe, "kv_cache": kv_cache, "slot_mapping": slot_mapping}


def run(kv_c, k_pe, kv_cache, slot_mapping, scale):
    scale = torch.as_tensor(scale, device=kv_c.device, dtype=torch.float32)
    _custom_ops.concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping, "auto", scale)
    return kv_cache


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
