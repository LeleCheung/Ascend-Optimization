REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def _legacy_gen_inputs(ctx, device):
    key_shape = tuple(ctx["key__shape"])
    num_blocks = ctx["num_blocks"]
    block_size = ctx["block_size"]
    T = key_shape[0]
    slot_mapping = torch.randperm(num_blocks * block_size, device=device, dtype=torch.int64)[:T]
    return {"slot_mapping": slot_mapping}


def run(key, value, slot_mapping, num_blocks, block_size):
    T, H, D = key.shape
    k_scale = torch.tensor(1.0, device=key.device, dtype=torch.float32)
    v_scale = torch.tensor(1.0, device=key.device, dtype=torch.float32)
    key_cache = torch.zeros(num_blocks, block_size, H, D, device=key.device, dtype=key.dtype)
    value_cache = torch.zeros(num_blocks, block_size, H, D, device=key.device, dtype=key.dtype)
    _custom_ops.reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, "auto", k_scale, v_scale)
    return (key_cache, value_cache)


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
