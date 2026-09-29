REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def _legacy_gen_inputs(ctx, device):
    kc_shape = tuple(ctx["key_caches__shape"])
    vc_shape = tuple(ctx["value_caches__shape"])
    num_blocks = kc_shape[0]
    num_mappings = tuple(ctx["block_mapping__shape"])[0]
    kc_dtype = getattr(torch, ctx["key_caches__dtype"])
    vc_dtype = getattr(torch, ctx["value_caches__dtype"])
    key_caches = [torch.randn(kc_shape, device=device, dtype=kc_dtype)]
    value_caches = [torch.randn(vc_shape, device=device, dtype=vc_dtype)]
    perm = torch.randperm(num_blocks, device=device)
    src = perm[:num_mappings].to(torch.int64)
    dst = perm[num_mappings:2 * num_mappings].to(torch.int64)
    block_mapping = torch.stack([src, dst], dim=1)
    return {"key_caches": key_caches, "value_caches": value_caches, "block_mapping": block_mapping}


def run(key_caches, value_caches, block_mapping):
    _custom_ops.copy_blocks(key_caches, value_caches, block_mapping)
    return key_caches, value_caches


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
