REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def _legacy_gen_inputs(ctx, device):
    num_blocks = tuple(ctx["src__shape"])[0]
    mapping_rows = tuple(ctx["block_mapping__shape"])[0]
    kind = ctx["block_mapping__mapping_kind"]
    if kind == "identity":
        src_idx = dst_idx = torch.arange(num_blocks, dtype=torch.int64)
    elif kind == "reverse":
        src_idx = torch.arange(num_blocks, dtype=torch.int64)
        dst_idx = torch.arange(num_blocks - 1, -1, -1, dtype=torch.int64)
    else:
        src_idx = torch.randperm(num_blocks, dtype=torch.int64)[:mapping_rows]
        dst_idx = torch.randperm(num_blocks, dtype=torch.int64)[:mapping_rows]
    block_mapping = torch.stack([src_idx, dst_idx], dim=1).to("cpu", dtype=torch.int64)
    return {"block_mapping": block_mapping}


def run(src, dst, block_mapping):
    out = dst.clone()
    _custom_ops.swap_blocks(src, out, block_mapping)
    return out


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
