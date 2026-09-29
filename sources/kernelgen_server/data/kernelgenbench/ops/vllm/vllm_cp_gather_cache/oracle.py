REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def _legacy_gen_inputs(ctx, device):
    num_blocks = ctx["src_cache__shape"][0]
    block_size = ctx["src_cache__shape"][1]
    num_heads = ctx["src_cache__shape"][2]
    head_size = ctx["src_cache__shape"][3]
    batch_size = ctx["batch_size"]
    blocks_per_seq = 2
    total_tokens = batch_size * blocks_per_seq * block_size
    src_cache = torch.randn(num_blocks, block_size, num_heads, head_size, device=device, dtype=torch.float16)
    dst = torch.zeros(total_tokens, num_heads, head_size, device=device, dtype=torch.float16)
    block_table = torch.randint(0, num_blocks, (batch_size, blocks_per_seq), device=device, dtype=torch.int32)
    cu_seq_lens = torch.arange(0, total_tokens + 1, blocks_per_seq * block_size, device=device, dtype=torch.int32)
    return {"src_cache": src_cache, "dst": dst, "block_table": block_table, "cu_seq_lens": cu_seq_lens}


def run(src_cache, dst, block_table, cu_seq_lens, batch_size):
    _custom_ops.cp_gather_cache(src_cache, dst, block_table, cu_seq_lens, batch_size, None)
    return dst


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
