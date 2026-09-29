REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def _legacy_gen_inputs(ctx, device):
    num_blocks = ctx['src_cache__shape'][0]
    block_size = ctx['src_cache__shape'][1]
    batch, blocks_per_seq = ctx['block_table__shape']
    seq_len = blocks_per_seq * block_size
    total_tokens = batch * seq_len
    block_table = torch.randint(0, num_blocks, (batch, blocks_per_seq), device=device, dtype=torch.int32)
    cu_seq_lens = torch.arange(0, total_tokens + 1, seq_len, device=device, dtype=torch.int32)
    token_to_seq = torch.repeat_interleave(torch.arange(batch, device=device, dtype=torch.int32), seq_len)
    scale = torch.tensor([1.0], device=device, dtype=torch.float32)
    return {'block_table': block_table, 'cu_seq_lens': cu_seq_lens, 'token_to_seq': token_to_seq, 'scale': scale}


def run(src_cache, block_table, cu_seq_lens, token_to_seq, num_tokens, kv_cache_dtype, scale):
    num_heads = src_cache.shape[2]
    head_dim = src_cache.shape[3]
    dst = torch.zeros(num_tokens, num_heads, head_dim, device=src_cache.device, dtype=torch.float16)
    _custom_ops.gather_and_maybe_dequant_cache(src_cache, dst, block_table, cu_seq_lens, token_to_seq, num_tokens, kv_cache_dtype, scale, None)
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
