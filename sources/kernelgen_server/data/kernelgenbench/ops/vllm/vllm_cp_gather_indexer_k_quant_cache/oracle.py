REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def _legacy_gen_inputs(ctx, device):
    num_blocks = ctx['kv_cache__shape'][0]
    block_size = ctx['kv_cache__shape'][1]
    batch, blocks_per_seq = ctx['block_table__shape']
    block_table = torch.randint(0, num_blocks, (batch, blocks_per_seq), device=device, dtype=torch.int32)
    total_tokens = batch * blocks_per_seq * block_size
    cu_seq_lens = torch.arange(0, total_tokens + 1, blocks_per_seq * block_size, device=device, dtype=torch.int32)
    return {'block_table': block_table, 'cu_seq_lens': cu_seq_lens}


def run(kv_cache, block_table, cu_seq_lens):
    num_heads = kv_cache.shape[2]
    head_size = kv_cache.shape[3]
    total_tokens = int(cu_seq_lens[-1].item())
    dst_k = torch.zeros(total_tokens, num_heads, head_size, device=kv_cache.device, dtype=torch.float16)
    dst_scale = torch.zeros(total_tokens, num_heads, device=kv_cache.device, dtype=torch.float32)
    _custom_ops.cp_gather_indexer_k_quant_cache(kv_cache, dst_k, dst_scale, block_table, cu_seq_lens)
    return dst_k, dst_scale


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
