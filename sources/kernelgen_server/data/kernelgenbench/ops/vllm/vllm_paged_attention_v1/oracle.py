REFERENCE_DEVICE = 'target'

import torch

try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def _legacy_gen_inputs(ctx, device):
    num_seqs, num_heads, head_size = ctx["query__shape"]
    # Workload metadata preserves the source layout:
    # [total_blocks, block_size, num_kv_heads, head_size]. vLLM's CUDA op
    # consumes the vectorized K cache and transposed V cache below.
    total_blocks, block_size, num_kv_heads, _ = ctx["key_cache__shape"]
    dtype = getattr(torch, ctx["query__dtype"])
    max_context_len = ctx["max_context_len"]
    query = torch.randn(
        num_seqs, num_heads, head_size, dtype=dtype, device=device
    )
    key_cache = torch.randn(
        total_blocks,
        num_kv_heads,
        head_size // 16,
        block_size,
        16,
        dtype=dtype,
        device=device,
    )
    value_cache = torch.randn(
        total_blocks,
        num_kv_heads,
        head_size,
        block_size,
        dtype=dtype,
        device=device,
    )
    seq_lens_list = [max(1, max_context_len - i % 3) for i in range(num_seqs)]
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
    max_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    block_tables = torch.zeros(
        num_seqs, max_blocks_per_seq, dtype=torch.int32, device=device
    )
    current = 0
    for seq, length in enumerate(seq_lens_list):
        count = (length + block_size - 1) // block_size
        block_tables[seq, :count] = torch.arange(
            current, current + count, dtype=torch.int32, device=device
        )
        current += count
    return {
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_tables": block_tables,
        "seq_lens": seq_lens,
    }


def run(query, key_cache, value_cache, block_tables, seq_lens, scale,
        alibi_slopes, max_context_len, kv_cache_dtype):
    output = torch.empty_like(query)
    cache_scale = torch.ones(1, dtype=torch.float32, device=query.device)
    _custom_ops.paged_attention_v1(
        output,
        query,
        key_cache,
        value_cache,
        num_kv_heads=key_cache.shape[1],
        scale=scale,
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=key_cache.shape[3],
        max_seq_len=max_context_len,
        alibi_slopes=alibi_slopes,
        kv_cache_dtype=kv_cache_dtype,
        k_scale=cache_scale,
        v_scale=cache_scale,
    )
    return output


def _legacy_valid(ref_outs, cand_outs, inputs, ctx):
    ref_out = ref_outs[0]
    cand_out = cand_outs[0]
    if ref_out.shape != cand_out.shape:
        return {"passed": False, "message": "shape mismatch", "metrics": {}}
    max_diff = (ref_out.float() - cand_out.float()).abs().max().item()
    threshold = 0.01 + 0.01 * ref_out.abs().float().max().item()
    return {
        "passed": max_diff <= threshold,
        "message": f"max_diff={max_diff:.6f}",
        "metrics": {"max_absolute_error": max_diff},
    }


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


VALID_OWNS_RETURN_CONTRACT = True

def valid(ref_outputs, sol_outputs, inputs, ctx):
    ordered = [inputs[name] for name in ['query', 'key_cache', 'value_cache', 'block_tables', 'seq_lens', 'scale', 'alibi_slopes', 'max_context_len', 'kv_cache_dtype']]
    return _legacy_valid(
        ref_outputs, sol_outputs, ordered, _legacy_context(ctx)
    )
