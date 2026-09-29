REFERENCE_DEVICE = 'target'

import torch

# Groups are processed in chunks to bound the size of the gathered state tile.
_GROUP_CHUNK = 256


def _as_int_list(value, length=None, default=0):
    if value is None:
        return [] if length is None else [default] * length
    if torch.is_tensor(value):
        values = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [int(value)] if length is None else [int(value)] * length
    values = [int(item) for item in values]
    if length is not None:
        values = (values + [default] * max(length - len(values), 0))[:length]
    return values


def _compressed_group_count(start, used, cmp_ratio):
    return max(
        (int(start) + max(int(used), 0)) // cmp_ratio - int(start) // cmp_ratio, 0
    )


def _batch_metadata(x, rope_sin, state_block_table, cu_seqlens, seqused, start_pos,
                    cmp_ratio):
    """Token bases / used lengths / starts, following the golden's _batch_metadata."""
    if x.dim() == 3:
        batch, seq_len, _ = x.shape
        token_bases = [i * seq_len for i in range(batch)]
        seq_lens = [seq_len] * batch
        seq_used = (
            _as_int_list(seqused, batch, seq_len) if seqused is not None else seq_lens
        )
        starts = _as_int_list(start_pos, batch, 0)
        out_per_batch = (seq_len + cmp_ratio - 1) // cmp_ratio
        return token_bases, seq_used, starts, batch * out_per_batch, out_per_batch

    total_tokens = x.shape[0]
    if cu_seqlens is not None:
        cumulative = _as_int_list(cu_seqlens)
        batch = max(len(cumulative) - 1, 0)
        token_bases = cumulative[:-1]
        seq_lens = [
            max(cumulative[i + 1] - cumulative[i], 0) for i in range(batch)
        ]
    elif state_block_table is not None:
        batch = int(state_block_table.shape[0])
        if batch != 1:
            raise ValueError(
                "cu_seqlens is required for rank-2 x when batch size is greater than one"
            )
        token_bases, seq_lens = [0], [total_tokens]
    else:
        batch, token_bases, seq_lens = 1, [0], [total_tokens]
    seq_used = _as_int_list(seqused, batch, 0) if seqused is not None else seq_lens
    starts = _as_int_list(start_pos, batch, 0)
    return token_bases, seq_used, starts, int(rope_sin.shape[0]), 0


def _cache_blocks(state_block_table, batch_index, positions, block_size, state_blocks):
    """Vectorized _cache_block_id: returns (cache_block, valid_mask) per position.

    A position is absent when its block index is past the table, when the table
    entry is 0 or beyond the cache, or when the position itself is negative --
    matching the golden's `0 < cache_block < state_blocks` test and its
    `position < 0` guard.
    """
    block_index = torch.div(positions, block_size, rounding_mode="floor")
    valid = positions >= 0
    if state_block_table is None:
        cache_block = block_index
        valid &= (block_index >= 0) & (block_index < state_blocks)
        return cache_block, valid
    max_blocks = state_block_table.shape[1]
    valid &= (block_index >= 0) & (block_index < max_blocks)
    safe = torch.where(valid, block_index, torch.zeros_like(block_index))
    cache_block = state_block_table[batch_index].to(torch.long)[safe]
    valid &= (cache_block > 0) & (cache_block < state_blocks)
    return cache_block, valid


def _apply_rope(x, rope_sin, rope_cos, rope_head_dim, rotary_mode):
    """Transcribed from the golden's _apply_rope.

    Only the trailing rope_head_dim lanes are rotated. rotary_mode 1 is the
    half-split form; any other value (the cases use 2) is the interleaved
    even/odd form.
    """
    if rope_head_dim <= 0:
        return x
    output = x.clone()
    start = x.shape[-1] - rope_head_dim
    rope = x[..., start:]
    sin = rope_sin.to(torch.float32)
    cos = rope_cos.to(torch.float32)
    if rotary_mode == 1:
        half = rope_head_dim // 2
        rotated = torch.cat((-rope[..., half:], rope[..., :half]), dim=-1)
    else:
        even, odd = rope[..., 0::2], rope[..., 1::2]
        rotated = torch.empty_like(rope)
        rotated[..., 0::2] = -odd
        rotated[..., 1::2] = even
    output[..., start:] = rope * cos + rotated * sin
    return output


def run(
    x,
    wkv,
    wgate,
    state_cache,
    ape,
    norm_weight,
    rope_sin,
    rope_cos,
    state_block_table=None,
    cu_seqlens=None,
    seqused=None,
    start_pos=None,
    rope_head_dim=64,
    cmp_ratio=128,
    coff=1,
    norm_eps=1e-6,
    rotary_mode=2,
    cache_mode=1,
):
    """Compressor, FlagGems-vllm PR #789 semantics.

    Transcribed from the delivery's own `compressor_torch` golden (the test
    compares the Triton kernel against it, and separately against the AscendC
    baseline), then vectorized: the golden's per-token scatter loop and its
    per-group gather loop are both O(tokens) in Python, which is far too slow at
    the case sizes (up to 8 x 4096 tokens). The arithmetic and the ordering are
    unchanged:

      1. project every token: kv = x @ wkv.T and gate = x @ wgate.T in float32,
         and scatter them into state_cache at (block, offset) derived from
         position = start_pos[b] + token_index, with the gate half offset by
         ape[position % cmp_ratio];
      2. for each compressed group, gather cmp_ratio*coff cached rows, softmax
         the gate rows across the group (per feature lane), take that weighted
         sum of the kv rows, RMS-normalize with norm_eps, scale by norm_weight,
         and apply RoPE.

    Absent cache blocks contribute a zero kv row and a -float32max gate row, so
    they vanish from the softmax -- the mechanism the golden uses to mask
    positions before the start of the cache. `coff == 2` splits each group into a
    previous-window half (dim_start 0) and a current-window half (dim_start
    head_dim); `coff == 1` reads a single window at dim_start 0.

    state_cache is updated in place. Output rows past each request's group count
    stay zero, which is what the test asserts for the padding region.
    """
    del cache_mode  # the golden ignores it; only the paged layout is implemented
    hidden_size = x.shape[-1]
    head_dim = int(norm_weight.shape[0])
    projection_dim = coff * head_dim
    device = x.device

    token_bases, seq_used, starts, flat_rows, out_per_batch = _batch_metadata(
        x, rope_sin, state_block_table, cu_seqlens, seqused, start_pos, cmp_ratio
    )

    x_2d = x.reshape(-1, hidden_size).to(torch.float32)
    kv_projection = x_2d @ wkv.to(torch.float32).t()
    score_projection = x_2d @ wgate.to(torch.float32).t()
    block_size, state_blocks = int(state_cache.shape[1]), int(state_cache.shape[0])
    cache_flat = state_cache.view(-1, state_cache.shape[-1])
    ape32 = ape.to(torch.float32)

    # ---- scatter phase ----
    for batch_index, (token_base, used, start) in enumerate(
        zip(token_bases, seq_used, starts)
    ):
        used = max(int(used), 0)
        if used == 0:
            continue
        token_index = torch.arange(used, device=device, dtype=torch.long)
        positions = token_index + int(start)
        cache_block, valid = _cache_blocks(
            state_block_table, batch_index, positions, block_size, state_blocks
        )
        if not bool(valid.any()):
            continue
        positions = positions[valid]
        rows = cache_block[valid] * block_size + (positions % block_size)
        src = token_base + token_index[valid]
        cache_flat[rows, :projection_dim] = kv_projection[src]
        cache_flat[rows, projection_dim : 2 * projection_dim] = (
            score_projection[src] + ape32[positions % cmp_ratio, :projection_dim]
        )

    # ---- output geometry ----
    if x.dim() == 3:
        output = torch.zeros(
            (len(token_bases), out_per_batch, head_dim),
            device=device,
            dtype=torch.float32,
        )
        output_bases = [i * out_per_batch for i in range(len(token_bases))]
        group_counts = [
            min(_compressed_group_count(s, u, cmp_ratio), out_per_batch)
            for s, u in zip(starts, seq_used)
        ]
    else:
        output = torch.zeros((flat_rows, head_dim), device=device, dtype=torch.float32)
        output_bases, group_counts, cursor = [], [], 0
        for s, u in zip(starts, seq_used):
            output_bases.append(cursor)
            groups = max(
                min(_compressed_group_count(s, u, cmp_ratio), flat_rows - cursor), 0
            )
            group_counts.append(groups)
            cursor += groups

    output_flat = output.reshape(-1, head_dim)
    rope_sin_flat = rope_sin.reshape(-1, rope_head_dim)
    rope_cos_flat = rope_cos.reshape(-1, rope_head_dim)
    large_negative = torch.finfo(torch.float32).min

    group_span = cmp_ratio * coff
    offsets = torch.arange(group_span, device=device, dtype=torch.long)
    if coff == 1:
        pos_delta = offsets
        dim_start = torch.zeros(group_span, device=device, dtype=torch.long)
    else:
        # coff == 2: the first cmp_ratio offsets read the previous window at
        # dim_start 0, the rest read the current window at dim_start head_dim.
        # Both halves use the same position delta (offset - cmp_ratio); only the
        # feature-dim offset differs.
        in_prev = offsets < cmp_ratio
        pos_delta = offsets - cmp_ratio
        dim_start = torch.where(
            in_prev,
            torch.zeros_like(offsets),
            torch.full_like(offsets, head_dim),
        )
    lane = torch.arange(head_dim, device=device, dtype=torch.long)
    kv_cols = dim_start[:, None] + lane[None, :]
    gate_cols = projection_dim + kv_cols

    # ---- compression phase ----
    for batch_index, (output_base, groups, start) in enumerate(
        zip(output_bases, group_counts, starts)
    ):
        for chunk_start in range(0, groups, _GROUP_CHUNK):
            chunk_end = min(chunk_start + _GROUP_CHUNK, groups)
            group_index = torch.arange(
                chunk_start, chunk_end, device=device, dtype=torch.long
            )
            chunk_len = chunk_end - chunk_start
            group_start = (
                torch.div(
                    int(start) + group_index * cmp_ratio, cmp_ratio,
                    rounding_mode="floor",
                )
                * cmp_ratio
            )
            positions = group_start[:, None] + pos_delta[None, :]
            cache_block, valid = _cache_blocks(
                state_block_table, batch_index, positions, block_size, state_blocks
            )
            rows = cache_block * block_size + (positions % block_size)
            rows = torch.where(valid, rows, torch.zeros_like(rows))
            tile = cache_flat[rows.reshape(-1)]
            tile = tile.reshape(chunk_len, group_span, -1)
            kv_rows = torch.gather(
                tile, 2, kv_cols.unsqueeze(0).expand(chunk_len, -1, -1)
            )
            gate_rows = torch.gather(
                tile, 2, gate_cols.unsqueeze(0).expand(chunk_len, -1, -1)
            )
            mask = valid.unsqueeze(-1)
            kv_rows = torch.where(mask, kv_rows, torch.zeros_like(kv_rows))
            gate_rows = torch.where(
                mask, gate_rows, torch.full_like(gate_rows, large_negative)
            )
            weights = torch.softmax(gate_rows, dim=1)
            compressed = (weights * kv_rows).sum(dim=1)
            normalized = compressed * torch.rsqrt(
                (compressed * compressed).mean(dim=-1, keepdim=True) + norm_eps
            )
            normalized = normalized * norm_weight.to(torch.float32)
            rows_out = output_base + group_index
            output_flat[rows_out] = _apply_rope(
                normalized,
                rope_sin_flat[rows_out],
                rope_cos_flat[rows_out],
                rope_head_dim,
                rotary_mode,
            )

    return output.to(dtype=x.dtype)


# Case registry transcribed from tests/compressor_cases.py::CASE_SPECS:
# (batch, hidden, seq_lens, head_dim, block_size, rope_dim, cmp_ratio, coff, start_pos)
_CASE_SPECS = {
    "Prefill0": (1, 4096, 8192, 512, 128, 64, 4, 2, 0),
    "Prefill1": (1, 4096, 8192, 128, 128, 64, 4, 2, 0),
    "Prefill2": (1, 4096, 8192, 512, 128, 64, 128, 1, 0),
    "Prefill0_b2": (2, 4096, 4096, 512, 128, 64, 4, 2, 0),
    "Prefill1_b2": (2, 4096, 4096, 128, 128, 64, 4, 2, 0),
    "Prefill2_b2": (2, 4096, 4096, 512, 128, 64, 128, 1, 0),
    "Prefill0_b2_rank3": (2, 4096, 4096, 512, 128, 64, 4, 2, 0),
    "Prefill1_b2_rank3": (2, 4096, 4096, 128, 128, 64, 4, 2, 0),
    "Prefill2_b2_rank3": (2, 4096, 4096, 512, 128, 64, 128, 1, 0),
    "prefill_b8_concurrency": (8, 4096, 1024, 512, 128, 64, 4, 2, 0),
    "prefill_b32_concurrency": (32, 4096, 320, 512, 128, 64, 4, 2, 0),
    "prefill_b2_chunk_tail": (
        2, 4096, (2944, 7296), 512, 128, 64, 4, 2, (128128, 117888),
    ),
    "decode0_start8195": (1, 4096, 1, 512, 128, 64, 4, 2, 8195),
    "decode1_start8195": (1, 4096, 1, 128, 128, 64, 4, 2, 8195),
    "decode2_start8195": (1, 4096, 1, 512, 128, 64, 128, 1, 8195),
    "decode3": (8, 4096, 3, 512, 128, 64, 4, 2, 8193),
    "decode4_start8193": (8, 4096, 3, 128, 128, 64, 4, 2, 8193),
    "decode5": (8, 4096, 3, 512, 128, 64, 128, 1, 8193),
    "decode0_b32": (32, 4096, 1, 512, 128, 64, 4, 2, 8195),
    "decode3_b32": (32, 4096, 3, 512, 128, 64, 4, 2, 8193),
    "decode5_b32": (32, 4096, 3, 512, 128, 64, 128, 1, 8193),
    "prefill_b1_nonzero_c2": (1, 4096, 1024, 128, 128, 64, 4, 2, 8195),
    "prefill_b1_nonzero_c1": (1, 4096, 1024, 512, 128, 64, 128, 1, 8193),
    "prefill_b2_nonzero_c1": (
        2, 4096, (1024, 1152), 512, 128, 64, 128, 1, (8193, 12231),
    ),
}


def _per_request_values(value, batch, name):
    values = (value,) * batch if isinstance(value, int) else tuple(value)
    if len(values) != batch:
        raise ValueError(f"{name} must contain {batch} values, got {len(values)}")
    return values


def _uniform(shape, low, high, dtype, device, generator):
    return (
        torch.rand(shape, dtype=torch.float32, device=device, generator=generator)
        * (high - low)
        + low
    ).to(dtype)



def _scalars(ctx):
    """Unwrap ctx["inputs"] recipe entries to plain values.

    The harness passes each input as its raw recipe mapping (e.g.
    {"type": "scalar", "value": 4}), so scalar knobs must be read out of the
    "value" field; non-recipe entries are already plain.
    """
    out = {}
    for key, raw in ctx["inputs"].items():
        if isinstance(raw, dict) and "value" in raw:
            out[key] = raw["value"]
        else:
            out[key] = raw
    return out


def gen_inputs(ctx, device):
    """Build one registered compressor case.

    Transcribed from tests/compressor_cases.py::make_inputs. The geometry is
    interdependent in ways a recipe cannot express: the block table is sized from
    max(start + len), the number of rope rows differs between the rank-2 and
    rank-3 forms, and `state_cache` must be float32 while the activations are
    bfloat16. Cases whose name ends in `_rank3` use the [batch, seq, hidden]
    form with cu_seqlens=None; the others are flat with cu_seqlens set.
    """
    case_name = _scalars(ctx)["case"]
    batch, hidden, seq_spec, head_dim, block_size, rope_dim, ratio, coff, start_spec = (
        _CASE_SPECS[case_name]
    )
    seq_lens = _per_request_values(seq_spec, batch, "sequence lengths")
    starts = _per_request_values(start_spec, batch, "start positions")

    dtype = torch.bfloat16
    total_tokens = sum(seq_lens)
    projection_dim = coff * head_dim

    generator = torch.Generator(device=device)
    generator.manual_seed(int(ctx["seed"]))

    cumulative = torch.tensor((0, *_accumulate(seq_lens)), dtype=torch.int32)
    start_pos = torch.tensor(starts, dtype=torch.int32)
    maximum_position = max(s + l for s, l in zip(starts, seq_lens))
    max_blocks = (maximum_position + block_size - 1) // block_size
    block_table = torch.arange(
        1, batch * max_blocks + 1, dtype=torch.int32
    ).reshape(batch, max_blocks)
    state_blocks = batch * max_blocks + 1

    rank3 = case_name.endswith("_rank3")
    out_per_batch = (seq_lens[0] + ratio - 1) // ratio
    rope_rows = (
        batch * out_per_batch
        if rank3
        else min(total_tokens, total_tokens // ratio + batch)
    )

    x = _uniform((total_tokens, hidden), -10, 10, dtype, device, generator)
    rope_sin = _uniform((rope_rows, rope_dim), -1, 1, dtype, device, generator)
    rope_cos = _uniform((rope_rows, rope_dim), -1, 1, dtype, device, generator)
    if rank3:
        x = x.reshape(batch, seq_lens[0], hidden)
        rope_sin = rope_sin.reshape(batch, out_per_batch, rope_dim)
        rope_cos = rope_cos.reshape(batch, out_per_batch, rope_dim)

    return {
        "x": x,
        "wkv": _uniform((projection_dim, hidden), -10, 10, dtype, device, generator),
        "wgate": _uniform((projection_dim, hidden), -10, 10, dtype, device, generator),
        "state_cache": _uniform(
            (state_blocks, block_size, 2 * projection_dim),
            -10, 10, torch.float32, device, generator,
        ),
        "ape": _uniform(
            (ratio, projection_dim), -10, 10, torch.float32, device, generator
        ),
        "norm_weight": _uniform((head_dim,), -10, 10, dtype, device, generator),
        "rope_sin": rope_sin,
        "rope_cos": rope_cos,
        "state_block_table": block_table.to(device=device),
        "cu_seqlens": None if rank3 else cumulative.to(device=device),
        "seqused": None,
        "start_pos": start_pos.to(device=device),
        "rope_head_dim": rope_dim,
        "cmp_ratio": ratio,
        "coff": coff,
        "norm_eps": 1e-6,
        "rotary_mode": 2,
        "cache_mode": 1,
    }


def _accumulate(values):
    total = 0
    for value in values:
        total += value
        yield total
