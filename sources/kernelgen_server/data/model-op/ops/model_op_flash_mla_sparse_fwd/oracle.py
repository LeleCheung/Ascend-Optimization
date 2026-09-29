REFERENCE_DEVICE = 'target'

import torch

Q_HEADS = 128
KV_HEADS = 1
HEAD_DIM = 512
ROPE_DIM = 64
PAGE_SIZE = 128

# Query tokens are scored in chunks so the [chunk*heads, topk] score tile stays
# bounded at topk = 2048.
_TOKEN_CHUNK = 8


def _gather_sparse_kv(key, value, key_rope, sparse_indices, block_table, token,
                      batch):
    """Gather K (with rope appended) and V at the sparse positions of one token.

    Transcribed from fused_pa_rope_to_sparse_kernel:

        block_id        = sparse_idx // BLOCK_SIZE
        bs_offset       = sparse_idx %  BLOCK_SIZE
        actual_block_id = block_table[b, block_id]

    K becomes concat([k_nope, k_rope]) along the last dim; V keeps only its
    HEAD_DIM lanes (no rope), which is why the output width differs between the
    two.
    """
    idx = sparse_indices[token, 0].to(torch.long)
    block_id = torch.div(idx, PAGE_SIZE, rounding_mode="floor")
    offset = idx % PAGE_SIZE
    physical = block_table[batch].to(torch.long)[block_id]

    k_nope = key[physical, offset, 0].to(torch.float32)
    v_gathered = value[physical, offset, 0].to(torch.float32)
    if key_rope is not None:
        k_rope = key_rope[physical, offset, 0].to(torch.float32)
        k_gathered = torch.cat([k_nope, k_rope], dim=-1)
    else:
        k_gathered = k_nope
    return k_gathered, v_gathered


def run(
    query,
    key,
    value,
    sparse_indices,
    scale_value,
    sparse_block_size=1,
    actual_seq_lengths_query=None,
    actual_seq_lengths_kv=None,
    query_rope=None,
    key_rope=None,
    layout_query='BSND',
    layout_kv='BSND',
    sparse_mode=0,
    block_table=None,
):
    """Sparse MLA flash attention, FlagTree 01-sparse-flash-attn-tle.py semantics.

    The shipped implementation is a fused Ascend pipeline: a PA -> BNSD gather
    that also concatenates the rope parts, then a TND attention kernel with an
    online softmax. The math it computes is:

        q_full = concat([query, query_rope])              # [T, 128, 576]
        k_full = concat([k_gathered, k_rope_gathered])    # [T, 128, 576]
        qk     = q_full @ k_full.T * scale_value
        p      = exp(qk - rowmax)                         # online softmax
        out    = (p @ v_gathered) / p.sum()               # [T, 128, 512]

    Details carried over from the kernels:
      * the gather maps a sparse index through block_table
        (page = idx // 128, offset = idx % 128), so indices address logical KV
        positions, not physical pages;
      * K is widened by its rope part but V is not, hence the 576 vs 512 split
        between the score and the output projections;
      * `scale_value` multiplies the raw QK before the row max is taken;
      * `actual_seq_lengths_query` is *cumulative* (the source's
        trans_tnd_actseq differences it), while `actual_seq_lengths_kv` is
        per-request; batch b therefore owns query tokens
        [cum[b-1], cum[b]).

    The delivery validates its Triton kernel against
    torch_npu.npu_sparse_flash_attention. That vendor op is not used here: it is
    the fused operator this entry benchmarks, so calling it would make the
    baseline the thing under optimization. It was used during development as an
    independent cross-check instead.

    Accumulation is float32 and the output is cast once to query's dtype.
    """
    if layout_query != 'TND' or layout_kv != 'PA_BSND':
        raise NotImplementedError("only layout_query=TND and layout_kv=PA_BSND")
    if sparse_mode not in (0, 3):
        raise NotImplementedError("only sparse_mode 0 and 3 are covered")
    if sparse_block_size != 1:
        raise NotImplementedError("only sparse_block_size=1 is covered")
    if block_table is None:
        raise ValueError("PA_BSND layout requires block_table")

    total_tokens, heads_q, head_dim = query.shape
    out = torch.zeros(
        total_tokens, heads_q, head_dim, dtype=torch.float32, device=query.device
    )

    cumulative = [int(v) for v in actual_seq_lengths_query.detach().cpu().tolist()]
    # cumulative -> per-request token counts, matching trans_tnd_actseq
    lengths, previous = [], 0
    for value_ in cumulative:
        lengths.append(value_ - previous)
        previous = value_

    token = 0
    for batch, length in enumerate(lengths):
        for _ in range(length):
            if token >= total_tokens:
                break
            k_gathered, v_gathered = _gather_sparse_kv(
                key, value, key_rope, sparse_indices, block_table, token, batch
            )
            q_full = query[token].to(torch.float32)
            if query_rope is not None:
                q_full = torch.cat([q_full, query_rope[token].to(torch.float32)], -1)
            scores = (q_full @ k_gathered.transpose(0, 1)) * scale_value
            probs = torch.softmax(scores, dim=-1)
            out[token] = probs @ v_gathered
            token += 1

    return out.to(query.dtype)


def _scalars(ctx):
    """Unwrap ctx["inputs"] recipe entries to plain values."""
    out = {}
    for name, raw in ctx["inputs"].items():
        out[name] = raw["value"] if isinstance(raw, dict) and "value" in raw else raw
    return out


def gen_inputs(ctx, device):
    """Build the paged sparse-MLA inputs.

    Transcribed from flash_mla_sparse_fwd.py::test_op, which pins the geometry:
    Q_N=128 query heads, KV_N=1 (MLA's compressed KV), D=512, D_rope=64, page
    size 128, sparse_indices as consecutive [0, sparse_size), cumulative
    actual_seq_lengths_query = arange(1, B+1), and a dense block_table.

    Note test_op sets query_rope/key_rope to None when D_rope == 0, but this
    machine's torch_npu.npu_sparse_flash_attention rejects that with
    "rope_head_dim should be 64, but got 0" -- the rope width is not actually
    optional on this hardware revision. The no-rope variant is therefore
    expressed as zero-filled rope tensors of width 64, which is numerically the
    same as an unrotated key while staying inside the vendor contract used for
    cross-checking.
    """
    spec = _scalars(ctx)
    total_tokens = int(spec["T"])
    batch = int(spec["B"])
    kv_len = int(spec["KV_S"])
    sparse_size = int(spec["sparse_size"])
    rope_dim = int(spec.get("D_rope", ROPE_DIM))
    scale_value = float(spec.get("scale_value", 0.5))
    block_size = int(spec.get("block_size", PAGE_SIZE))
    act_kv = int(spec.get("act_kv_s", kv_len))
    dtype = getattr(torch, spec.get("dtype", "float16"))

    if (batch * kv_len) % block_size != 0:
        raise ValueError("B * KV_S must be divisible by block_size")
    if sparse_size > kv_len:
        raise ValueError("sparse_size must not exceed KV_S")

    generator = torch.Generator(device="cpu").manual_seed(int(ctx["seed"]))
    pages = batch * kv_len // block_size

    def normal(*shape):
        return torch.empty(*shape, dtype=torch.float32).normal_(
            mean=0.0, std=0.5, generator=generator
        )

    query = normal(total_tokens, Q_HEADS, HEAD_DIM).to(device=device, dtype=dtype)
    key = normal(pages, block_size, KV_HEADS, HEAD_DIM).to(
        device=device, dtype=dtype
    )
    value = key.clone()

    sparse_indices = (
        torch.arange(sparse_size, dtype=torch.int32)
        .view(1, 1, -1)
        .expand(total_tokens, KV_HEADS, -1)
        .contiguous()
        .to(device)
    )
    actual_seq_lengths_query = torch.arange(
        1, batch + 1, dtype=torch.int32, device=device
    )
    actual_seq_lengths_kv = torch.tensor(
        [act_kv] * batch, dtype=torch.int32, device=device
    )
    block_table = (
        torch.arange(pages, dtype=torch.int32).reshape(batch, -1).to(device)
    )

    # The vendor cross-check requires a 64-wide rope; a "no rope" case is
    # expressed as zeros rather than as a 0-width or absent tensor.
    if rope_dim == 0:
        query_rope = torch.zeros(
            total_tokens, Q_HEADS, ROPE_DIM, dtype=dtype, device=device
        )
        key_rope = torch.zeros(
            pages, block_size, KV_HEADS, ROPE_DIM, dtype=dtype, device=device
        )
    else:
        query_rope = normal(total_tokens, Q_HEADS, rope_dim).to(
            device=device, dtype=dtype
        )
        key_rope = normal(pages, block_size, KV_HEADS, rope_dim).to(
            device=device, dtype=dtype
        )

    return {
        "query": query,
        "key": key,
        "value": value,
        "sparse_indices": sparse_indices,
        "scale_value": scale_value,
        "sparse_block_size": 1,
        "actual_seq_lengths_query": actual_seq_lengths_query,
        "actual_seq_lengths_kv": actual_seq_lengths_kv,
        "query_rope": query_rope,
        "key_rope": key_rope,
        "layout_query": "TND",
        "layout_kv": "PA_BSND",
        "sparse_mode": int(spec.get("sparse_mode", 0)),
        "block_table": block_table,
    }
