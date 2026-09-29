REFERENCE_DEVICE = 'target'

import math

import torch

Q_HEADS = 64
KV_HEADS = 1
HEAD_DIM = 512
PAGE_SIZE = 128
SOFTMAX_SCALE = 0.04419417
SEED = 43

# Query rows are scored in chunks so the [rows, window+topk] score tile stays
# bounded at Q=8192.
_ROW_CHUNK = 256


def _logical_kv(kv, block_table, batch, token_count):
    """Resolve a PA_ND paged cache into a logical [token_count, HEAD_DIM] view."""
    if token_count <= 0:
        return kv.new_zeros((0, kv.shape[-1]), dtype=torch.float32)
    page_count = math.ceil(token_count / kv.shape[1])
    pages = block_table[batch, :page_count].to(torch.long)
    flat = kv.index_select(0, pages).reshape(-1, kv.shape[-1])
    return flat[:token_count].to(torch.float32)


def run(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    seqused_kv=None,
    sinks=None,
    softmax_scale=0.04419417,
    cmp_ratio=0,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="TND",
    layout_kv="PA_ND",
):
    """SparseAttnSharedKV, FlagGems-vllm PR #792 semantics.

    Provenance note: this delivery ships no torch golden for this operator. Its
    test compares the Triton kernels against the vLLM-Ascend AscendC custom op
    (torch.ops._C_ascend.npu_sparse_attn_sharedkv) and skips when that op is
    absent, so there is nothing to transcribe. This reference is therefore
    derived from the shipped kernels' own arithmetic, specifically the decode
    compact-attention kernels and the prefill CFA staged QK/softmax kernel:

      * shared KV: an entry of ori_kv / cmp_kv is used as both key and value,
        which is what makes the cache "shared";
      * query position: q_position = (kv_len - q_len) + t, matching the kernels'
        `token + Q_POSITION_OFFSET` with the offset set to the sequence's q_start;
      * window (all modes): original tokens in
        [q_position - ori_win_left, q_position + ori_win_right], clamped to
        [0, kv_len); at ori_win_left=127 / ori_win_right=0 this is the 128-token
        causal window the pack kernels build;
      * CFA: plus the contiguous compressed prefix, positions
        [0, (q_position + 1) // cmp_ratio);
      * SCFA: plus the compressed positions named by cmp_sparse_indices[t, 0],
        keeping only entries with 0 <= idx < (q_position + 1) // cmp_ratio --
        the kernels' `valid = (sparse_idx >= 0) & (sparse_idx < cmp_threshold)`;
      * sink: one softmax spans sink, window and compressed scores. The sink is a
        logit contributing exp(sink - row_max) to the denominator and no value,
        i.e. the kernels' `row_sum` initialized to 1.0 against `row_max = sink`,
        equivalently the staged kernel's explicit
        `denom = exp(sink - row_max) + sum(ori) + sum(cmp)`.

    Scores are computed in float32 and the output is cast back to q's dtype. A
    query row whose visible set is empty except the sink yields a zero row, since
    the sink carries no value.
    """
    if layout_q != "TND" or layout_kv != "PA_ND":
        raise NotImplementedError("only layout_q=TND and layout_kv=PA_ND are supported")
    if ori_mask_mode != 4 or cmp_mask_mode != 3:
        raise ValueError("ori_mask_mode and cmp_mask_mode must be 4 and 3")
    if ori_win_left != 127 or ori_win_right != 0:
        raise ValueError("ori_win_left and ori_win_right must be 127 and 0")
    if q.dim() != 3 or tuple(q.shape[1:]) != (Q_HEADS, HEAD_DIM):
        raise ValueError("q must have shape [T, 64, 512]")

    device = q.device
    total_q = int(q.shape[0])
    out = torch.zeros((total_q, Q_HEADS, HEAD_DIM), device=device, dtype=torch.float32)
    sink = sinks.to(torch.float32).reshape(1, Q_HEADS)

    cu = [int(v) for v in cu_seqlens_q.detach().cpu().tolist()]
    kv_lens = [int(v) for v in seqused_kv.detach().cpu().tolist()]
    has_cmp = cmp_kv is not None
    has_sparse = has_cmp and cmp_sparse_indices is not None

    for batch in range(len(kv_lens)):
        q_start, q_end = cu[batch], cu[batch + 1]
        q_len, kv_len = q_end - q_start, kv_lens[batch]
        if q_len <= 0:
            continue
        position_offset = kv_len - q_len

        ori_logical = _logical_kv(ori_kv, ori_block_table, batch, kv_len)
        cmp_visible_max = kv_len // cmp_ratio if (has_cmp and cmp_ratio) else 0
        cmp_logical = (
            _logical_kv(cmp_kv, cmp_block_table, batch, cmp_visible_max)
            if cmp_visible_max
            else None
        )

        window = ori_win_left + ori_win_right + 1
        for base in range(0, q_len, _ROW_CHUNK):
            rows = torch.arange(
                base, min(base + _ROW_CHUNK, q_len), device=device, dtype=torch.long
            )
            positions = rows + position_offset
            query = q[q_start + rows].to(torch.float32)

            # ---- sliding window over the original KV ----
            offsets = torch.arange(window, device=device, dtype=torch.long)
            ori_pos = (positions - ori_win_left)[:, None] + offsets[None, :]
            ori_ok = (ori_pos >= 0) & (ori_pos < kv_len)
            safe_ori = torch.where(ori_ok, ori_pos, torch.zeros_like(ori_pos))
            ori_kv_tile = ori_logical[safe_ori.reshape(-1)].reshape(
                rows.numel(), window, HEAD_DIM
            )
            ori_scores = (
                torch.einsum("rhd,rnd->rhn", query, ori_kv_tile) * softmax_scale
            )

            # ---- compressed contribution ----
            if has_sparse:
                sparse = cmp_sparse_indices[q_start + rows, 0].to(torch.long)
                threshold = torch.div(
                    positions + 1, cmp_ratio, rounding_mode="floor"
                )
                cmp_ok = (sparse >= 0) & (sparse < threshold[:, None])
                safe_cmp = torch.where(cmp_ok, sparse, torch.zeros_like(sparse))
            elif has_cmp:
                threshold = torch.div(
                    positions + 1, cmp_ratio, rounding_mode="floor"
                )
                span = int(threshold.max().item()) if threshold.numel() else 0
                span = max(span, 1)
                safe_cmp = torch.arange(span, device=device, dtype=torch.long)[
                    None, :
                ].expand(rows.numel(), span)
                cmp_ok = safe_cmp < threshold[:, None]
                safe_cmp = torch.where(cmp_ok, safe_cmp, torch.zeros_like(safe_cmp))
            else:
                safe_cmp = cmp_ok = None

            if safe_cmp is not None and cmp_logical is not None:
                in_range = safe_cmp < cmp_logical.shape[0]
                cmp_ok = cmp_ok & in_range
                safe_cmp = torch.where(in_range, safe_cmp, torch.zeros_like(safe_cmp))
                cmp_kv_tile = cmp_logical[safe_cmp.reshape(-1)].reshape(
                    rows.numel(), safe_cmp.shape[1], HEAD_DIM
                )
                cmp_scores = (
                    torch.einsum("rhd,rnd->rhn", query, cmp_kv_tile) * softmax_scale
                )
            else:
                cmp_kv_tile = cmp_scores = None

            # ---- one softmax across sink, window and compressed ----
            neg_inf = float("-inf")
            ori_scores = torch.where(
                ori_ok[:, None, :], ori_scores, torch.full_like(ori_scores, neg_inf)
            )
            row_max = ori_scores.amax(dim=-1)
            if cmp_scores is not None:
                cmp_scores = torch.where(
                    cmp_ok[:, None, :], cmp_scores, torch.full_like(cmp_scores, neg_inf)
                )
                row_max = torch.maximum(row_max, cmp_scores.amax(dim=-1))
            row_max = torch.maximum(row_max, sink)
            row_max = torch.where(
                torch.isfinite(row_max), row_max, torch.zeros_like(row_max)
            )

            ori_weights = torch.exp(ori_scores - row_max[:, :, None])
            ori_weights = torch.where(
                ori_ok[:, None, :], ori_weights, torch.zeros_like(ori_weights)
            )
            denom = torch.exp(sink - row_max) + ori_weights.sum(dim=-1)
            numerator = torch.einsum("rhn,rnd->rhd", ori_weights, ori_kv_tile)
            if cmp_scores is not None:
                cmp_weights = torch.exp(cmp_scores - row_max[:, :, None])
                cmp_weights = torch.where(
                    cmp_ok[:, None, :], cmp_weights, torch.zeros_like(cmp_weights)
                )
                denom = denom + cmp_weights.sum(dim=-1)
                numerator = numerator + torch.einsum(
                    "rhn,rnd->rhd", cmp_weights, cmp_kv_tile
                )

            out[q_start + rows] = numerator / denom[:, :, None]

    return out.to(dtype=q.dtype), torch.empty(
        (0,), dtype=torch.float32, device=device
    )


# Case registry transcribed from tests/sparse_attn_sharedkv_utils.py::CASES:
# name -> (mode, q_len, kv_len, cmp_ratio, topk)
_CASES = {
    "scfa_decode": ("SCFA", 1, 8193, 4, 512),
    "swa_decode": ("SWA", 1, 8193, 0, 0),
    "cfa_decode": ("CFA", 1, 8193, 128, 0),
    "scfa_prefill": ("SCFA", 8192, 8192, 4, 512),
    "swa_prefill": ("SWA", 8192, 8192, 0, 0),
    "cfa_prefill": ("CFA", 8192, 8192, 128, 0),
}


def _paged_tensor(token_count, generator, device):
    """Transcribed from the case helper: a shuffled page table over a random cache."""
    page_count = math.ceil(token_count / PAGE_SIZE)
    block_table = torch.randperm(
        page_count, generator=generator, dtype=torch.int32, device="cpu"
    ).view(1, page_count)
    kv = torch.rand(
        (page_count, PAGE_SIZE, KV_HEADS, HEAD_DIM),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    return (kv * 15 - 5).to(torch.bfloat16).to(device), block_table.to(device)


def _scalars(ctx):
    """Unwrap ctx["inputs"] recipe entries to plain values."""
    out = {}
    for name, raw in ctx["inputs"].items():
        out[name] = raw["value"] if isinstance(raw, dict) and "value" in raw else raw
    return out


def gen_inputs(ctx, device):
    """Build one registered case.

    Transcribed from tests/sparse_attn_sharedkv_utils.py::make_inputs, minus the
    `metadata` entry: that input is produced by a second vendor custom op
    (npu_sparse_attn_sharedkv_metadata) and is consumed only by the vendor
    reference and the graph-safe dispatch path, not by the attention math.

    Generation is seeded on CPU so the paged tables and KV are reproducible
    regardless of device RNG. The sparse index rows mirror the source exactly:
    for decode every row sees the whole compressed prefix, while for prefill row
    q_pos sees (q_pos + 1) // cmp_ratio compressed tokens, and rows with fewer
    visible tokens than topk keep -1 in the tail.
    """
    spec = _scalars(ctx)
    mode, q_len, kv_len, cmp_ratio, topk = _CASES[spec["case"]]
    has_cmp = mode != "SWA"
    has_sparse = mode == "SCFA"

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(ctx.get("seed", SEED)))

    query = torch.rand(
        (q_len, Q_HEADS, HEAD_DIM), generator=generator, dtype=torch.float32,
        device="cpu",
    )
    query = (query * 20 - 10).to(torch.bfloat16).to(device)
    ori_kv, ori_block_table = _paged_tensor(kv_len, generator, device)
    cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
    seqused_kv = torch.tensor([kv_len], dtype=torch.int32, device=device)

    cmp_kv = cmp_block_table = cmp_sparse_indices = None
    if has_cmp:
        cmp_kv, cmp_block_table = _paged_tensor(
            kv_len // cmp_ratio, generator, device
        )
        if has_sparse:
            indices = torch.full(
                (q_len, KV_HEADS, topk), -1, dtype=torch.int32, device="cpu"
            )
            for q_pos in range(q_len):
                visible = (
                    kv_len // cmp_ratio if q_len == 1 else (q_pos + 1) // cmp_ratio
                )
                count = min(visible, topk)
                if count:
                    indices[q_pos, 0, :count] = torch.randperm(
                        visible, generator=generator, dtype=torch.int32, device="cpu"
                    )[:count]
            cmp_sparse_indices = indices.to(device)

    sinks = torch.rand(
        (Q_HEADS,), generator=generator, dtype=torch.float32, device="cpu"
    ).to(device)

    return {
        "q": query,
        "ori_kv": ori_kv,
        "cmp_kv": cmp_kv,
        "cmp_sparse_indices": cmp_sparse_indices,
        "ori_block_table": ori_block_table,
        "cmp_block_table": cmp_block_table,
        "cu_seqlens_q": cu_seqlens_q,
        "seqused_kv": seqused_kv,
        "sinks": sinks,
        "softmax_scale": SOFTMAX_SCALE,
        "cmp_ratio": cmp_ratio,
        "ori_mask_mode": 4,
        "cmp_mask_mode": 3,
        "ori_win_left": 127,
        "ori_win_right": 0,
        "layout_q": "TND",
        "layout_kv": "PA_ND",
    }
