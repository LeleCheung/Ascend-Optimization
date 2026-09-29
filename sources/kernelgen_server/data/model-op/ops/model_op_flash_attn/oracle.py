REFERENCE_DEVICE = 'target'

import torch

DIM = 64
BLOCK_N = 32

# Query rows are scored in chunks so the [chunk, S] score tile stays bounded at
# S = 4096 with many heads.
_ROW_CHUNK = 512


def run(q, k, v, combine_batch, is_causal=False):
    """Flash attention forward, FlagTree fa_triton_arch.py semantics.

    The kernel is a three-task (MM1 / Vec / MM2) Ascend schedule with GM
    ping-pong workspaces and an online softmax. Chunking, ring slots and
    `combine_batch` are all scheduling: mathematically the operator is plain
    scaled-dot-product attention over BNSD tensors with grouped KV heads, so the
    reference evaluates that directly.

    Details taken from the kernel:
      * scale is `sm_scale = (1.0 / D) ** 0.5`, i.e. sqrt(1/head_dim) rather
        than the more common 1/sqrt(head_dim) written that way -- numerically
        identical, but worth stating since the source spells it as a reciprocal
        under the root;
      * the scale multiplies the raw QK score *before* the row max is taken
        (`exp(sm_scale * score + neg_max_new)`), so masking happens on unscaled
        scores and the max is over scaled ones;
      * the causal mask is `q_row_idx >= kv_col_idx` on global indices, filled
        with -inf;
      * GQA: query head h reads KV head h // (Hq // Hkv), the kernel's
        `gqa_group` division.

    `combine_batch` only has to satisfy the kernel's divisibility constraint
    (num_kv_blocks % CB == 0, with CB clamped to num_kv_blocks); it is validated
    here so a workload cannot silently pass an illegal value, but it does not
    enter the arithmetic.

    Accumulation is float32 and the output is cast once to q's dtype.
    """
    batch, heads_q, seq_len, head_dim = q.shape
    heads_kv = k.shape[1]
    if head_dim != DIM:
        raise AssertionError(f"head_dim must be {DIM}, got {head_dim}")
    if seq_len % BLOCK_N != 0:
        raise AssertionError(f"seq_len must be a multiple of {BLOCK_N}")
    if heads_q % heads_kv != 0:
        raise AssertionError("Hq must be divisible by Hkv")

    num_kv_blocks = seq_len // BLOCK_N
    combine = int(combine_batch)
    if num_kv_blocks < combine:
        combine = num_kv_blocks
    if num_kv_blocks % combine != 0:
        raise AssertionError(
            f"num_kv_blocks ({num_kv_blocks}) must be divisible by "
            f"combine_batch ({combine})"
        )

    scale = (1.0 / head_dim) ** 0.5
    group = heads_q // heads_kv
    out = torch.empty_like(q)

    positions = torch.arange(seq_len, device=q.device)
    for b in range(batch):
        for h in range(heads_q):
            kv_head = h // group
            keys = k[b, kv_head].to(torch.float32)
            values = v[b, kv_head].to(torch.float32)
            for base in range(0, seq_len, _ROW_CHUNK):
                stop = min(base + _ROW_CHUNK, seq_len)
                query = q[b, h, base:stop].to(torch.float32)
                scores = (query @ keys.transpose(0, 1)) * scale
                if is_causal:
                    rows = positions[base:stop]
                    mask = rows[:, None] >= positions[None, :]
                    scores = scores.masked_fill(~mask, float("-inf"))
                probs = torch.softmax(scores, dim=-1)
                out[b, h, base:stop] = (probs @ values).to(q.dtype)
    return out


def _scalars(ctx):
    """Unwrap ctx["inputs"] recipe entries to plain values."""
    out = {}
    for name, raw in ctx["inputs"].items():
        out[name] = raw["value"] if isinstance(raw, dict) and "value" in raw else raw
    return out


def gen_inputs(ctx, device):
    """Build BNSD q/k/v with the kernel's GQA head split.

    A recipe cannot express the Hq/Hkv relationship between three tensors, so
    they are generated together; head_dim is pinned to the kernel's DIM = 64.
    """
    spec = _scalars(ctx)
    batch = int(spec["batch"])
    heads_q = int(spec["heads_q"])
    heads_kv = int(spec["heads_kv"])
    seq_len = int(spec["seq_len"])
    dtype = getattr(torch, spec["dtype"])

    gen = torch.Generator(device="cpu").manual_seed(int(ctx["seed"]))

    def rnd(heads):
        return torch.randn(
            batch, heads, seq_len, DIM, generator=gen, dtype=torch.float32
        ).to(device=device, dtype=dtype)

    return {"q": rnd(heads_q), "k": rnd(heads_kv), "v": rnd(heads_kv)}
