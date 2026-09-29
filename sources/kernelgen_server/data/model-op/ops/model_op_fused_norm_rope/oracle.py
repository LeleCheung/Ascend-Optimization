REFERENCE_DEVICE = 'target'

import torch


def _rms_norm(x, w, eps):
    """kernels.py::_rms_norm — float32 mean-square, rsqrt, then weight."""
    x32 = x.to(torch.float32)
    mean_sq = (x32 * x32).sum(-1, keepdim=True) / x32.shape[-1]
    rrms = torch.rsqrt(mean_sq + eps)
    return (x32 * rrms) * w.to(torch.float32)


def _cos_sin(cache, positions, half_rot):
    """kernels.py::_get_cos_sin — cos in [0, half), sin in [half, 2*half)."""
    rows = cache.index_select(0, positions.to(torch.long)).to(torch.float32)
    return rows[:, :half_rot], rows[:, half_rot : 2 * half_rot]


def run(
    positions,
    q_c,
    q_rms_norm_w,
    q_rms_eps,
    kv_c,
    kv_rms_norm_w,
    kv_rms_eps,
    k_pe,
    k_rope_cos_sin_cache,
    index_k,
    index_k_layer_norm_w,
    index_k_layer_norm_bias,
    index_k_layer_norm_eps,
    index_k_rope_cos_sin_cache,
    topk_indices_buffer,
    has_indexer=True,
    index_rope_interleave=False,
):
    """Fused norm + RoPE for DeepSeek-V3.2 MLA, vLLM kernels.py semantics.

    Transcribed from _fused_norm_rope_kernel, whose work is split across four
    program ids. This reference reproduces the three that transform tensors:

      pid 2 — Q RMSNorm: _rms_norm(q_c, q_rms_norm_w, q_rms_eps) over q_dim.
              Runs for every row, including padding rows (the kernel comments
              that the slot-based skip must not gate it, since queries are not
              sharded under DCP).

      pid 1 — KV RMSNorm plus interleaved RoPE on k_pe. The rotation reads
              adjacent pairs (`dim_off * 2` and `dim_off * 2 + 1`), not split
              halves:
                  r1 = x1 * cos - x2 * sin
                  r2 = x2 * cos + x1 * sin
              and writes them back interleaved.

      pid 0 — indexer K: LayerNorm (mean/var over index_k_dim, affine w and
              bias) followed by RoPE over the leading 2 * half_rot lanes.
              The kernel rotates each lane against its partner:
                * interleaved: partner = lane ^ 1, sign -1 on even lanes;
                * NeoX: partner = lane ^ half_rot, sign -1 on the low half;
              with `roped = normed * cos + sign * normed_partner * sin`, and
              lanes outside the rotation region passing through unrotated.
              cos/sin are indexed by lane // 2 (interleaved) or lane % half_rot
              (NeoX).

      pid 3 — fills topk_indices_buffer with -1 when has_indexer; a shared
              (non-indexer) layer deliberately leaves the previous layer's
              buffer untouched.

    With has_indexer false the kernel substitutes dummy indexer tensors and
    skips pid 0 and pid 3 entirely, so index_k_out is returned unchanged and the
    topk buffer is preserved.

    Out of scope, and therefore not implemented here: the paged cache writes
    (slot_mapping / indexer_k_cache / mla_kv_cache). Those target external
    caches and branch across several fp8 layouts (ue8m0, ds_mla per-tile scales,
    plain e4m3), which is a separate operator's worth of contract.

    All norms and rotations are evaluated in float32 and cast once on store.
    """
    num_tokens = positions.shape[0]
    if positions.dim() != 1:
        raise AssertionError("positions must be rank 1")
    for name, tensor in (("q_c", q_c), ("kv_c", kv_c), ("k_pe", k_pe)):
        if tensor.dim() != 2:
            raise AssertionError(f"{name} must be rank 2")
    if topk_indices_buffer.dim() != 2:
        raise AssertionError("topk_indices_buffer must be rank 2")

    # ---- pid 2: Q RMSNorm ----
    q_c_out = _rms_norm(q_c, q_rms_norm_w, q_rms_eps).to(q_c.dtype)

    # ---- pid 1: KV RMSNorm + interleaved RoPE on k_pe ----
    kv_c_out = _rms_norm(kv_c, kv_rms_norm_w, kv_rms_eps).to(kv_c.dtype)

    half_rot = k_pe.shape[-1] // 2
    cos, sin = _cos_sin(k_rope_cos_sin_cache, positions, half_rot)
    kpe32 = k_pe.to(torch.float32)
    x1 = kpe32[:, 0::2]
    x2 = kpe32[:, 1::2]
    r1 = x1 * cos - x2 * sin
    r2 = x2 * cos + x1 * sin
    k_pe_out = torch.empty_like(kpe32)
    k_pe_out[:, 0::2] = r1
    k_pe_out[:, 1::2] = r2
    k_pe_out = k_pe_out.to(k_pe.dtype)

    # ---- pid 0 + pid 3: indexer K, and the top-k buffer fill ----
    if not has_indexer:
        return q_c_out, kv_c_out, k_pe_out, index_k, topk_indices_buffer

    index_k_dim = index_k.shape[-1]
    idx32 = index_k.to(torch.float32)
    mean = idx32.mean(-1, keepdim=True)
    diff = idx32 - mean
    var = (diff * diff).mean(-1, keepdim=True)
    rstd = torch.rsqrt(var + index_k_layer_norm_eps)
    weight = index_k_layer_norm_w.to(torch.float32)
    bias = index_k_layer_norm_bias.to(torch.float32)
    normed = diff * rstd * weight + bias

    idx_half = index_k_rope_cos_sin_cache.shape[-1] // 2
    lane = torch.arange(index_k_dim, device=index_k.device)
    in_rope = lane < 2 * idx_half
    if index_rope_interleave:
        cos_idx = torch.div(lane, 2, rounding_mode="floor")
        partner = torch.where(in_rope, torch.bitwise_xor(lane, 1), lane)
        sign = torch.where(lane % 2 == 0, -1.0, 1.0)
    else:
        cos_idx = lane % idx_half
        partner = torch.where(in_rope, torch.bitwise_xor(lane, idx_half), lane)
        sign = torch.where(lane < idx_half, -1.0, 1.0)

    idx_rows = index_k_rope_cos_sin_cache.index_select(
        0, positions.to(torch.long)
    ).to(torch.float32)
    cos_full = torch.where(in_rope, idx_rows[:, cos_idx], torch.ones(()))
    sin_full = torch.where(
        in_rope, idx_rows[:, idx_half + cos_idx], torch.zeros(())
    )
    # The kernel re-derives the partner's normalized value from the raw input
    # using the same per-token mean/rstd, which is what `normed` already holds.
    normed_partner = normed[:, partner]
    roped = normed * cos_full + sign * normed_partner * sin_full
    index_k_out = torch.where(in_rope, roped, normed).to(index_k.dtype)

    topk_indices_buffer.fill_(-1)
    return q_c_out, kv_c_out, k_pe_out, index_k_out, topk_indices_buffer


def _scalars(ctx):
    """Unwrap ctx["inputs"] recipe entries to plain values."""
    out = {}
    for name, raw in ctx["inputs"].items():
        out[name] = raw["value"] if isinstance(raw, dict) and "value" in raw else raw
    return out


def gen_inputs(ctx, device):
    """Build the position-indexed RoPE caches and matching activations.

    A plain recipe cannot express these: the cos/sin caches must be genuine
    inverse-frequency tables whose rows are indexed by `positions`, and
    `positions` must stay inside the cache. The tables are built the standard
    way -- inv_freq = base ** -(arange(0, rot, 2) / rot), outer product with the
    position index, then cat([cos, sin]) along the last dim -- matching the
    layout _get_cos_sin reads.
    """
    spec = _scalars(ctx)
    num_tokens = int(spec["num_tokens"])
    q_dim = int(spec["q_dim"])
    kv_dim = int(spec["kv_dim"])
    rot_dim = int(spec["rot_dim"])
    index_k_dim = int(spec["index_k_dim"])
    index_rot_dim = int(spec["index_rot_dim"])
    topk = int(spec["topk"])
    max_pos = int(spec["max_position"])
    dtype = getattr(torch, spec["dtype"])
    base = float(spec.get("base", 10000.0))

    gen = torch.Generator(device="cpu").manual_seed(int(ctx["seed"]))

    def table(rot, dim_max):
        inv_freq = 1.0 / (
            base ** (torch.arange(0, rot, 2, dtype=torch.float32) / rot)
        )
        pos = torch.arange(dim_max, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        return torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(device)

    positions = torch.randint(
        0, max_pos, (num_tokens,), generator=gen, dtype=torch.int64
    ).to(device)

    def rnd(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    return {
        "positions": positions,
        "q_c": rnd(num_tokens, q_dim).to(device=device, dtype=dtype),
        "q_rms_norm_w": rnd(q_dim).to(device=device, dtype=dtype),
        "kv_c": rnd(num_tokens, kv_dim).to(device=device, dtype=dtype),
        "kv_rms_norm_w": rnd(kv_dim).to(device=device, dtype=dtype),
        "k_pe": rnd(num_tokens, rot_dim).to(device=device, dtype=dtype),
        "k_rope_cos_sin_cache": table(rot_dim, max_pos),
        "index_k": rnd(num_tokens, index_k_dim).to(device=device, dtype=dtype),
        "index_k_layer_norm_w": rnd(index_k_dim).to(device),
        "index_k_layer_norm_bias": rnd(index_k_dim).to(device),
        "index_k_rope_cos_sin_cache": table(index_rot_dim, max_pos),
        "topk_indices_buffer": torch.zeros(
            num_tokens, topk, dtype=torch.int32, device=device
        ),
    }
