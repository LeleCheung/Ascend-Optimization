REFERENCE_DEVICE = 'target'

import torch

RMS_SIZE = 512
ROPE_SIZE = 64


def run(
    kv,
    gamma,
    cos,
    sin,
    index,
    k_cache,
    ckv_cache,
    k_rope_scale,
    c_kv_scale,
    k_rope_offset,
    c_kv_offset,
    epsilon,
    cache_mode,
    is_output_kv,
):
    """Fused RMSNorm + RoPE + paged cache scatter, PR #722 semantics.

    RoPE half, from _apply_rotary_pos_emb_kernel. The kernel slices the RoPE
    input with `extract_slice(x, (0, 0 | 1), ..., strides=(1, 2))`, i.e. x1 is
    the even lanes and x2 the odd lanes of kv[..., 512:], while cos/sin are
    sliced with stride 1 into a contiguous first and second half:

        first_half  = x1 * cos[:half] - x2 * sin[:half]
        second_half = x2 * cos[half:] + x1 * sin[half:]
        result      = cat([first_half, second_half])

    That interleaved-input / contiguous-halves asymmetry is easy to get wrong;
    it was confirmed bit-exact against torch_npu.npu_kv_rmsnorm_rope_cache.

    RMS half, from _rms_norm_kernel:

        var  = sum(x * x, axis=1) * (1 / N)
        rrms = sqrt(var + eps)          # note: the *root*, not its reciprocal
        y    = x / rrms * w

    so the division is by sqrt(var + eps) rather than a multiply by rsqrt. Both
    halves accumulate in float32 (`cdtype = tl.float32`) and cast once on store.

    Cache scatter: `index` is a flat token -> slot map, and each cache is written
    as a flat (num_slots, last_dim) array, so slot s of k_cache holds the RoPE
    result for token s and likewise for ckv_cache. Only PA / PA_BNSD is
    implemented upstream.

    Returns (k_cache, ckv_cache, k_rope, c_kv) when is_output_kv, else
    (k_cache, ckv_cache); the caches are updated in place either way.
    """
    if kv.shape[-1] != RMS_SIZE + ROPE_SIZE:
        raise ValueError(
            f"kv last dimension must be {RMS_SIZE + ROPE_SIZE}, got {kv.shape[-1]}"
        )
    if cache_mode not in ("PA", "PA_BNSD"):
        raise NotImplementedError(f"unsupported cache_mode {cache_mode!r}")
    if any(t is not None for t in (k_rope_scale, c_kv_scale, k_rope_offset, c_kv_offset)):
        raise NotImplementedError("quantized scale/offset path is not implemented")

    out_dtype = kv.dtype
    half = ROPE_SIZE // 2

    rope_in = kv[..., RMS_SIZE:].to(torch.float32)
    x1 = rope_in[..., 0::2]
    x2 = rope_in[..., 1::2]
    cos32 = cos.to(torch.float32)
    sin32 = sin.to(torch.float32)
    first_half = x1 * cos32[..., :half] - x2 * sin32[..., :half]
    second_half = x2 * cos32[..., half:] + x1 * sin32[..., half:]
    k_rope = torch.cat([first_half, second_half], dim=-1).to(out_dtype)

    rms_in = kv[..., :RMS_SIZE].to(torch.float32)
    var = (rms_in * rms_in).sum(-1, keepdim=True) * (1.0 / RMS_SIZE)
    c_kv = (rms_in / torch.sqrt(var + epsilon) * gamma.to(torch.float32)).to(out_dtype)

    slots = index.to(torch.long).reshape(-1)
    k_cache.view(-1, ROPE_SIZE)[slots] = k_rope.reshape(-1, ROPE_SIZE)
    ckv_cache.view(-1, RMS_SIZE)[slots] = c_kv.reshape(-1, RMS_SIZE)

    if is_output_kv:
        return k_cache, ckv_cache, k_rope, c_kv
    return k_cache, ckv_cache



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
    """Build the fused KV projection and its paged caches.

    Transcribed from tests/test_kv_rmsnorm_rope_cache.py::_make_inputs, which
    fixes the DeepSeek-MLA head geometry (rms 512 + rope 64) and a
    PAGE_NUM x PAGE_SIZE paged cache, with `index` a flat token -> slot map.
    """
    spec = _scalars(ctx)
    batch = int(spec["batch_size"])
    seq_len = int(spec["seq_len"])
    page_num = int(spec["page_num"])
    page_size = int(spec["page_size"])
    dtype = getattr(torch, spec["dtype"])
    epsilon = float(spec.get("epsilon", 1e-5))

    torch.manual_seed(int(ctx["seed"]))

    kv = torch.randn(
        batch, 1, seq_len, RMS_SIZE + ROPE_SIZE, dtype=dtype, device=device
    )
    gamma = torch.ones(RMS_SIZE, dtype=dtype, device=device)
    cos = torch.randn(batch, 1, seq_len, ROPE_SIZE, dtype=dtype, device=device)
    sin = torch.randn(batch, 1, seq_len, ROPE_SIZE, dtype=dtype, device=device)
    k_cache = torch.zeros(page_num, page_size, 1, ROPE_SIZE, dtype=dtype, device=device)
    ckv_cache = torch.zeros(
        page_num, page_size, 1, RMS_SIZE, dtype=dtype, device=device
    )
    index = torch.arange(batch * seq_len, dtype=torch.int64, device=device)

    return {
        "kv": kv,
        "gamma": gamma,
        "cos": cos,
        "sin": sin,
        "index": index,
        "k_cache": k_cache,
        "ckv_cache": ckv_cache,
        "k_rope_scale": None,
        "c_kv_scale": None,
        "k_rope_offset": None,
        "c_kv_offset": None,
        "epsilon": epsilon,
        "cache_mode": spec.get("cache_mode", "PA_BNSD"),
        "is_output_kv": bool(spec.get("is_output_kv", True)),
    }
