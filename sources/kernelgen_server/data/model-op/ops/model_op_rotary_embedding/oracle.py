REFERENCE_DEVICE = 'target'

import torch


def _apply_rotary_emb(x, cos, sin, is_neox_style):
    """vLLM/SGLang torch-native rotary core.

    Verbatim from sgl_kernel/testing/rotary_embedding.py::_apply_rotary_emb,
    which is the function sgl-kernel/tests/test_rotary_embedding.py asserts the
    fused kernel against.

    x:   [num_tokens, num_heads, rotary_dim]
    cos: [num_tokens, rotary_dim // 2]
    sin: [num_tokens, rotary_dim // 2]
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    return torch.stack((o1, o2), dim=-1).flatten(-2)


def _rotate(t, head_size, rotary_dim, cos, sin, is_neox):
    """Rotate the leading rotary_dim lanes of every head, pass the rest through.

    Mirrors RotaryEmbedding.forward_native: compute in float32, then cast back.
    """
    orig_dtype = t.dtype
    num_tokens = t.shape[0]
    tf = t.to(torch.float32).view(num_tokens, -1, head_size)
    rot = tf[..., :rotary_dim]
    passthrough = tf[..., rotary_dim:]
    rot = _apply_rotary_emb(rot, cos, sin, is_neox)
    out = torch.cat((rot, passthrough), dim=-1)
    return out.reshape(num_tokens, -1).to(orig_dtype)


def run(positions, query, key, head_size, cos_sin_cache, is_neox=True):
    """SGLang apply_rope_with_cos_sin_cache_inplace semantics.

    The kernel writes the rotated query/key back into the caller's buffers and
    returns None, which is the ABI used by
    sglang/srt/layers/rotary_embedding.py::RotaryEmbedding.forward_cuda.

    rotary_dim comes from the cache, not from a parameter -- documented in the
    sgl_kernel docstring: "The rotary dimension is determined by the cosine cache
    and sine cache", with cos in the first half of the last dim and sin in the
    second half.

    Reference arithmetic is RotaryEmbedding.forward_native: gather cos/sin for
    the given positions, upcast query/key to float32, rotate the leading
    rotary_dim lanes of each head, leave the remaining lanes untouched, and cast
    back to the original dtype. The float32 upcast is load-bearing -- the SGLang
    source flags it as required for the embedding to be numerically correct.
    """
    head_size = int(head_size)
    rotary_dim = int(cos_sin_cache.shape[-1])

    flat_positions = positions.flatten()
    cos_sin = cos_sin_cache.to(torch.float32).index_select(0, flat_positions.long())
    cos, sin = cos_sin.chunk(2, dim=-1)

    query.copy_(_rotate(query, head_size, rotary_dim, cos, sin, is_neox))
    key.copy_(_rotate(key, head_size, rotary_dim, cos, sin, is_neox))
    return None


def gen_inputs(ctx, device):
    """Build a real RoPE cache plus matching positions.

    Needed because two inputs are not plain random tensors:
      * cos_sin_cache must be an actual inverse-frequency table, otherwise the
        rotation is not a rotation and the operator has no meaningful semantics;
      * positions must be valid indices into that table.

    Construction follows sgl_kernel/testing/rotary_embedding.py:
      RotaryEmbedding._compute_cos_sin_cache -- inv_freq = 1/(base**(arange(0,
      rotary_dim,2)/rotary_dim)), freqs = outer(arange(max_position), inv_freq),
      cache = cat([cos, sin], dim=-1) -- and create_inputs, which lays positions
      out as arange(seq_len).repeat(batch_size).
    """
    spec = ctx["inputs"]
    cfg = spec.get("rope_config")
    if not isinstance(cfg, dict):
        raise ValueError("workload must provide a rope_config context block")

    head_size = int(cfg["head_size"])
    rotary_dim = int(cfg["rotary_dim"])
    max_position = int(cfg["max_position_embeddings"])
    base = float(cfg["base"])
    batch_size = int(cfg["batch_size"])
    seq_len = int(cfg["seq_len"])
    num_q_heads = int(cfg["num_q_heads"])
    num_kv_heads = int(cfg["num_kv_heads"])
    dtype = getattr(torch, cfg["dtype"])

    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    t = torch.arange(max_position, dtype=torch.float32)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(device)

    # positions index the cache; seq_len never exceeds max_position in the
    # authoritative cases, but clamp defensively so a workload can't sample
    # out of range.
    pos = torch.arange(seq_len, device=device) % max_position
    positions = pos.repeat(batch_size).to(torch.int64)

    num_tokens = batch_size * seq_len
    generator = torch.Generator(device="cpu").manual_seed(int(ctx.get("seed", 0)))
    query = torch.randn(
        num_tokens, num_q_heads * head_size, generator=generator, dtype=torch.float32
    ).to(device=device, dtype=dtype)
    key = torch.randn(
        num_tokens, num_kv_heads * head_size, generator=generator, dtype=torch.float32
    ).to(device=device, dtype=dtype)

    return {
        "positions": positions,
        "query": query,
        "key": key,
        "head_size": head_size,
        "cos_sin_cache": cos_sin_cache,
    }
