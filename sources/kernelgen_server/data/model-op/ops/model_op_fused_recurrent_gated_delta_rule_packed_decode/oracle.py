REFERENCE_DEVICE = 'target'

import torch

# Linear threshold above which softplus(x) is replaced by x, matching the
# SOFTPLUS_THRESHOLD constexpr the wrapper passes to the kernel.
_SOFTPLUS_THRESHOLD = 20.0


def run(mixed_qkv, a, b, A_log, dt_bias, scale, initial_state, out,
        ssm_state_indices, use_qk_l2norm_in_kernel=False):
    """One fused GDN decode step, straight off the packed QKV projection.

    Semantics transcribed from
    vllm/model_executor/layers/fla/ops/fused_recurrent.py
    ::fused_recurrent_gated_delta_rule_packed_decode_kernel.

    Shapes are inferred exactly as the wrapper does:
        HV, V, K = initial_state.shape[-3:]
        qk_dim   = mixed_qkv.shape[1] - HV * V
        q_dim    = qk_dim // 2
        H        = q_dim // K
    so mixed_qkv packs [q | k | v] as 2*H*K + HV*V along the last dim.

    Head mapping is grouped value attention: the kernel uses
    i_h = i_hv // (HV // H), i.e. value head i_hv reads key/query head
    i_hv // ratio. That is a repeat_interleave of q/k by ratio over the head
    axis, the same expansion the non-packed reference performs.

    Per sequence n with slot = ssm_state_indices[n]:
      * slot <= 0 is NULL_BLOCK_ID / PAD_SLOT_ID: the kernel stores zeros into
        out[n] and returns WITHOUT touching the state. Both halves matter -- a
        pad slot must not leave stale values in out, and must not corrupt state.
      * otherwise: state = initial_state[slot]
            state *= exp(g)
            v     -= state @ k          (prediction for k)
            v     *= beta
            state += outer(v, k)        (rank-1 delta-rule update)
            out[n] = state @ q          (read out AFTER the update)
            initial_state[slot] = state

    Fused gating, in float32:
        x        = a + dt_bias
        softplus = x <= 20 ? log1p(exp(x)) : x
        g        = -exp(A_log) * softplus
        beta     = sigmoid(b), rounded through b.dtype and back to float32
    The beta round-trip through b's dtype is deliberate in the kernel
    (`tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)`); dropping it
    changes low bits for float16/bfloat16 inputs.

    The whole recurrence runs in float32 and is cast once on store, into out's
    dtype and initial_state's dtype respectively.

    Both out and initial_state are mutated in place and returned, so the two
    logical outputs alias those two inputs.
    """
    B = mixed_qkv.shape[0]
    HV, V, K = initial_state.shape[-3:]

    qk_dim = mixed_qkv.shape[1] - HV * V
    q_dim = qk_dim // 2
    H = q_dim // K
    ratio = HV // H

    # Unpack q | k | v out of the packed projection. mixed_qkv may be a strided
    # view into a wider buffer (stride(0) > shape[1]); slicing handles that, the
    # kernel handles it via stride_mixed_qkv_tok.
    q = mixed_qkv[:, :q_dim].float().reshape(B, H, K)
    k = mixed_qkv[:, q_dim:2 * q_dim].float().reshape(B, H, K)
    v = mixed_qkv[:, 2 * q_dim:].float().reshape(B, HV, V)

    if use_qk_l2norm_in_kernel:
        # epsilon inside the sqrt, as in the kernel
        q = q / (q.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
        k = k / (k.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
    q = q * scale

    if ratio > 1:
        q = q.repeat_interleave(ratio, dim=1)
        k = k.repeat_interleave(ratio, dim=1)

    x = a.float() + dt_bias.float()
    softplus_x = torch.where(
        x <= _SOFTPLUS_THRESHOLD,
        torch.log1p(torch.exp(torch.clamp(x, max=_SOFTPLUS_THRESHOLD))),
        x,
    )
    g = -torch.exp(A_log.float()) * softplus_x
    beta = torch.sigmoid(b.float()).to(b.dtype).float()

    idx = ssm_state_indices.long()
    valid = idx > 0

    out_view = out.view(B, HV, V)

    pad_rows = (~valid).nonzero(as_tuple=True)[0]
    if pad_rows.numel():
        out_view[pad_rows] = 0

    rows = valid.nonzero(as_tuple=True)[0]
    if rows.numel():
        slots = idx[rows]
        state = initial_state[slots].float()                      # [n, HV, V, K]
        state = state * g[rows].exp()[:, :, None, None]
        pred = torch.einsum("nhvk,nhk->nhv", state, k[rows])
        v_corr = (v[rows] - pred) * beta[rows].unsqueeze(-1)
        state = state + v_corr.unsqueeze(-1) * k[rows].unsqueeze(-2)
        out_view[rows] = torch.einsum("nhvk,nhk->nhv", state, q[rows]).to(out.dtype)
        initial_state[slots] = state.to(initial_state.dtype)

    return out, initial_state


def gen_inputs(ctx, device):
    """Materialize the packed decode inputs, including the layouts under test.

    A plain recipe cannot express any of these:
      * mixed_qkv is one packed buffer whose q/k/v split is determined by the
        other tensors' shapes, and the kernel explicitly supports it being a
        strided view into a wider projection buffer;
      * ssm_state_indices must be valid slot ids into initial_state, with
        PAD entries (<= 0) to exercise the skip branch;
      * out must be a contiguous [B, 1, HV, V] buffer the operator writes into;
      * initial_state is paged: its first dim is a slot count, not the batch.

    Layout construction follows
    vllm/tests/kernels/test_fused_recurrent_packed_decode.py.
    """
    cfg = ctx["inputs"]["gdn_config"]
    B = int(cfg["batch"])
    H = int(cfg["h"])
    HV = int(cfg["hv"])
    K = int(cfg["k_dim"])
    V = int(cfg["v_dim"])
    dtype = getattr(torch, cfg["dtype"])
    strided = bool(cfg.get("strided_mixed_qkv", False))
    num_slots = int(cfg.get("num_slots", B + 1))
    pad_count = int(cfg.get("pad_count", 0))
    # index_base=0 reproduces the upstream test exactly (indices = arange(B), so
    # slot 0 is itself treated as a pad because the guard is `<= 0`).
    # index_base=1 is the ordinary paged case where every non-pad slot is live.
    index_base = int(cfg.get("index_base", 1))

    gen = torch.Generator(device="cpu").manual_seed(int(ctx.get("seed", 0)))
    qkv_dim = 2 * (H * K) + (HV * V)

    def rnd(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    if strided:
        # stride(0) > shape[1]: a packed view into a larger projection buffer.
        proj = rnd(B, qkv_dim + 64).to(device=device, dtype=dtype)
        mixed_qkv = proj[:, :qkv_dim]
    else:
        mixed_qkv = rnd(B, qkv_dim).to(device=device, dtype=dtype)

    a = rnd(B, HV).to(device=device, dtype=dtype)
    b = rnd(B, HV).to(device=device, dtype=dtype)
    A_log = rnd(HV).to(device=device, dtype=dtype)
    dt_bias = rnd(HV).to(device=device, dtype=dtype)

    # Distinct slots per sequence: the kernel writes the state back per slot, so
    # sharing a slot between two sequences in the same batch would make the
    # result order-dependent. Continuous batching never does that.
    indices = (torch.arange(B, dtype=torch.int32) + index_base)
    if pad_count:
        indices[-pad_count:] = -1
    ssm_state_indices = indices.to(device)

    initial_state = rnd(num_slots, HV, V, K).to(device=device, dtype=dtype)
    out = torch.empty(B, 1, HV, V, device=device, dtype=dtype)

    return {
        "mixed_qkv": mixed_qkv,
        "a": a,
        "b": b,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "scale": K ** -0.5,
        "initial_state": initial_state,
        "out": out,
        "ssm_state_indices": ssm_state_indices,
    }
