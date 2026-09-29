REFERENCE_DEVICE = 'target'

import torch


def run(q, k, v, g, beta, scale, initial_state, output_final_state,
        use_qk_l2norm_in_kernel=False):
    """Gated delta-rule recurrence, verbatim from the FlagGems-sglang reference.

    Source: FlagGems-sglang flaggems_reference/fused_recurrent_gdn.py, which is
    the pure-torch ground truth that both tests/test_fused_recurrent_gdn.py and
    benchmark/test_fused_recurrent_gdn.py compare the Triton kernel against.

    Semantics per timestep t, all in float32:
      * optional L2 norm on q and k with a 1e-6 floor inside the sqrt
        (x / sqrt(sum(x^2) + 1e-6)), i.e. the epsilon is inside, not added to the
        norm -- this is what use_qk_l2norm_in_kernel means;
      * q is scaled by `scale` after the normalization;
      * grouped value attention: when HV > H, k and q are repeat_interleave'd by
        HV // H along the head axis so each value head sees its key head;
      * state decays multiplicatively by exp(g), broadcast over (V, K);
      * delta rule: v_corr = (v - state @ k) * beta, then
        state += outer(v_corr, k);
      * output is state @ q, read out AFTER the state update.

    beta is per-head [B, T, HV] or per-channel [B, T, HV, V]; the reference
    distinguishes them by `beta.dim() == v.dim()`, and the per-head form is
    unsqueezed on the last axis to broadcast over V.

    o is returned in v's dtype; final_state stays float32 and is None when
    output_final_state is false.
    """
    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[-1]
    ratio = HV // H
    beta_headwise = beta.dim() == v.dim()

    if initial_state is not None:
        state = initial_state.float().clone()
    else:
        state = q.new_zeros(B, HV, V, K, dtype=torch.float32)

    o = q.new_zeros(B, T, HV, V, dtype=torch.float32)

    for t in range(T):
        qt = q[:, t].float()
        kt = k[:, t].float()
        vt = v[:, t].float()
        gt = g[:, t].float()

        if use_qk_l2norm_in_kernel:
            qt = qt / (qt.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
            kt = kt / (kt.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
        qt = qt * scale

        kt_e = kt.repeat_interleave(ratio, dim=1) if ratio > 1 else kt  # (B, HV, K)
        qt_e = qt.repeat_interleave(ratio, dim=1) if ratio > 1 else qt  # (B, HV, K)

        state = state * gt.exp()[:, :, None, None]

        pred = torch.einsum("bhvk,bhk->bhv", state, kt_e)
        vt_corr = vt - pred

        if beta_headwise:
            bt = beta[:, t].float()
        else:
            bt = beta[:, t].float().unsqueeze(-1)
        vt_corr = vt_corr * bt

        state = state + vt_corr.unsqueeze(-1) * kt_e.unsqueeze(-2)

        o[:, t] = torch.einsum("bhvk,bhk->bhv", state, qt_e)

    final_state = state if output_final_state else None
    return o.to(v.dtype), final_state


def gen_inputs(ctx, device):
    """Build GDN inputs with the sign/range constraints the recurrence needs.

    Plain random recipes are wrong here for three reasons, all taken from the
    FlagGems-sglang harness `_case()` helper:
      * g is a LOG decay and must be negative (-rand * 0.1); a positive g makes
        exp(g) > 1 and the state diverges over T steps;
      * beta is a gate in (0, 1), produced as sigmoid(randn);
      * initial_state is float32 [B, HV, V, K] and is present only for the cases
        that ask for it.
    scale is k_dim ** -0.5, as the caller always passes.
    """
    cfg = ctx["inputs"]["gdn_config"]
    B = int(cfg["batch"]); T = int(cfg["t"])
    H = int(cfg["h"]); HV = int(cfg["hv"])
    K = int(cfg["k_dim"]); V = int(cfg["v_dim"])
    dtype = getattr(torch, cfg["dtype"])
    beta_headwise = bool(cfg["beta_headwise"])
    has_init = bool(cfg["has_init"])

    gen = torch.Generator(device="cpu").manual_seed(int(ctx.get("seed", 0)))

    def rnd(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    q = rnd(B, T, H, K).to(device=device, dtype=dtype)
    k = rnd(B, T, H, K).to(device=device, dtype=dtype)
    v = rnd(B, T, HV, V).to(device=device, dtype=dtype)
    g = (-torch.rand(B, T, HV, generator=gen, dtype=torch.float32) * 0.1).to(device)
    if beta_headwise:
        beta = torch.sigmoid(rnd(B, T, HV, V)).to(device)
    else:
        beta = torch.sigmoid(rnd(B, T, HV)).to(device)
    initial_state = rnd(B, HV, V, K).to(device) if has_init else None

    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "scale": K ** -0.5,
        "initial_state": initial_state,
    }
