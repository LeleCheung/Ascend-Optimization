REFERENCE_DEVICE = 'target'

import torch


def _l2norm(x):
    """vllm_ascend/ops/triton/fla/l2norm.py::l2norm_fwd, eps default 1e-6.

    The kernel upcasts to float32, computes rsqrt(sum(x*x) + eps) and stores
    x * rsqrt back in the input dtype.
    """
    x32 = x.to(torch.float32)
    return x32 * torch.rsqrt((x32 * x32).sum(-1, keepdim=True) + 1e-6)


def _segment(q, k, v, g, beta, scale, state, l2norm):
    """Run the recurrence over one packed segment; returns (o, final_state).

    state is [H, K, V] float32 and is updated and returned.
    """
    t_len, heads, k_dim = q.shape
    v_dim = v.shape[-1]
    out = q.new_zeros(t_len, heads, v_dim, dtype=torch.float32)
    for step in range(t_len):
        qt = q[step].to(torch.float32)
        kt = k[step].to(torch.float32)
        vt = v[step].to(torch.float32)
        gt = g[step].to(torch.float32)
        bt = beta[step].to(torch.float32)

        if l2norm:
            qt = _l2norm(qt)
            kt = _l2norm(kt)
        qt = qt * scale

        # decay: state[h] *= exp(g[h])
        state = state * gt[:, None, None].exp()
        # prediction for k, then delta-rule correction scaled by beta
        pred = torch.einsum("hkv,hk->hv", state, kt)
        v_corr = (vt - pred) * bt[:, None]
        # rank-1 update, then read out with q AFTER the update
        state = state + kt[:, :, None] * v_corr[:, None, :]
        out[step] = torch.einsum("hkv,hk->hv", state, qt)
    return out, state


def run(
    q,
    k,
    v,
    g,
    beta,
    scale=None,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    use_qk_l2norm_in_kernel=False,
):
    """Chunked gated delta-rule forward, vLLM-Ascend chunk.py semantics.

    The shipped implementation is a chunked schedule (chunk_size=64) built from
    chunk_local_cumsum + chunk_scaled_dot_kkt + solve_tril + recompute_w_u plus
    an AscendC state kernel, and it needs a live vLLM forward context. Chunking
    is a scheduling decision: the chunked and fused-recurrent forms compute the
    same gated delta-rule recurrence, so this reference evaluates the recurrence
    directly, which is both portable and the correct performance baseline.

    Semantics, from chunk_gated_delta_rule and ChunkGatedDeltaRuleFunction:
      * g is already in log space; the state decays by exp(g) per step;
      * use_qk_l2norm_in_kernel applies l2norm_fwd to q and k *before* the
        `scale` multiply, with the eps inside the sqrt (rsqrt(sum + 1e-6));
      * scale defaults to k.shape[-1] ** -0.5 when None;
      * with cu_seqlens the batch must be 1, the sequences are packed along T,
        and initial_state carries one [H, K, V] slab per segment;
      * the recurrence accumulates in float32 and o is cast to q's dtype.

    Layout note: initial_state / final_state are [N, H, K, V] here, i.e. the
    key axis precedes the value axis. That is transposed relative to
    key_ops_fused_recurrent_gated_delta_rule, whose state is [N, HV, V, K]. The
    rank-1 update is therefore outer(k, v_corr) rather than outer(v_corr, k).

    float32 q/k/v is rejected upstream (`q.dtype != torch.float32`), so the
    workloads use bfloat16 and float16 only.
    """
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise AssertionError("q, k and v must share a dtype")
    if q.dtype == torch.float32:
        raise AssertionError(
            "chunk_gated_delta_rule does not support float32; use bfloat16"
        )
    if beta.dim() != 3:
        raise AssertionError("beta must be of shape [B, T, H]")

    if scale is None:
        scale = k.shape[-1] ** -0.5

    batch, t_len, heads, k_dim = q.shape
    v_dim = v.shape[-1]

    if cu_seqlens is not None:
        if batch != 1:
            raise ValueError(
                "batch size is expected to be 1 when using cu_seqlens; "
                f"got {batch}"
            )
        bounds = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
        num_segments = len(bounds) - 1
        if initial_state is not None and initial_state.shape[0] != num_segments:
            raise ValueError(
                "the number of initial states must equal the number of "
                f"sequences, i.e. {num_segments} rather than "
                f"{initial_state.shape[0]}"
            )
        out = q.new_zeros(1, t_len, heads, v_dim, dtype=torch.float32)
        finals = []
        for seg in range(num_segments):
            begin, end = bounds[seg], bounds[seg + 1]
            state = (
                initial_state[seg].to(torch.float32).clone()
                if initial_state is not None
                else q.new_zeros(heads, k_dim, v_dim, dtype=torch.float32)
            )
            if end > begin:
                seg_out, state = _segment(
                    q[0, begin:end], k[0, begin:end], v[0, begin:end],
                    g[0, begin:end], beta[0, begin:end], scale, state,
                    use_qk_l2norm_in_kernel,
                )
                out[0, begin:end] = seg_out
            finals.append(state)
        final_state = torch.stack(finals, dim=0) if output_final_state else None
        return out.to(q.dtype), final_state

    out = q.new_zeros(batch, t_len, heads, v_dim, dtype=torch.float32)
    finals = []
    for b in range(batch):
        state = (
            initial_state[b].to(torch.float32).clone()
            if initial_state is not None
            else q.new_zeros(heads, k_dim, v_dim, dtype=torch.float32)
        )
        seg_out, state = _segment(
            q[b], k[b], v[b], g[b], beta[b], scale, state,
            use_qk_l2norm_in_kernel,
        )
        out[b] = seg_out
        finals.append(state)
    final_state = torch.stack(finals, dim=0) if output_final_state else None
    return out.to(q.dtype), final_state


def _scalars(ctx):
    """Unwrap ctx["inputs"] recipe entries to plain values."""
    out = {}
    for name, raw in ctx["inputs"].items():
        out[name] = raw["value"] if isinstance(raw, dict) and "value" in raw else raw
    return out


def gen_inputs(ctx, device):
    """Build the interdependent gate / state tensors the recipe cannot express.

    Constraints, matching the operator's own docstring example:
      * g is a LOG gate produced by F.logsigmoid, hence strictly negative --
        a positive g makes exp(g) > 1 and the state diverges over T steps;
      * beta is a sigmoid gate in (0, 1);
      * k is L2-normalized in the example, which keeps the delta-rule update
        contractive;
      * initial_state is [N, H, K, V] float32, one slab per segment, where N is
        the batch for the packed form and the segment count under cu_seqlens.
    """
    spec = _scalars(ctx)
    batch = int(spec["batch"])
    t_len = int(spec["t"])
    heads = int(spec["h"])
    k_dim = int(spec["k_dim"])
    v_dim = int(spec["v_dim"])
    dtype = getattr(torch, spec["dtype"])
    has_init = bool(spec["has_init"])
    varlen = bool(spec.get("varlen", False))

    gen = torch.Generator(device="cpu").manual_seed(int(ctx["seed"]))

    def rnd(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    q = rnd(batch, t_len, heads, k_dim)
    k = rnd(batch, t_len, heads, k_dim)
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    v = rnd(batch, t_len, heads, v_dim)
    g = torch.nn.functional.logsigmoid(
        torch.rand(batch, t_len, heads, generator=gen, dtype=torch.float32)
    )
    beta = torch.sigmoid(rnd(batch, t_len, heads))

    cu_seqlens = None
    num_states = batch
    if varlen:
        # pack `batch` equal segments into a single row, as the docstring does
        # with rearrange('b t ... -> 1 (b t) ...')
        q = q.reshape(1, batch * t_len, heads, k_dim)
        k = k.reshape(1, batch * t_len, heads, k_dim)
        v = v.reshape(1, batch * t_len, heads, v_dim)
        g = g.reshape(1, batch * t_len, heads)
        beta = beta.reshape(1, batch * t_len, heads)
        cu_seqlens = torch.tensor(
            [i * t_len for i in range(batch + 1)], dtype=torch.int64, device=device
        )
        num_states = batch

    initial_state = (
        rnd(num_states, heads, k_dim, v_dim).to(device) if has_init else None
    )

    return {
        "q": q.to(device=device, dtype=dtype),
        "k": k.to(device=device, dtype=dtype),
        "v": v.to(device=device, dtype=dtype),
        "g": g.to(device),
        "beta": beta.to(device),
        "scale": k_dim ** -0.5,
        "initial_state": initial_state,
        "cu_seqlens": cu_seqlens,
    }
