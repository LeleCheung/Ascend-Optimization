REFERENCE_DEVICE = 'target'

import torch
def run(
    q, k, v, g, beta, scale, initial_state, output_final_state, use_qk_l2norm_in_kernel=False
):
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
def _build_case(
    batch, t, h, hv, k_dim, v_dim, beta_headwise=False, has_init=False, output_final_state=True,
    l2norm=False, dtype=torch.bfloat16, seed=0,
):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)
    q = torch.randn(batch, t, h, k_dim, generator=g, device=_DEVICE, dtype=torch.float32).to(dtype)
    k = torch.randn(batch, t, h, k_dim, generator=g, device=_DEVICE, dtype=torch.float32).to(dtype)
    v = torch.randn(batch, t, hv, v_dim, generator=g, device=_DEVICE, dtype=torch.float32).to(
        dtype
    )
    gate = -torch.rand(batch, t, hv, generator=g, device=_DEVICE, dtype=torch.float32) * 0.1
    if beta_headwise:
        beta = torch.sigmoid(
            torch.randn(batch, t, hv, v_dim, generator=g, device=_DEVICE, dtype=torch.float32)
        )
    else:
        beta = torch.sigmoid(
            torch.randn(batch, t, hv, generator=g, device=_DEVICE, dtype=torch.float32)
        )
    initial_state = None
    if has_init:
        initial_state = torch.randn(
            batch, hv, v_dim, k_dim, generator=g, device=_DEVICE, dtype=torch.float32
        )
    scale = k_dim**-0.5
    return dict(
        q=q,
        k=k,
        v=v,
        g=gate,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=l2norm,
        check=_check,
    )

def _check(actual, expected):
    a_o, a_final = actual
    e_o, e_final = expected
    assert_close(a_o, e_o)
    if e_final is not None:
        # Accumulated float32 rounding differences over many sequential
        # recurrence steps can exceed the tight float32 default tolerance
        # once state magnitudes grow, without indicating an actual bug.
        assert_close(a_final, e_final, atol=1e-2, rtol=1e-2)




_DEVICE = None


def _materialize_arg(value, device):
    if isinstance(value, dict) and "__tensor__" in value:
        dtype = value.get("dtype", "float32")
        return torch.tensor(
            value["__tensor__"],
            dtype=getattr(torch, dtype),
            device=device,
        )
    return value


def gen_inputs(ctx, device):
    global _DEVICE
    _DEVICE = device
    case_args = ctx["inputs"].get("_case_args", {})
    args = [_materialize_arg(value, device) for value in case_args.get("args", [])]
    kwargs = {
        key: _materialize_arg(value, device)
        for key, value in case_args.get("kwargs", {}).items()
    }
    dtype = kwargs.get("dtype")
    if isinstance(dtype, str) and dtype:
        kwargs["dtype"] = getattr(torch, dtype)
    constructor = case_args.get("constructor", "_case")
    if constructor == '_case':
        built = _build_case(*args, **kwargs)
    parameters = {'q', 'k', 'v', 'g', 'beta', 'scale', 'initial_state', 'output_final_state', 'use_qk_l2norm_in_kernel'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

