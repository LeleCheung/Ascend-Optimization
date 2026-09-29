REFERENCE_DEVICE = 'target'

import torch
def run(k, v, beta, g_cumsum, A, cu_seqlens):
    assert cu_seqlens is None, "varlen path is out of scope"
    B, T, Hg, K = k.shape
    H, V = v.shape[-2], v.shape[-1]
    BT = A.shape[-1]
    heads_per_kv = H // Hg

    w = k.new_empty(B, T, H, K)
    u = torch.empty_like(v)

    for c in range(T // BT):
        rows = slice(c * BT, (c + 1) * BT)
        # [B, H, BT, BT]
        Ac = A[:, rows].permute(0, 2, 1, 3).float()
        b_beta = beta[:, rows].permute(0, 2, 1).float()          # [B, H, BT]
        b_g = torch.exp(g_cumsum[:, rows].permute(0, 2, 1).float())

        vb = v[:, rows].permute(0, 2, 1, 3).float() * b_beta[..., None]
        u[:, rows] = (Ac @ vb.to(v.dtype).float()).permute(0, 2, 1, 3).to(v.dtype)

        kh = k[:, rows].permute(0, 2, 1, 3).float()              # [B, Hg, BT, K]
        kh = kh.repeat_interleave(heads_per_kv, dim=1)           # [B, H, BT, K]
        kb = kh * b_beta[..., None] * b_g[..., None]
        w[:, rows] = (Ac @ kb.to(k.dtype).float()).permute(0, 2, 1, 3).to(k.dtype)

    return w, u
def _build_case(b, t, h, hg, k_dim=128, v_dim=128, bt=64, seed=0):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)

    def r(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device=_DEVICE, generator=g).to(dtype)

    return dict(
        k=r(b, t, hg, k_dim),
        v=r(b, t, h, v_dim),
        beta=r(b, t, h, dtype=torch.float32).sigmoid(),
        # g_cumsum is a non-positive cumulative log-decay; exp() of it is a gate.
        g_cumsum=-r(b, t, h, dtype=torch.float32).abs(),
        A=r(b, t, h, bt),
        cu_seqlens=None,
        check=_check,
    )

def _check(actual, expected):
    assert_close(actual[0], expected[0])
    assert_close(actual[1], expected[1])




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
    parameters = {'k', 'v', 'beta', 'g_cumsum', 'A', 'cu_seqlens'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

