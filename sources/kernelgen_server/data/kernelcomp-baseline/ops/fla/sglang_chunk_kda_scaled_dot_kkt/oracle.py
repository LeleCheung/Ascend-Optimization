REFERENCE_DEVICE = 'target'

import torch
def run(q, k, gk, beta, scale, gk_scale, chunk_size):
    b, t, h, kdim = k.shape
    bt = chunk_size
    a_out = torch.zeros(b, t, h, bt, device=k.device, dtype=torch.float32)
    aqk_out = torch.zeros(b, t, h, bt, device=k.device, dtype=torch.float32)

    q32, k32 = q.to(torch.float32), k.to(torch.float32)
    g32 = gk.to(torch.float32) * gk_scale
    beta32 = beta.to(torch.float32)

    for lo in range(0, t, bt):
        hi = min(lo + bt, t)
        n = hi - lo
        gc = g32[:, lo:hi]  # [B, n, H, K]
        # exp2(g_i - g_j) per (row i, col j, head, channel)
        decay = torch.exp2(gc[:, :, None] - gc[:, None, :])  # [B, n, n, H, K]

        kk = k32[:, lo:hi]
        qq = q32[:, lo:hi]
        # A[i,j] = beta_i * sum_k k_i k_j exp2(g_i - g_j)
        a = torch.einsum("bihk,bjhk,bijhk->bijh", kk, kk, decay)
        a = a * beta32[:, lo:hi][:, :, None, :]
        aqk = torch.einsum("bihk,bjhk,bijhk->bijh", qq, kk, decay) * scale

        rows = torch.arange(n, device=k.device)
        strict = rows[:, None] > rows[None, :]
        lower = rows[:, None] >= rows[None, :]
        a = torch.where(strict[None, :, :, None], a, torch.zeros_like(a))
        aqk = torch.where(lower[None, :, :, None], aqk, torch.zeros_like(aqk))

        a_out[:, lo:hi, :, :n] = a.permute(0, 1, 3, 2)
        aqk_out[:, lo:hi, :, :n] = aqk.permute(0, 1, 3, 2)

    return a_out, aqk_out
def _build_case(b, t, h, kdim, chunk_size=64, dtype=torch.bfloat16, seed=0, gk_scale=1.0):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)
    q = torch.randn(b, t, h, kdim, dtype=torch.float32, device=_DEVICE, generator=g).to(dtype)
    k = torch.randn(b, t, h, kdim, dtype=torch.float32, device=_DEVICE, generator=g).to(dtype)
    # gk is a within-chunk cumsum of a non-positive gate, so it decreases down
    # the chunk; keep the span small so exp2 of the difference stays bounded.
    gate = -0.05 * torch.rand(
        b, t, h, kdim, dtype=torch.float32, device=_DEVICE, generator=g
    )
    gk = gate.reshape(b, -1, chunk_size, h, kdim).cumsum(dim=2).reshape(b, t, h, kdim)
    beta = torch.rand(b, t, h, dtype=torch.float32, device=_DEVICE, generator=g)
    return dict(
        q=q.contiguous(),
        k=k.contiguous(),
        gk=gk.contiguous(),
        beta=beta.contiguous(),
        scale=kdim**-0.5,
        gk_scale=gk_scale,
        chunk_size=chunk_size,
        check=_check,
    )

def _check(actual, expected):
    for a, e in zip(actual, expected):
        assert_close(a, e, dtype=torch.float32, atol=2e-2, rtol=2e-2)




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
    parameters = {'q', 'k', 'gk', 'beta', 'scale', 'gk_scale', 'chunk_size'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

