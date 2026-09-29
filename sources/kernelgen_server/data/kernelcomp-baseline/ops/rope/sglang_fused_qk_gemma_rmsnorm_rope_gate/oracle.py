REFERENCE_DEVICE = 'target'

import torch
def _norm_rope(x, weight, cos, sin, eps, rotary_dim, out_dtype):
    """x: [T, H, head_dim]; cos/sin: [T, rotary_dim//2]."""
    x32 = x.to(torch.float32)
    var = (x32 * x32).mean(dim=-1, keepdim=True)
    # GemmaRMSNorm: weight is a delta around 1. The normalized value is rounded
    # to the output dtype *before* RoPE, matching the unfused two-kernel path.
    xn = (x32 * torch.rsqrt(var + eps) * (weight.to(torch.float32) + 1.0)).to(out_dtype)
    xn = xn.to(torch.float32)

    half = rotary_dim // 2
    x1, x2 = xn[..., :half], xn[..., half:rotary_dim]
    c, s = cos[:, None, :], sin[:, None, :]
    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s
    return torch.cat([o1, o2, xn[..., rotary_dim:]], dim=-1).to(out_dtype)
def run(
    q_gate,
    k,
    q_weight,
    k_weight,
    cos_sin_cache,
    positions,
    eps,
    num_q_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    has_gate,
):
    t = q_gate.shape[0]
    dt = q_gate.dtype
    half = rotary_dim // 2
    cache = cos_sin_cache.to(torch.float32)[positions.long()]
    cos, sin = cache[:, :half], cache[:, half:rotary_dim]

    if has_gate:
        qg = q_gate.reshape(t, num_q_heads, 2, head_dim)
        q, gate = qg[:, :, 0, :], qg[:, :, 1, :].contiguous()
    else:
        q, gate = q_gate.reshape(t, num_q_heads, head_dim), None

    q_out = _norm_rope(q, q_weight, cos, sin, eps, rotary_dim, dt)
    k_out = _norm_rope(
        k.reshape(t, num_kv_heads, head_dim), k_weight, cos, sin, eps, rotary_dim, dt
    )
    return (
        q_out.reshape(t, num_q_heads * head_dim),
        k_out.reshape(t, num_kv_heads * head_dim),
        gate if has_gate else None,
    )
_EPS = 1e-06
def _build_case(t, nq, nkv, head_dim, rotary_dim, dtype=torch.bfloat16, seed=0, has_gate=True):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)

    def _t(*shape):
        return torch.randn(*shape, dtype=torch.float32, device=_DEVICE, generator=g).to(
            dtype
        )

    q_cols = nq * (2 if has_gate else 1) * head_dim
    max_seq = 4096
    angle = torch.rand(max_seq, rotary_dim // 2, device=_DEVICE, generator=g) * 6.28
    cos_sin_cache = torch.cat([angle.cos(), angle.sin()], dim=-1).to(dtype)
    return dict(
        q_gate=_t(t, q_cols).contiguous(),
        k=_t(t, nkv * head_dim).contiguous(),
        q_weight=(0.5 * _t(head_dim).float()).to(dtype),
        k_weight=(0.5 * _t(head_dim).float()).to(dtype),
        cos_sin_cache=cos_sin_cache,
        positions=torch.randint(
            0, max_seq, (t,), device=_DEVICE, generator=g, dtype=torch.int32
        ),
        eps=_EPS,
        num_q_heads=nq,
        num_kv_heads=nkv,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        has_gate=has_gate,
        check=_check,
    )

def _check(actual, expected):
    for a, e in zip(actual, expected):
        if e is None:
            assert a is None
            continue
        assert_close(a, e)




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
    parameters = {'q_gate', 'k', 'q_weight', 'k_weight', 'cos_sin_cache', 'positions', 'eps', 'num_q_heads', 'num_kv_heads', 'head_dim', 'rotary_dim', 'has_gate'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

