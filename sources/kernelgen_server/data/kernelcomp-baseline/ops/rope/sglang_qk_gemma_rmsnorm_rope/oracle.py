REFERENCE_DEVICE = 'target'

import torch
def _norm_rope(x, weight, cos_sin_cache, positions, eps, head_dim, rotary_dim, neox):
    m, cols = x.shape
    heads = cols // head_dim
    x32 = x.to(torch.float32).reshape(m, heads, head_dim)
    var = (x32 * x32).mean(dim=-1, keepdim=True)
    normed = x32 * torch.rsqrt(var + eps) * (1.0 + weight.to(torch.float32))
    # Match the unfused path: GemmaRMSNorm writes bf16/fp16 and RoPE reads that
    # rounded value back.
    normed = normed.to(x.dtype).to(torch.float32)

    half = rotary_dim // 2
    cache = cos_sin_cache.to(torch.float32)[positions.long()]  # [M, rotary_dim]
    cos_half, sin_half = cache[:, :half], cache[:, half:rotary_dim]

    rot = normed[..., :rotary_dim]
    if neox:
        x1, x2 = rot[..., :half], rot[..., half:]
        c, s = cos_half[:, None, :], sin_half[:, None, :]
        rotated = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)
    else:
        x1, x2 = rot[..., 0::2], rot[..., 1::2]
        c, s = cos_half[:, None, :], sin_half[:, None, :]
        o1, o2 = x1 * c - x2 * s, x2 * c + x1 * s
        rotated = torch.stack([o1, o2], dim=-1).flatten(-2)

    out = torch.cat([rotated, normed[..., rotary_dim:]], dim=-1)
    return out.to(x.dtype).reshape(m, cols)
def run(
    q, k, q_weight, k_weight, positions, cos_sin_cache, eps, head_dim, rotary_dim, is_neox_style
):
    q_out = _norm_rope(
        q, q_weight, cos_sin_cache, positions, eps, head_dim, rotary_dim, is_neox_style
    )
    k_out = _norm_rope(
        k, k_weight, cos_sin_cache, positions, eps, head_dim, rotary_dim, is_neox_style
    )
    return q_out, k_out
_EPS = 1e-06
def _build_case(m, nq, nkv, head_dim, rotary_dim, neox=True, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)

    def _t(*shape):
        return torch.randn(*shape, dtype=torch.float32, device=_DEVICE, generator=g).to(
            dtype
        )

    max_seq = 4096
    angle = torch.rand(max_seq, rotary_dim // 2, device=_DEVICE, generator=g) * 6.28
    cos_sin_cache = torch.cat([angle.cos(), angle.sin()], dim=-1).to(dtype)
    return dict(
        q=_t(m, nq * head_dim).contiguous(),
        k=_t(m, nkv * head_dim).contiguous(),
        q_weight=(0.5 * _t(head_dim).float()).to(dtype),
        k_weight=(0.5 * _t(head_dim).float()).to(dtype),
        positions=torch.randint(
            0, max_seq, (m,), device=_DEVICE, generator=g, dtype=torch.int32
        ),
        cos_sin_cache=cos_sin_cache,
        eps=_EPS,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        is_neox_style=neox,
        check=_check,
    )

def _check(actual, expected):
    for a, e in zip(actual, expected):
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
    parameters = {'q', 'k', 'q_weight', 'k_weight', 'positions', 'cos_sin_cache', 'eps', 'head_dim', 'rotary_dim', 'is_neox_style'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

