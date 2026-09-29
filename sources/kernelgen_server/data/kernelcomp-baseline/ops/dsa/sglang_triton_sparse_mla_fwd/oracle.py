REFERENCE_DEVICE = 'target'

import torch
def run(q_nope, q_rope, kv, indices, sm_scale, d_v):
    seq, h, _ = q_nope.shape
    dim = kv.shape[-1]
    topk = indices.shape[-1]

    q = torch.cat([q_nope.to(torch.float32), q_rope.to(torch.float32)], dim=-1)
    idx = indices.reshape(seq, topk)
    valid = idx >= 0
    gathered = kv.reshape(-1, dim).to(torch.float32)[idx.clamp(min=0).to(torch.int64)]

    qk = torch.einsum("shd,skd->shk", q, gathered) * sm_scale
    qk = torch.where(valid[:, None, :], qk, torch.full_like(qk, float("-inf")))

    # Rows with no valid key produce zeros rather than NaN.
    p = torch.softmax(qk, dim=-1)
    p = torch.nan_to_num(p, nan=0.0)

    out = torch.einsum("shk,skd->shd", p, gathered[..., :d_v])
    return out.to(torch.bfloat16).unsqueeze(0)
_DIM = 576
_D_V = 512
def _build_case(seq, h, topk, num_pages, invalid_frac=0.0, seed=0):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)
    fp8 = torch.float8_e4m3fn

    def _q(width):
        return (
            0.5 * torch.randn(seq, h, width, dtype=torch.float32, device=_DEVICE, generator=g)
        ).to(fp8)

    kv = (
        0.5 * torch.randn(num_pages, 1, _DIM, dtype=torch.float32, device=_DEVICE, generator=g)
    ).to(fp8)
    idx = torch.randint(
        0, num_pages, (seq, 1, topk), device=_DEVICE, generator=g, dtype=torch.int32
    )
    if invalid_frac:
        drop = torch.rand(idx.shape, device=_DEVICE, generator=g) < invalid_frac
        idx = torch.where(drop, torch.full_like(idx, -1), idx)
    return dict(
        q_nope=_q(_D_V),
        q_rope=_q(_DIM - _D_V),
        kv=kv,
        indices=idx,
        sm_scale=(_DIM**-0.5),
        d_v=_D_V,
        check=_check,
    )

def _check(actual, expected):
    # The kernel requantizes the softmax probabilities to fp8 (p * 448 -> e4m3)
    # before the P@V dot, so it cannot match an fp32 reference tightly. e4m3
    # carries 3 mantissa bits (~6% half-ulp), and averaging over topk keys
    # damps that to the low percent range on the output.
    assert_close(actual, expected, atol=6e-2, rtol=6e-2)




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
    parameters = {'q_nope', 'q_rope', 'kv', 'indices', 'sm_scale', 'd_v'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

