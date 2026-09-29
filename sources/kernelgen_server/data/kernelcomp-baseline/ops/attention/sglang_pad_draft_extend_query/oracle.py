REFERENCE_DEVICE = 'target'

import torch
def run(q, padded_q, seq_lens_q, cu_seqlens_q):
    out = padded_q.clone()
    bs = cu_seqlens_q.shape[0] - 1
    for b in range(bs):
        s = int(seq_lens_q[b])
        beg = int(cu_seqlens_q[b])
        out[b, :s] = q[beg : beg + s]
    return out
def _build_case(bs, max_seq_len=8, num_heads=16, head_dim=128, seed=0):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)
    seq_lens = torch.randint(
        1, max_seq_len + 1, (bs,), dtype=torch.int32, device=_DEVICE, generator=g
    )
    cu = torch.zeros(bs + 1, dtype=torch.int32, device=_DEVICE)
    cu[1:] = torch.cumsum(seq_lens, dim=0)
    total = int(cu[-1])
    q = torch.randn(
        total, num_heads, head_dim, device=_DEVICE, generator=g
    ).to(torch.bfloat16)
    padded_q = torch.zeros(
        bs, max_seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=_DEVICE
    )
    return dict(
        q=q,
        padded_q=padded_q,
        seq_lens_q=seq_lens,
        cu_seqlens_q=cu,
        check=assert_close,
    )

def assert_close(actual, expected, *, dtype=None, **overrides):
    tol = tolerance_for(dtype if dtype is not None else expected.dtype)
    tol.update(overrides)
    torch.testing.assert_close(
        actual.to(torch.float32) if actual.dtype.is_floating_point else actual,
        expected.to(torch.float32) if expected.dtype.is_floating_point else expected,
        **tol,
    )

_DEFAULT_TOLERANCE = {"atol": 0.01, "rtol": 0.01}
_TOLERANCES = {"float32": {"atol": 0.0001, "rtol": 0.0001}, "bfloat16": {"atol": 0.015, "rtol": 0.015}, "float16": {"atol": 0.01, "rtol": 0.01}}
def tolerance_for(dtype: torch.dtype) -> dict:
    return dict(_TOLERANCES.get(dtype, _DEFAULT_TOLERANCE))




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
    parameters = {'q', 'padded_q', 'seq_lens_q', 'cu_seqlens_q'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

