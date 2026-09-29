REFERENCE_DEVICE = 'target'

import torch
def run(extend_seq_lens, seq_lens, total_len, max_q_len):
    device = extend_seq_lens.device
    out = torch.empty(total_len, dtype=torch.int32, device=device)
    pos = 0
    for i in range(extend_seq_lens.numel()):
        qo = int(extend_seq_lens[i])
        kv = int(seq_lens[i])
        vals = torch.arange(qo, dtype=torch.int32, device=device) + (kv - qo + 1)
        out[pos : pos + qo] = vals.clamp(min=0)
        pos += qo
    return out
def _build_case(n, max_q=8, seed=0, include_short_kv=True):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)
    qo = torch.randint(1, max_q + 1, (n,), dtype=torch.int32, device=_DEVICE, generator=g)
    kv = torch.randint(
        1, 8192, (n,), dtype=torch.int32, device=_DEVICE, generator=g
    )
    if include_short_kv:
        kv[:: max(n // 4, 1)] = 0
    return dict(
        extend_seq_lens=qo,
        seq_lens=kv,
        total_len=int(qo.sum()),
        max_q_len=int(qo.max()),
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
    parameters = {'extend_seq_lens', 'seq_lens', 'total_len', 'max_q_len'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

