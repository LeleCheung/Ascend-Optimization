REFERENCE_DEVICE = 'target'

import torch
_EPS = 1e-10
def _group_reference(x, group_size, dtype=torch.int8):
    iinfo = torch.iinfo(dtype)
    int8_min, int8_max = iinfo.min, iinfo.max

    x_ = x.reshape(x.numel() // group_size, group_size)
    amax = x_.abs().max(dim=-1, keepdim=True)[0].clamp(min=_EPS).to(torch.float32)
    x_s = amax / int8_max
    x_q = (x_ / x_s).clamp(min=int8_min, max=int8_max).to(dtype)
    x_q = x_q.reshape(x.shape)
    x_s = x_s.reshape(x.shape[:-1] + (x.shape[-1] // group_size,))
    return x_q, x_s
def run(x):
    return _group_reference(x, group_size=x.shape[-1])
def _source_check(actual, expected):
    aq, asc = actual
    eq, esc = expected
    assert_close(asc, esc, dtype=torch.float32)
    a_deq = aq.to(torch.float32) * asc.repeat_interleave(
        aq.shape[-1] // asc.shape[-1], dim=-1
    )
    e_deq = eq.to(torch.float32) * esc.repeat_interleave(
        eq.shape[-1] // esc.shape[-1], dim=-1
    )
    atol, rtol = 2e-2, 2e-2
    mismatch = (a_deq - e_deq).abs() > (atol + rtol * e_deq.abs())
    assert mismatch.float().mean() < 1e-2, "too many int8 rounding-boundary mismatches"

def assert_close(actual, expected, *, dtype=None, **overrides):
    tol = tolerance_for(dtype if dtype is not None else expected.dtype)
    tol.update(overrides)
    torch.testing.assert_close(
        actual.to(torch.float32) if actual.dtype.is_floating_point else actual,
        expected.to(torch.float32) if expected.dtype.is_floating_point else expected,
        **tol,
    )
def tolerance_for(dtype: torch.dtype) -> dict:
    return dict(_TOLERANCES.get(dtype, _DEFAULT_TOLERANCE))
_TOLERANCES = {
    torch.float32: dict(atol=1e-4, rtol=1e-4),
    torch.bfloat16: dict(atol=1.5e-2, rtol=1.5e-2),
    torch.float16: dict(atol=1e-2, rtol=1e-2),
}
_DEFAULT_TOLERANCE = dict(atol=1e-2, rtol=1e-2)



def valid(ref_outputs, sol_outputs, inputs, ctx):
    try:
        if len(ref_outputs) > 1:
            _source_check(tuple(ref_outputs), tuple(sol_outputs))
        else:
            _source_check(ref_outputs[0], sol_outputs[0])
    except AssertionError as exc:
        return {"passed": False, "message": str(exc), "metrics": {}}
    return {"passed": True, "message": "", "metrics": {}}



