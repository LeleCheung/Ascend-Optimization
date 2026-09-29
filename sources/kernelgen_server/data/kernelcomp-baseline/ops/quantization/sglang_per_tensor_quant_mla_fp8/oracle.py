REFERENCE_DEVICE = 'target'

import torch
_FP8_DTYPE = torch.float8_e4m3fn
def run(x, eps=1e-12):
    fp8_max = torch.finfo(_FP8_DTYPE).max
    fp8_min = -fp8_max
    absmax = x.abs().float().max().clamp(min=eps)
    scale = (absmax / fp8_max).reshape(1)
    x_q = (x.to(torch.float32) / scale).clamp(fp8_min, fp8_max).to(_FP8_DTYPE)
    return x_q, scale
def _source_check(actual, expected):
    aq, asc = actual
    eq, esc = expected
    assert_close(asc, esc, dtype=torch.float32)
    a_deq = aq.to(torch.float32) * asc
    e_deq = eq.to(torch.float32) * esc
    atol, rtol = 2e-2, 2e-2
    mismatch = (a_deq - e_deq).abs() > (atol + rtol * e_deq.abs())
    assert mismatch.float().mean() < 1e-2, "too many fp8 rounding-boundary mismatches"

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



