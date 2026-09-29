REFERENCE_DEVICE = 'target'

import torch
def _rmsnorm32(v32, w32, eps):
    rms = torch.sqrt((v32 * v32).mean(dim=-1, keepdim=True) + eps)
    return v32 / rms * w32
def run(x, residual, weight1, weight2, eps):
    mid = residual + _rmsnorm32(x.float(), weight1.float(), eps).to(residual.dtype)
    out = _rmsnorm32(mid.float(), weight2.float(), eps).to(x.dtype)
    return out, mid
def _source_check(actual, expected):
    a_out, a_mid = actual
    e_out, e_mid = expected
    assert_close(a_out, e_out)
    assert_close(a_mid, e_mid)

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



