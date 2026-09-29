REFERENCE_DEVICE = 'target'

import torch
def run(x, gate, residual, weight, eps):
    dim = x.shape[-1]
    xf = x.reshape(-1, dim).to(torch.float32)
    normed = (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)).to(x.dtype)
    normed = normed * weight

    g = gate.reshape(-1, dim)
    repeat = normed.shape[0] // g.shape[0]
    gated = torch.tanh(g.to(torch.float32)).to(x.dtype).repeat_interleave(repeat, dim=0)
    return (residual.reshape(-1, dim) + gated * normed).reshape(x.shape)
def _source_check(actual, expected):
    # `residual + gated * normed` cancels for a handful of elements, where a
    # one-ulp difference upstream flips the sum to (or off) exact zero and
    # the relative error is unbounded. Allow a small mismatching fraction
    # rather than loosening the tolerance for everything.
    a, e = actual.to(torch.float32), expected.to(torch.float32)
    atol, rtol = 2e-2, 2e-2
    mismatch = (a - e).abs() > (atol + rtol * e.abs())
    assert mismatch.float().mean() < 1e-4, "too many mismatching elements"
    assert (a - e).abs().max() < 0.1, "a mismatch is too large to be cancellation"

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



