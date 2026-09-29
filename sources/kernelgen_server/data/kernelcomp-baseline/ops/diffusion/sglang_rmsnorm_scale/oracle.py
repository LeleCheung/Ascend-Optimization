REFERENCE_DEVICE = 'target'

import torch
def run(x, weight, scale, eps):
    dim = x.shape[-1]
    xf = x.reshape(-1, dim).to(torch.float32)
    normed = (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)).to(x.dtype)
    normed = normed * weight
    # `scale` may have fewer rows than `x`; each scale row covers a contiguous
    # run of x rows (x_rows must be a multiple of scale_rows).
    s = scale.reshape(-1, dim)
    repeat = normed.shape[0] // s.shape[0]
    return (normed * s.repeat_interleave(repeat, dim=0)).reshape(x.shape)
def _source_check(actual, expected):
    # Three chained bf16 roundings (norm, weight, scale): a stray element
    # can sit one ulp off the reference's rounding, ~1.6% relative at bf16.
    assert_close(actual, expected, atol=2e-2, rtol=2e-2)

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



