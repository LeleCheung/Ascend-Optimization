REFERENCE_DEVICE = 'target'

import torch
_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = 448.0
_FP8_MIN = -448.0
def _exact_exp2_int(exp_tensor):
    """Compute exact 2^n for integer exponents via IEEE 754 bit manipulation.

    torch.exp2 on some GPUs has 1-ULP error for integer inputs; this is bit-exact.
    """
    exp_int = exp_tensor.to(torch.int32)
    bits = (exp_int + 127).clamp(1, 254) << 23
    return bits.view(torch.float32)
def run(x, block_size=128, scale_fmt=None):
    round_scale = scale_fmt is not None
    orig_shape = x.shape
    N = orig_shape[-1]

    x_flat = x.reshape(-1, N).float()
    m = x_flat.shape[0]
    x_g = x_flat.view(m, N // block_size, block_size)

    amax = x_g.abs().amax(dim=-1).clamp(min=1e-4)
    if round_scale:
        scale = _exact_exp2_int(torch.ceil(torch.log2(amax / _FP8_MAX)))
    else:
        scale = amax / _FP8_MAX

    y = (x_g / scale.unsqueeze(-1)).clamp(_FP8_MIN, _FP8_MAX).to(_FP8_DTYPE)
    y = y.view(m, N).view(orig_shape)
    s = scale.view(*orig_shape[:-1], N // block_size)

    return y, s
def _source_check(actual, expected):
    aq, asc = actual
    eq, esc = expected
    assert_close(asc, esc, dtype=torch.float32)
    ratio = aq.shape[-1] // asc.shape[-1]
    a_deq = aq.float() * asc.repeat_interleave(ratio, dim=-1)
    e_deq = eq.float() * esc.repeat_interleave(ratio, dim=-1)
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



