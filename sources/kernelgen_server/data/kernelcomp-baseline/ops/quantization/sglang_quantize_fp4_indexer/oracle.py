REFERENCE_DEVICE = 'target'

import torch
_GROUP_N = 32
_BLOCK_N = 128
_THRESHOLDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
def _ceil_ue8m0_exp(x):
    """UE8M0 exponent of x, rounded up (mantissa != 0 bumps the exponent)."""
    bits = x.view(torch.int32)
    exp = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    exp = exp + (mantissa != 0).to(torch.int32)
    return exp.clamp(1, 254)
def _fp4_e2m1_code(x):
    ax = x.abs().clamp(max=6.0)
    idx = torch.zeros_like(ax, dtype=torch.int32)
    for t in _THRESHOLDS:
        idx += (ax > t).to(torch.int32)
    sign = ((x < 0) & (idx != 0)).to(torch.int32)
    return idx | (sign << 3)
def run(x):
    x = x.contiguous().view(-1, _BLOCK_N).to(torch.float32)
    n = x.shape[0]
    groups = x.view(n, _BLOCK_N // _GROUP_N, _GROUP_N)

    amax = groups.abs().amax(dim=-1)  # [n, 4]
    sf = torch.clamp(amax / 6.0, min=1.0e-4)
    exp = _ceil_ue8m0_exp(sf)  # [n, 4] int32

    packed_sf = (
        exp[:, 0] | (exp[:, 1] << 8) | (exp[:, 2] << 16) | (exp[:, 3] << 24)
    ).to(torch.int32)

    # Reconstruct the scale as a pure power of two from the stored exponent.
    scale = (exp << 23).view(torch.float32)  # [n, 4]
    v = groups / scale[:, :, None]
    code = _fp4_e2m1_code(v).view(n, _BLOCK_N)

    lo, hi = code[:, 0::2], code[:, 1::2]
    packed = ((lo & 0x0F) | ((hi & 0x0F) << 4)).to(torch.uint8)
    return packed.view(torch.int8), packed_sf
def _source_check(actual, expected):
    a_q, a_sf = actual
    e_q, e_sf = expected
    # Both the packed nibbles and the packed UE8M0 exponents are exact
    # integers: the scale is a power of two, so the ladder comparison is not
    # sensitive to rounding.
    assert_close(a_sf, e_sf)
    assert_close(a_q.view(torch.uint8), e_q.view(torch.uint8))

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



