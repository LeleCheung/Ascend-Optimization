REFERENCE_DEVICE = 'target'

import torch
_FP8_DTYPE = torch.float8_e4m3fn
_EPS = 1e-10
def run(x, group_size):
    fp8_max = torch.finfo(_FP8_DTYPE).max
    fp8_min = -fp8_max

    orig_shape = x.shape
    x32 = x.reshape(-1, group_size).to(torch.float32)
    amax = x32.abs().amax(dim=-1, keepdim=True).clamp(min=_EPS)
    scale = amax / fp8_max
    x_q = (x32 / scale).clamp(fp8_min, fp8_max).to(_FP8_DTYPE)

    x_q = x_q.reshape(orig_shape)
    x_s = scale.reshape(orig_shape[:-1] + (orig_shape[-1] // group_size,))
    return x_q, x_s
def _source_check(actual, expected):
    aq, asc = actual
    eq, esc = expected
    assert_close(asc, esc, dtype=torch.float32)
    # Compare dequantized values, not raw fp8 codewords: a sub-ULP scale
    # difference shifts many elements to an adjacent codeword even though
    # the reconstructed value is essentially unchanged.
    a_deq = aq.to(torch.float32) * asc.repeat_interleave(
        aq.shape[-1] // asc.shape[-1], dim=-1
    )
    e_deq = eq.to(torch.float32) * esc.repeat_interleave(
        eq.shape[-1] // esc.shape[-1], dim=-1
    )
    # fp8 is coarse: elements exactly at a quantization bin boundary can land
    # on the adjacent codeword from sub-ULP fp32-reduction-order differences
    # between this reference and the kernel. Allow a small mismatching
    # fraction rather than requiring every element within tolerance
    # (matches SGLang's own per_token_group_quant_fp8 test convention).
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



