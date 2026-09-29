REFERENCE_DEVICE = 'target'

import torch
def run(x, group_size):
    num_tokens, hidden = x.shape
    num_groups = hidden // group_size
    blocks = x.float().reshape(num_tokens, num_groups, group_size)
    amax = blocks.abs().amax(dim=-1).clamp(min=1e-30)
    exp = torch.ceil(torch.log2(amax / 448.0)).clamp(-127.0, 127.0)
    scale = torch.exp2(exp)
    xq = (blocks / scale[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    sf_bytes = (exp + 127.0).to(torch.uint8).contiguous()
    x_sf = sf_bytes.view(torch.int32).reshape(num_tokens, num_groups // 4)
    return xq.reshape(num_tokens, hidden), x_sf
def _source_check(actual, expected):
    aq, asf = actual
    eq, esf = expected
    torch.testing.assert_close(asf, esf)
    num_groups = asf.shape[1] * 4
    rep = aq.shape[-1] // num_groups
    a_deq = aq.float() * _decode(asf, num_groups).repeat_interleave(rep, dim=-1)
    e_deq = eq.float() * _decode(esf, num_groups).repeat_interleave(rep, dim=-1)
    mismatch = (a_deq - e_deq).abs() > (2e-2 + 2e-2 * e_deq.abs())
    assert mismatch.float().mean() < 1e-2, "too many fp8 rounding-boundary mismatches"

def _decode(x_sf, num_groups):
    sf = x_sf.view(torch.uint8).reshape(x_sf.shape[0], num_groups).float()
    return torch.exp2(sf - 127.0)

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



