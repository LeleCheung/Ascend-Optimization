REFERENCE_DEVICE = 'target'

import torch
_BLOCK = 32
def run(x):
    orig = x.shape
    x2d = x.contiguous().reshape(-1, orig[-1]).float()
    blocks = x2d.reshape(x2d.shape[0], -1, _BLOCK)
    amax = blocks.abs().amax(dim=-1).clamp(min=1e-30)
    scale_biased = (torch.ceil(torch.log2(amax / 448.0)) + 127.0).clamp(0.0, 254.0)
    descale = torch.exp2(scale_biased - 127.0)
    xq = (blocks / descale[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return (
        xq.reshape(orig),
        scale_biased.to(torch.uint8).reshape(*orig[:-1], orig[-1] // _BLOCK),
    )
def _source_check(actual, expected):
    aq, asc = actual
    eq, esc = expected
    torch.testing.assert_close(asc, esc)
    rep = aq.shape[-1] // asc.shape[-1]
    a_deq = aq.float() * torch.exp2(asc.float() - 127.0).repeat_interleave(rep, dim=-1)
    e_deq = eq.float() * torch.exp2(esc.float() - 127.0).repeat_interleave(rep, dim=-1)
    mismatch = (a_deq - e_deq).abs() > (2e-2 + 2e-2 * e_deq.abs())
    assert mismatch.float().mean() < 1e-2, "too many mxfp8 rounding mismatches"

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



