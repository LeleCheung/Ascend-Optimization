REFERENCE_DEVICE = 'target'

import torch
def run(recv_output, recv_lse, is_lse_base_on_e, return_lse):
    lse = recv_lse.to(torch.float32)
    # NaN and +inf shards are treated as "no contribution".
    lse = torch.where(torch.isnan(lse) | (lse == float("inf")), float("-inf"), lse)

    lse_max = lse.amax(dim=0)
    lse_max = torch.where(lse_max == float("-inf"), torch.zeros_like(lse_max), lse_max)

    centered = lse - lse_max[None]
    w = torch.exp(centered) if is_lse_base_on_e else torch.exp2(centered)
    weight_sum = w.sum(dim=0)

    acc = (recv_output.to(torch.float32) * w[..., None]).sum(dim=0) / weight_sum[..., None]
    out = acc.to(recv_output.dtype)

    if not return_lse:
        return out, None
    log = torch.log if is_lse_base_on_e else torch.log2
    return out, (log(weight_sum) + lse_max).to(recv_lse.dtype)
def _source_check(actual, expected):
    a_out, a_lse = actual
    e_out, e_lse = expected
    assert_close(a_out, e_out)
    if e_lse is None:
        assert a_lse is None
    else:
        assert_close(a_lse, e_lse, dtype=torch.float32)

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



