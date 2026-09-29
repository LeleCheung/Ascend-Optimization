REFERENCE_DEVICE = 'target'

import torch
def run(prefix_output, prefix_lse, suffix_output, suffix_lse):
    p_lse = torch.where(
        prefix_lse == float("inf"), torch.full_like(prefix_lse, float("-inf")), prefix_lse
    ).float()
    s_lse = torch.where(
        suffix_lse == float("inf"), torch.full_like(suffix_lse, float("-inf")), suffix_lse
    ).float()

    max_lse = torch.maximum(p_lse, s_lse)
    p_lse = p_lse - max_lse
    s_lse = s_lse - max_lse
    p_se = torch.exp(p_lse)
    s_se = torch.exp(s_lse)
    out_se = p_se + s_se

    output_lse = (torch.log(out_se) + max_lse).to(prefix_lse.dtype)

    p_scale = (p_se / out_se).unsqueeze(-1)
    s_scale = (s_se / out_se).unsqueeze(-1)
    output = (
        prefix_output.float() * p_scale + suffix_output.float() * s_scale
    ).to(prefix_output.dtype)
    return output, output_lse
def _source_check(actual, expected):
    a_out, a_lse = actual
    e_out, e_lse = expected
    assert_close(a_out, e_out)
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



