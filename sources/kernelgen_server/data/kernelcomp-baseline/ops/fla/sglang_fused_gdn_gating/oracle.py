REFERENCE_DEVICE = 'target'

import torch
import torch.nn.functional as F
def run(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
    x = a.float() + dt_bias.float()
    softplus_x = torch.where(beta * x <= threshold, F.softplus(x, beta=beta), x)
    g = -torch.exp(A_log.float()) * softplus_x
    beta_output = torch.sigmoid(b.float())
    return g.unsqueeze(0).to(torch.float32), beta_output.unsqueeze(0).to(torch.float32)
def _source_check(actual, expected):
    a_g, a_beta = actual
    e_g, e_beta = expected
    assert_close(a_g, e_g, dtype=torch.float32)
    assert_close(a_beta, e_beta, dtype=torch.float32)

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



