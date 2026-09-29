REFERENCE_DEVICE = 'target'

import torch
def run(gate_up, alpha, beta, limit):
    orig_shape = gate_up.shape
    two_i = orig_shape[-1]
    n = two_i // 2

    x = gate_up.reshape(-1, two_i).to(torch.float32)
    gate, up = x[:, :n], x[:, n:]
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    # The activation stays fp32 all the way into the scale selection.
    activated = gate * torch.sigmoid(alpha * gate) * (up + beta)

    groups = activated.reshape(-1, n // 32, 32)
    amax = groups.abs().amax(dim=-1).clamp(min=1e-30)
    scale_biased = (torch.ceil(torch.log2(amax / 448.0)) + 127.0).clamp(0.0, 254.0)
    descale = torch.exp2(scale_biased - 127.0)
    q = (groups / descale[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

    return (
        q.reshape(*orig_shape[:-1], n),
        scale_biased.to(torch.uint8).reshape(*orig_shape[:-1], n // 32),
    )
def _source_check(actual, expected):
    a_q, a_s = actual
    e_q, e_s = expected
    # UE8M0 exponents and a power-of-two divide: both outputs are exact.
    assert_close(a_s, e_s)
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



