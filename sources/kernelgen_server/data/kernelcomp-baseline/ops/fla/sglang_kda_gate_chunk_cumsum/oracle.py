REFERENCE_DEVICE = 'target'

import torch
def run(g, A_log, chunk_size, scale, dt_bias, lower_bound):
    b, t, h, s = g.shape
    x = g.to(torch.float32)
    if dt_bias is not None:
        x = x + dt_bias.reshape(h, s).to(torch.float32)[None, None]

    a = A_log.reshape(h).to(torch.float32)[None, None, :, None]
    if lower_bound is None:
        gate = -torch.exp(a) * torch.nn.functional.softplus(x)
    else:
        # Safe gate variant.
        gate = lower_bound * torch.sigmoid(torch.exp(a) * x)

    # Cumsum along time, restarted at every chunk boundary.
    pad = (-t) % chunk_size
    padded = torch.nn.functional.pad(gate, (0, 0, 0, 0, 0, pad))
    chunks = padded.reshape(b, -1, chunk_size, h, s)
    out = chunks.cumsum(dim=2).reshape(b, -1, h, s)[:, :t]

    if scale is not None:
        out = out * scale
    return out.to(torch.float32)
def _source_check(actual, expected):
    assert_close(actual, expected, dtype=torch.float32, atol=1e-4, rtol=1e-4)

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



