REFERENCE_DEVICE = 'target'

import torch
def run(qkv, eps):
    # qkv: (B, N, 3, H, D). The RMS is taken over the whole (H, D) block of a
    # token, not per head.
    x = qkv.to(torch.float32)
    q, k = x[:, :, 0], x[:, :, 1]
    q_inv = torch.rsqrt(q.pow(2).mean(dim=(-2, -1)) + eps)
    k_inv = torch.rsqrt(k.pow(2).mean(dim=(-2, -1)) + eps)
    return q_inv.to(torch.float32), k_inv.to(torch.float32)
def _source_check(actual, expected):
    for a, e in zip(actual, expected):
        assert_close(a, e, dtype=torch.float32, atol=1e-3, rtol=1e-3)

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



