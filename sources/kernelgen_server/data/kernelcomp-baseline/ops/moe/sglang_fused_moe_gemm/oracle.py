REFERENCE_DEVICE = 'target'

import torch
def run(A, B, topk_weights, topk_ids, top_k):
    T, K = A.shape
    E, N, _ = B.shape
    A32 = A.float()
    B32 = B.float()

    out = torch.empty(T, top_k, N, dtype=A.dtype, device=A.device)
    for t in range(T):
        for j in range(top_k):
            e = int(topk_ids[t, j].item())
            row = A32[t] @ B32[e].t()
            out[t, j] = (row * topk_weights[t, j].float()).to(A.dtype)
    return out
def _source_check(actual, expected):
    assert_close(actual, expected, atol=0.5, rtol=1e-2)

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



