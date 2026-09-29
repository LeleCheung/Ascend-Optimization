REFERENCE_DEVICE = 'target'

import torch
def run(topk_ids, topk_weights, num_fused_shared_experts, scale_factor, N):
    m, k = topk_ids.shape
    s = int(num_fused_shared_experts)
    if s <= 0:
        return topk_ids, topk_weights

    shared_ids = (
        torch.arange(N, N + s, device=topk_ids.device, dtype=topk_ids.dtype)
        .reshape(1, s)
        .expand(m, s)
    )
    shared_w = torch.full(
        (m, s), scale_factor, dtype=topk_weights.dtype, device=topk_weights.device
    )
    return (
        torch.cat([topk_ids, shared_ids], dim=1).contiguous(),
        torch.cat([topk_weights, shared_w], dim=1).contiguous(),
    )
def _source_check(actual, expected):
    for a, e in zip(actual, expected):
        assert_close(a, e)

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



