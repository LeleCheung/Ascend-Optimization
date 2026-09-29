REFERENCE_DEVICE = 'target'

import torch
def run(gating_output, per_expert_scale, topk):
    logits = gating_output.to(torch.float32)
    # Descending by logit, ties broken toward the lower expert id: a stable
    # sort keeps the original (ascending) id order among equal logits.
    order = torch.argsort(-logits, dim=-1, stable=True)
    top_ids = order[:, :topk]
    top_logits = torch.gather(logits, 1, top_ids)

    # Softmax over the kept logits only, then the per-expert scale.
    weights = torch.softmax(top_logits, dim=-1)
    weights = weights * per_expert_scale.to(torch.float32)[top_ids]
    return weights.to(torch.float32), top_ids.to(torch.int32)
def _source_check(actual, expected):
    a_w, a_i = actual
    e_w, e_i = expected
    assert_close(a_i, e_i)
    assert_close(a_w, e_w, dtype=torch.float32)

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



