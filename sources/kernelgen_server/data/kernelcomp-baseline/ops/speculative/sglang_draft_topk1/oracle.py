REFERENCE_DEVICE = 'target'

import torch
def run(next_token_logits, positions, draft_tokens=None, draft_token_column=0):
    bs = next_token_logits.shape[0]
    topk_index = next_token_logits.argmax(dim=-1, keepdim=True).to(torch.int64)
    topk_p = torch.ones(bs, 1, dtype=torch.float32, device=next_token_logits.device)

    out_positions = positions + 1
    out_draft_tokens = None
    if draft_tokens is not None:
        out_draft_tokens = draft_tokens.clone()
        out_draft_tokens[:, draft_token_column] = topk_index.squeeze(-1)

    return topk_p, topk_index, out_positions, out_draft_tokens
def _source_check(actual, expected):
    a_p, a_idx, a_pos, a_dt = actual
    e_p, e_idx, e_pos, e_dt = expected
    assert_close(a_p, e_p, dtype=torch.float32)
    assert torch.equal(a_idx, e_idx), "argmax index mismatch"
    assert torch.equal(a_pos, e_pos), "positions mismatch"
    if a_dt is not None:
        assert torch.equal(a_dt, e_dt), "draft_tokens mismatch"

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



