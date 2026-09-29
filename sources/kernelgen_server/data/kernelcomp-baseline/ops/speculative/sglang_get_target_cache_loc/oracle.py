REFERENCE_DEVICE = 'target'

import torch
def run(num_correct_drafts, to_free_num_slots, out_cache_loc, num_verify_tokens):
    bs = num_correct_drafts.shape[0]
    rows = out_cache_loc.reshape(bs, num_verify_tokens)
    n_tgt = int(num_correct_drafts.sum()) + bs
    n_free = int(to_free_num_slots.sum())
    tgt = torch.zeros(max(n_tgt, 1), dtype=out_cache_loc.dtype, device=rows.device)
    free = torch.zeros(max(n_free, 1), dtype=out_cache_loc.dtype, device=rows.device)

    tgt_pos = 0
    free_pos = 0
    for b in range(bs):
        keep = int(num_correct_drafts[b]) + 1
        tgt[tgt_pos : tgt_pos + keep] = rows[b, :keep]
        tgt_pos += keep
        f = int(to_free_num_slots[b])
        if f:
            free[free_pos : free_pos + f] = rows[b, num_verify_tokens - f :]
            free_pos += f
    return tgt, free
def _source_check(actual, expected):
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])

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



