REFERENCE_DEVICE = 'target'

import torch
def run(topk_ids, num_local_experts, m_max):
    flat = topk_ids.reshape(-1)
    device = flat.device
    masked_m = torch.zeros(num_local_experts, dtype=torch.int32, device=device)
    src2dst = torch.empty(flat.numel(), dtype=torch.int32, device=device)
    counts = [0] * num_local_experts
    flat_cpu = flat.tolist()
    dst = []
    for e in flat_cpu:
        if e < 0:
            dst.append(0)
            continue
        dst.append(e * m_max + counts[e])
        counts[e] += 1
    src2dst.copy_(torch.tensor(dst, dtype=torch.int32, device=device))
    masked_m.copy_(torch.tensor(counts, dtype=torch.int32, device=device))
    return masked_m, src2dst
def _source_check(actual, expected):
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(
        torch.sort(actual[1]).values, torch.sort(expected[1]).values
    )

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



