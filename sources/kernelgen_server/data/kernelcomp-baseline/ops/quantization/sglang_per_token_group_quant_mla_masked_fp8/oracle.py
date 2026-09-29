REFERENCE_DEVICE = 'target'

import torch
_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = float(torch.finfo(_FP8_DTYPE).max)
def run(x, group_size, eps):
    b, m, k = x.shape
    aligned_m = (m + 255) // 256 * 256  # 256 is the gemm kernel's max block_m
    num_tiles_k = k // group_size

    x32 = x.to(torch.float32).reshape(b, m, num_tiles_k, group_size)
    absmax = x32.abs().amax(dim=-1).clamp(min=eps)  # [b, m, num_tiles_k]
    scale = absmax / _FP8_MAX
    q = (x32 / scale[..., None]).clamp(-_FP8_MAX, _FP8_MAX)

    # Rows [m, aligned_m) are padding the kernel never writes; zero them here
    # so the reference is well defined, and let the check compare only [:m].
    x_q = torch.zeros((b, aligned_m, k), dtype=_FP8_DTYPE, device=x.device)
    x_q[:, :m] = q.reshape(b, m, k).to(_FP8_DTYPE)
    x_s = torch.zeros((b, aligned_m, num_tiles_k), dtype=torch.float32, device=x.device)
    x_s[:, :m] = scale
    masked_m = torch.full((b,), m, dtype=torch.int32, device=x.device)

    return x_q, x_s, masked_m, m, aligned_m
def _source_check(actual, expected):
    a_q, a_s, a_masked, a_m, a_am = actual
    e_q, e_s, e_masked, e_m, e_am = expected
    assert a_m == e_m and a_am == e_am, "m / aligned_m mismatch"
    assert_close(a_masked, e_masked)
    m = e_m
    # Only [:m] is defined: the baseline allocates x_q/x_s with `new_empty`
    # and the kernel's grid is (m, b), so the alignment padding is never
    # written and holds whatever was in the allocator's block.
    assert_close(a_s[:, :m], e_s[:, :m], dtype=torch.float32)
    a_deq = a_q[:, :m].to(torch.float32)
    e_deq = e_q[:, :m].to(torch.float32)
    atol, rtol = 2e-2, 2e-2
    mismatch = (a_deq - e_deq).abs() > (atol + rtol * e_deq.abs())
    assert mismatch.float().mean() < 1e-2, "too many fp8 rounding-boundary mismatches"

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



