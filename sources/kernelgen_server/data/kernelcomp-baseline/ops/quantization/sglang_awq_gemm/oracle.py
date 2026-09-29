REFERENCE_DEVICE = 'target'

import torch
_BITS = 4
_AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
def _reverse_awq_order(t: torch.Tensor) -> torch.Tensor:
    idx = torch.arange(t.shape[-1], dtype=torch.int32, device=t.device)
    idx = idx.view(-1, 32 // _BITS)[:, _AWQ_REVERSE_ORDER].view(-1)
    return (t[:, idx] & 0xF).contiguous()
def _dequantize(qweight, scales, zeros):
    group_size = qweight.shape[0] // scales.shape[0]
    shifts = torch.arange(0, 32, _BITS, device=qweight.device)

    iweights = torch.bitwise_right_shift(qweight[:, :, None], shifts[None, None, :]).to(
        torch.int8
    )
    iweights = _reverse_awq_order(iweights.view(iweights.shape[0], -1))

    zbits = torch.bitwise_right_shift(zeros[:, :, None], shifts[None, None, :]).to(
        torch.int8
    )
    zbits = _reverse_awq_order(zbits.view(zeros.shape[0], -1))

    iweights = torch.bitwise_and(iweights, (2**_BITS) - 1)
    zbits = torch.bitwise_and(zbits, (2**_BITS) - 1)

    scales_rep = scales.repeat_interleave(group_size, dim=0)
    zeros_rep = zbits.repeat_interleave(group_size, dim=0)
    return (iweights - zeros_rep) * scales_rep
def run(input, qweight, scales, qzeros, split_k_iters):
    w_deq = _dequantize(qweight, scales, qzeros)
    return torch.matmul(input, w_deq)
def _source_check(actual, expected):
    assert_close(actual.cpu(), expected.cpu(), atol=1e-1, rtol=1e-1)

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



