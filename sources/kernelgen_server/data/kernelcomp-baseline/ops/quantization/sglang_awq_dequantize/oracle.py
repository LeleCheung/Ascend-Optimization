REFERENCE_DEVICE = 'target'

import torch
_BITS = 4
_AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
def _reverse_awq_order(t: torch.Tensor) -> torch.Tensor:
    idx = torch.arange(t.shape[-1], dtype=torch.int32, device=t.device)
    idx = idx.view(-1, 32 // _BITS)[:, _AWQ_REVERSE_ORDER].view(-1)
    return (t[:, idx] & 0xF).contiguous()
def run(qweight, scales, zeros):
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
