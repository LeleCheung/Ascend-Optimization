"""Triton implementation of ``smooth_l1_loss_backward`` for Ascend 910B.

Elementwise backward of the smooth-L1 (Huber with ``beta``) loss::

    s   = 1 / numel(input)   when reduction == 'mean' else 1
    out[i] = (grad_output[i] * s) * phi(input[i] - target[i])
    phi(d) = d / beta   if |d| < beta
           = sign(d)    otherwise

Semantics mirror the reference exactly: the reduction scale is applied to the
gradient operand before the elementwise product (not after), ``grad_output`` is
a scalar operand for the reduced modes, and the output has the shape/dtype of
``input``.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 4096
_REDUCTION_STR = {"none": 0, "mean": 1, "sum": 2}
_MEAN = 1


@triton.jit
def _smooth_l1_loss_backward_kernel(
    grad_output_ptr,
    input_ptr,
    target_ptr,
    out_ptr,
    n_elements,
    inv_n,
    INV_BETA: tl.constexpr,
    BETA_ZERO: tl.constexpr,
    REDUCTION: tl.constexpr,
    GRAD_SCALAR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    t = tl.load(target_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    diff = x - t

    if BETA_ZERO:
        # reference behaviour for beta == 0: explicit +1/-1 with exactly 0.0 at
        # the kink (d == 0 must store 0.0, never NaN)
        grad = tl.where(diff > 0.0, 1.0, tl.where(diff < 0.0, -1.0, 0.0))
    else:
        scaled = diff * INV_BETA
        grad = tl.where(scaled < -1.0, -1.0, tl.where(scaled > 1.0, 1.0, scaled))

    if GRAD_SCALAR:
        grad_out = tl.load(grad_output_ptr).to(tl.float32)
    else:
        grad_out = tl.load(grad_output_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    if REDUCTION == 1:
        grad_out = grad_out * inv_n

    tl.store(out_ptr + offs, grad * grad_out, mask=mask)


def run(grad_output, input, target, reduction, beta):
    if isinstance(reduction, str):
        red = _REDUCTION_STR.get(reduction, -1)
        if red < 0:
            raise ValueError("reduction must be one of 'none', 'mean', or 'sum'")
    elif isinstance(reduction, int) and 0 <= reduction <= 2:
        red = reduction
    else:
        raise ValueError("reduction must be one of 'none', 'mean', or 'sum'")

    beta = float(beta)
    if beta < 0:
        raise RuntimeError("smooth_l1_loss does not support negative values for beta.")

    # reduction_elements is captured before any broadcast, as in the reference
    reduction_elements = input.numel()
    n_elements = reduction_elements
    if input.shape != target.shape:
        input, target = torch.broadcast_tensors(input, target)
        n_elements = input.numel()
    if not (input.is_contiguous() and target.is_contiguous()):
        input = input.contiguous()
        target = target.contiguous()

    grad_scalar = grad_output.numel() == 1
    if not grad_scalar:
        if grad_output.shape != input.shape:
            grad_output = torch.broadcast_to(grad_output, input.shape)
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

    out = torch.empty_like(input)
    if n_elements == 0:
        return out

    inv_n = (1.0 / reduction_elements) if red == _MEAN else 1.0
    grid = ((n_elements + _BLOCK - 1) // _BLOCK,)
    _smooth_l1_loss_backward_kernel[grid](
        grad_output,
        input,
        target,
        out,
        n_elements,
        inv_n,
        INV_BETA=(0.0 if beta == 0.0 else 1.0 / beta),
        BETA_ZERO=(beta == 0.0),
        REDUCTION=red,
        GRAD_SCALAR=grad_scalar,
        BLOCK_SIZE=_BLOCK,
    )
    return out
