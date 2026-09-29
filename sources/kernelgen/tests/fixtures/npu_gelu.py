import torch
import triton
import triton.language as tl


@triton.jit
def _gelu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask)
    coefficient = (2.0 / 3.141592653589793) ** 0.5
    inner = coefficient * (
        values + 0.044715 * values * values * values
    )
    result = 0.5 * values * (1.0 + tl.math.tanh(inner))
    tl.store(output_ptr + offsets, result, mask=mask)


def run(input_0):
    n_elements = input_0.numel()
    output = torch.empty_like(input_0)
    _gelu_kernel[(triton.cdiv(n_elements, 8192),)](
        input_0,
        output,
        n_elements,
        BLOCK_SIZE=8192,
    )
    return output
