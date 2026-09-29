import torch
import triton
import triton.language as tl


@triton.jit
def _gelu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    result = 0.5 * values * (1.0 + tl.erf(values * 0.7071067811865476))
    tl.store(output_ptr + offsets, result, mask=mask)


def run(input_0, output):
    n_elements = input_0.numel()
    _gelu_kernel[(triton.cdiv(n_elements, 256),)](
        input_0,
        output,
        n_elements,
        BLOCK_SIZE=256,
    )
    return output
