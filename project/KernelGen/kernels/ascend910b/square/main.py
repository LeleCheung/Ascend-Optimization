"""Triton implementation of aten::square (elementwise x*x).

Baseline architecture: one flat 1-D grid, one BLOCK of contiguous elements per
program, masked load / square / masked store.  The host wrapper only allocates
the output and launches the JIT kernel.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _square_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * x, mask=mask)


BLOCK_SIZE = 1024


def run(x):
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _square_kernel[grid](x, out, n, BLOCK=BLOCK_SIZE)
    return out
