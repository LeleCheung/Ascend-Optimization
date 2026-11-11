import torch
import triton
import triton.language as tl


@triton.jit
def _probe_kernel(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    v = tl.load(x_ptr + offs, mask=m, other=1.0)
    tl.store(x_ptr + offs, v * 0.5, mask=m)


def run(A):
    x = A
    n = x.numel()
    _probe_kernel[(triton.cdiv(n, 1024),)](x, n, BLOCK=1024)
    return x
