import triton
import triton.language as tl

@triton.jit
def _compensated_fma(a, b, c):
    product = a * b
    split = product + c
    mask = tl.full((), 0xfffff000, tl.uint32)
    ah = (a.to(tl.uint32, bitcast=True) & mask).to(tl.float32, bitcast=True)
    bh = (b.to(tl.uint32, bitcast=True) & mask).to(tl.float32, bitcast=True)
    al = a - ah
    bl = b - bh
    e1 = product - ah * bh
    e2 = e1 - al * bh
    e3 = e2 - ah * bl
    product_error = al * bl - e3
    virtual = split - product
    sum_error = (product - (split - virtual)) + (c - virtual)
    return split + (sum_error + product_error)


@triton.jit
def _corrected_div(a, b):
    quotient = tl.div_rn(a, b)
    residual = _compensated_fma(-quotient, b, a)
    correction = tl.div_rn(residual, b)
    corrected = quotient + correction
    ordinary = (b == 0.) | (tl.abs(quotient) == float('inf')) | (quotient != quotient)
    return tl.where(ordinary, quotient, corrected)


@triton.jit
def arithmetic(A, B, C, F, S, H, D, Q, R, RC, N: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    a = tl.load(A + i, i < N, other=1.)
    b = tl.load(B + i, i < N, other=1.)
    c = tl.load(C + i, i < N, other=0.)
    tl.store(F + i, tl.fma(a, b, c), i < N)
    tl.store(S + i, a * b + c, i < N)
    tl.store(H + i, _compensated_fma(a, b, c), i < N)
    tl.store(D + i, tl.div_rn(a, b), i < N)
    tl.store(Q + i, _corrected_div(a, b), i < N)
    tl.store(R + i, tl.div_rn(1., b), i < N)
    tl.store(RC + i, _corrected_div(1., b), i < N)
