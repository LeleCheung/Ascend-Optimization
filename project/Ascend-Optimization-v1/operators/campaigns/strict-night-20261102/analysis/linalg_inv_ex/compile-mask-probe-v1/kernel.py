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
    quotient = a / b
    residual = _compensated_fma(-quotient, b, a)
    correction = residual / b
    corrected = quotient + correction
    ordinary = (b == 0.) | (tl.abs(b) == float('inf')) | (tl.abs(quotient) == float('inf')) | (quotient != quotient)
    return tl.where(ordinary, quotient, corrected)



@triton.jit(do_not_specialize=["STEPS"])
def _masked_update(A, F, V, T, OUT, STEPS, BLOCK: tl.constexpr, MODE: tl.constexpr):
    pid=tl.program_id(0)
    i=tl.arange(0,BLOCK)
    threshold=tl.load(T+pid)
    active=i>=threshold
    column=tl.load(V+pid*BLOCK+i)
    for k in tl.range(0,STEPS):
        lhs=tl.load(A+pid*STEPS*BLOCK+k*BLOCK+i)
        factor=tl.load(F+pid*STEPS+k)
        if MODE==0:
            column=tl.where(active,_compensated_fma(lhs,factor,column),column)
        elif MODE==1:
            column=_compensated_fma(tl.where(active,lhs,0.),
                                    tl.where(active,factor,0.),column)
        else:
            column=tl.where(active,column+lhs*factor,column)
    tl.store(OUT+pid*BLOCK+i,column)


@triton.jit
def _reciprocal_forms(D, T, OLD, VECTOR, BLOCK: tl.constexpr):
    pid=tl.program_id(0)
    i=tl.arange(0,BLOCK)
    denominator=tl.load(D+pid)
    active=i<tl.load(T+pid)
    one=tl.full((BLOCK,),1.,tl.float32)
    old=_corrected_div(one,denominator)
    vector=_corrected_div(one,tl.where(active,denominator,1.))
    tl.store(OLD+pid*BLOCK+i,old)
    tl.store(VECTOR+pid*BLOCK+i,vector)
