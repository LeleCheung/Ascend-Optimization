import triton
import triton.language as tl

@triton.jit
def arithmetic(A,B,C,F,S,H,D,Q,N:tl.constexpr):
    i=tl.program_id(0)*256+tl.arange(0,256)
    a=tl.load(A+i,i<N,other=1.)
    b=tl.load(B+i,i<N,other=1.)
    c=tl.load(C+i,i<N,other=0.)
    product=a*b
    split=product+c
    mask=tl.full((),0xfffff000,tl.uint32)
    ah=(a.to(tl.uint32,bitcast=True)&mask).to(tl.float32,bitcast=True)
    bh=(b.to(tl.uint32,bitcast=True)&mask).to(tl.float32,bitcast=True)
    al=a-ah
    bl=b-bh
    e1=product-ah*bh
    e2=e1-al*bh
    e3=e2-ah*bl
    product_error=al*bl-e3
    virtual=split-product
    sum_error=(product-(split-virtual))+(c-virtual)
    compensated=split+(sum_error+product_error)
    tl.store(F+i,tl.fma(a,b,c),i<N)
    tl.store(S+i,split,i<N)
    tl.store(H+i,compensated,i<N)
    tl.store(D+i,tl.div_rn(a,b),i<N)
    tl.store(Q+i,a/b,i<N)

