def mulhi_const_u32(x,c):
    mask=0xffffffff
    x0=x&0xffff;x1=x>>16;c0=c&0xffff;c1=c>>16
    w0=(x0*c0)&mask
    t=(x1*c0+(w0>>16))&mask
    w1=(x0*c1+(t&0xffff))&mask
    return (x1*c1+(t>>16)+(w1>>16))&mask
