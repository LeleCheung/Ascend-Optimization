"""将连续叉积组织成小二维3有效列/4存储列tile，用split/join替代大gather。"""
from make_proposals import BATCH, proposal


def main():
    source=(BATCH/'starting-points/cross_out.py').read_text(encoding='utf8')
    begin=source.index('@triton.jit\ndef _cross_flat_kernel')
    end=source.index('\n\n\n# ---',begin)
    kernel='''@triton.jit
def _cross_flat_kernel(a_ptr,b_ptr,o_ptr,n_elem,BLOCK:tl.constexpr,STEP:tl.constexpr):
    row=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    col=tl.arange(0,4)
    offset=row[:,None]*3+col[None,:]
    mask=(row[:,None]<n_elem//3)&(col[None,:]<3)
    a=tl.load(a_ptr+offset,mask=mask,other=0.0)
    b=tl.load(b_ptr+offset,mask=mask,other=0.0)
    ae,ao=tl.split(tl.reshape(a,(BLOCK,2,2)))
    be,bo=tl.split(tl.reshape(b,(BLOCK,2,2)))
    a0,a2=tl.split(ae)
    a1,unused_a=tl.split(ao)
    b0,b2=tl.split(be)
    b1,unused_b=tl.split(bo)
    y0=a1*b2-a2*b1
    y1=a2*b0-a0*b2
    y2=a0*b1-a1*b0
    zero=tl.full((BLOCK,),0,y0.dtype)
    y=tl.reshape(tl.join(tl.join(y0,y2),tl.join(y1,zero)),(BLOCK,4))
    tl.store(o_ptr+offset,y,mask=mask)
'''
    source=source[:begin]+kernel+source[end:]
    source=source.replace('block = _pick_block(n, out.element_size())','block = 256')
    source=source.replace('step = (block // 3) * 3','step = block * 3')
    proposal('cross_out','matrix-split-join-v3',source,
             '连续叉积用256行、4存储列/3有效列二维tile，split/join拆装三分量，避免大tile动态gather',
             '尝试兼顾连续搬运和局部重排；禁止第4列越界读写，真实stride/广播/别名保留原通用路径；不同于已证实更慢的六次独立stride3读写')


if __name__=='__main__':main()
