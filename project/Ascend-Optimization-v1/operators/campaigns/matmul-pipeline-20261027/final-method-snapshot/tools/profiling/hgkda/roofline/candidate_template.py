"""单输出大块代替双输出小块；父路径保留其他形状与精度。"""
TEMPLATE=r'''

@triton.jit
def hg_roof_single_tile(A,B,Bias,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                        BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pm=tl.program_id(0);pn=tl.program_id(1)
    mi=pm*BM+tl.arange(0,BM);ni=pn*BN+tl.arange(0,BN);ki=tl.arange(0,BK)
    ap=A+mi[:,None]*K+ki[None,:];bp=B+ki[:,None]*N+ni[None,:]
    acc=tl.zeros((BM,BN),tl.float32)
    for start in range(K//BK):
        av=tl.load(ap);bv=tl.load(bp)
        acc=tl.dot(av,bv,acc,allow_tf32=False)
        ap+=BK;bp+=BK*N
    bias=tl.load(Bias+ni)
    value=acc+bias[None,:]
    value=tl.maximum(value,value*0.0)
    tl.store(C+mi[:,None]*N+ni[None,:],value)

_roof_parent=run
def run(input,weight,bias):
    m,k=input.shape;n=weight.shape[1]
    bm,bn,bk=__BM__,__BN__,__BK__
    if not (input.dtype in (torch.float16,torch.bfloat16) and min(m,n,k)>=512
            and m%bm==0 and n%bn==0 and k%bk==0 and k>=2*bk
            and input.is_contiguous() and weight.is_contiguous() and bias.is_contiguous() and bias.numel()==n):
        return _roof_parent(input,weight,bias)
    out=torch.empty((m,n),dtype=input.dtype,device=input.device)
    with torch_device_fn.device(input.device):
        hg_roof_single_tile[(m//bm,n//bn)](input,weight,bias.reshape(-1),out,m,n,k,bm,bn,bk,
            multibuffer=__MULTI__,unit_flag=__MULTI__,sync_solver=False)
    return out
matmul_bias_activation=run
'''

CONFIGS=[dict(id='single128x256-k128',bm=128,bn=256,bk=128,multi=True),
         dict(id='single128x256-k256',bm=128,bn=256,bk=256,multi=True),
         dict(id='single256x128-k128',bm=256,bn=128,bk=128,multi=True),
         dict(id='single128x128-k256',bm=128,bn=128,bk=256,multi=True),
         dict(id='single128x256-no-multi',bm=128,bn=256,bk=256,multi=False),
         dict(id='single128x128-k128',bm=128,bn=128,bk=128,multi=True),
         dict(id='single128x128-no-multi',bm=128,bn=128,bk=256,multi=False),
         dict(id='single64x128-k256',bm=64,bn=128,bk=256,multi=True),
         dict(id='single128x64-k256',bm=128,bn=64,bk=256,multi=True),
         dict(id='single64x256-k256',bm=64,bn=256,bk=256,multi=True)]

def generate(parent,config):
    text=parent+'\n'+TEMPLATE
    for key in ['bm','bn','bk','multi']:text=text.replace('__'+key.upper()+'__',repr(config[key]))
    return text
