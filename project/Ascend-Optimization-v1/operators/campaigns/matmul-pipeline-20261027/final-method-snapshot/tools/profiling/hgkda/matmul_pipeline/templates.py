"""生成通用多输出分块；保持数学、整除门禁和未知布局父路径。"""
import ast

CONFIGS=[
 dict(id='dual-n-constexpr',bm=128,bn=128,bk=256,tm=1,tn=2,group=0,persistent=0,multi=True,stages=2,saving=False,interleave=False),
 dict(id='dual-m-reuse-b',bm=128,bn=128,bk=256,tm=2,tn=1,group=0,persistent=0,multi=True,stages=2,saving=False,interleave=False),
 dict(id='quad-64x128',bm=64,bn=128,bk=256,tm=2,tn=2,group=0,persistent=0,multi=True,stages=2,saving=False,interleave=False),
 dict(id='quad-128x64',bm=128,bn=64,bk=256,tm=2,tn=2,group=0,persistent=0,multi=True,stages=2,saving=False,interleave=False),
 dict(id='four-n-64',bm=64,bn=64,bk=256,tm=1,tn=4,group=0,persistent=0,multi=True,stages=2,saving=False,interleave=False),
 dict(id='dual-n-saving',bm=128,bn=128,bk=256,tm=1,tn=2,group=0,persistent=0,multi=True,stages=2,saving=True,interleave=False),
 dict(id='dual-n-interleave',bm=128,bn=128,bk=256,tm=1,tn=2,group=0,persistent=0,multi=True,stages=2,saving=False,interleave=True),
 dict(id='dual-n-group4',bm=128,bn=128,bk=256,tm=1,tn=2,group=4,persistent=0,multi=True,stages=2,saving=False,interleave=False),
 dict(id='dual-n-persistent20',bm=128,bn=128,bk=256,tm=1,tn=2,group=0,persistent=20,multi=True,stages=2,saving=False,interleave=False),
 dict(id='dual-n-stage1',bm=128,bn=128,bk=256,tm=1,tn=2,group=0,persistent=0,multi=True,stages=1,saving=False,interleave=False),
]

def validate(c):
    if set(c)!=set(CONFIGS[0]):raise ValueError('配置键不完整')
    if any(type(c[k]) is not int for k in ['bm','bn','bk','tm','tn','group','persistent','stages']):raise ValueError('整数配置类型不符')
    if c['bm'] not in [64,128] or c['bn'] not in [64,128] or c['bk'] not in [128,256,512]:raise ValueError('分块越界')
    if c['tm'] not in [1,2] or c['tn'] not in [1,2,4] or c['tm']*c['tn']>4:raise ValueError('输出块数越界')
    if c['group'] not in [0,4] or c['persistent'] not in [0,20] or c['stages'] not in [1,2]:raise ValueError('流水配置越界')
    if any(type(c[k]) is not bool for k in ['multi','saving','interleave']):raise ValueError('布尔参数类型不符')

def generate(parent,c):
    validate(c);bm,bn,bk,tm,tn=[c[k] for k in ['bm','bn','bk','tm','tn']]
    body=['rk=tl.arange(0,BK)','nm=M//(BM*TM)','nn=N//(BN*TN)']
    if c['group']:
        body+=['width=4*nn','first=(tile//width)*4','group_size=tl.minimum(nm-first,4)',
               'pm=first+tile%group_size','pn=(tile%width)//group_size']
    else:body+=['pm=tile%nm','pn=tile//nm']
    for i in range(tm):body+=[f'm{i}=(pm*TM+{i})*BM+tl.arange(0,BM)',f'ap{i}=A+m{i}[:,None]*K+rk[None,:]']
    for j in range(tn):body+=[f'n{j}=(pn*TN+{j})*BN+tl.arange(0,BN)',f'bp{j}=B+rk[:,None]*N+n{j}[None,:]']
    for i in range(tm):
        for j in range(tn):body+=[f'c{i}{j}=tl.zeros((BM,BN),tl.float32)']
    loop=[]
    for i in range(tm):loop+=[f'a{i}=tl.load(ap{i})']
    if not c['interleave']:
        for j in range(tn):loop+=[f'b{j}=tl.load(bp{j})']
    for j in range(tn):
        if c['interleave']:loop+=[f'b{j}=tl.load(bp{j})']
        for i in range(tm):loop+=[f'c{i}{j}=tl.dot(a{i},b{j},c{i}{j},allow_tf32=False)']
    for i in range(tm):loop+=[f'ap{i}+=BK']
    for j in range(tn):loop+=[f'bp{j}+=BK*N']
    body+=['for kk in range(K//BK):']+['    '+x for x in loop]
    for j in range(tn):
        body+=[f'bias{j}=tl.load(Bias+n{j})']
        for i in range(tm):body+=[f'v{i}{j}=c{i}{j}+bias{j}[None,:]',f'v{i}{j}=tl.maximum(v{i}{j},v{i}{j}*0.0)',
            f'tl.store(C+m{i}[:,None]*N+n{j}[None,:],v{i}{j}.to(bias{j}.dtype))']
    definition='\n\n@triton.jit\ndef hg_mba_reuse_kernel(A,B,Bias,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,TM:tl.constexpr,TN:tl.constexpr):\n'
    if c['persistent']:
        definition+='    for tile in range(tl.program_id(0),(M//(BM*TM))*(N//(BN*TN)),tl.num_programs(0)):\n'
        definition+='\n'.join('        '+x for x in body)+'\n'
    else:definition+='    tile=tl.program_id(0)\n'+'\n'.join('    '+x for x in body)+'\n'
    wrapper=f'''
_hg_pipeline_parent=run
def run(input,weight,bias):
    m,k=input.shape;n=weight.shape[1]
    bm,bn,bk,tm,tn={bm},{bn},{bk},{tm},{tn}
    if not (input.dtype in (torch.float16,torch.bfloat16) and min(m,n,k)>=512
            and m%(bm*tm)==0 and n%(bn*tn)==0 and k%bk==0 and k>=2*bk
            and input.is_contiguous() and weight.is_contiguous() and bias.is_contiguous() and bias.numel()==n):
        return _hg_pipeline_parent(input,weight,bias)
    out=torch.empty((m,n),dtype=input.dtype,device=input.device)
    programs=(m//(bm*tm))*(n//(bn*tn))
    grid=min(programs,{c['persistent']}) if {c['persistent']} else programs
    with torch_device_fn.device(input.device):
        hg_mba_reuse_kernel[(grid,)](input,weight,bias.reshape(-1),out,m,n,k,bm,bn,bk,tm,tn,
            multibuffer={c['multi']},unit_flag=True,sync_solver=False,num_stages={c['stages']},enable_ubuf_saving={c['saving']})
    return out
matmul_bias_activation=run
'''
    text=parent+definition+wrapper;ast.parse(text);return text

def logical(c,m,n,k):
    bm,bn,tm,tn=[c[x] for x in ['bm','bn','tm','tn']]
    return dict(programs=(m//(bm*tm))*(n//(bn*tn)),
        logical_A_elements=m*k*(n//(bn*tn)),logical_B_elements=k*n*(m//(bm*tm)),
        accumulator_bytes=bm*bn*tm*tn*4,scope='源码逻辑请求与累加器下界；不是实际HBM／UB资源占用')
