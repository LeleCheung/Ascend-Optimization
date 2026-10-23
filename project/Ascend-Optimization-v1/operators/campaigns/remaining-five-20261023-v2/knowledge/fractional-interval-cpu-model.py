"""CPU验证池化窗口起点的scalar_t舍入，不能当作NPU验证或性能成绩。"""
from pathlib import Path
import hashlib,importlib.util,json
import torch
import torch.nn.functional as F

torch.set_num_threads(2)
batch=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/remaining-five-20261023-v2')
leaf=batch/'catalog/ops/fractional_max_pool3d'
spec=importlib.util.spec_from_file_location('fractional_cpu_oracle',leaf/'oracle.py')
oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)

def model(args,rounded):
    x=args['input'];samples=args['_random_samples'];dtype=x.dtype
    n,c,d,h,w=x.shape
    sizes=args['output_size'];ks=args['kernel_size']
    def seq(axis,ins,outs,k):
        sample=samples[...,axis].float()
        alpha=torch.tensor((ins-k)/(outs-1) if outs>1 else 0.,dtype=dtype if rounded else torch.float32).float()
        pos=torch.arange(outs).float()[None,None,:]+sample[...,None]
        if rounded:pos=pos.to(dtype).float()
        pos=pos*alpha;base=sample*alpha
        if rounded:pos=pos.to(dtype).float();base=base.to(dtype).float()
        seq=pos.long()-base.long()[...,None]
        if outs:seq[...,-1]=ins-k
        return seq
    sd,sh,sw=[seq(i,ins,outs,k) for i,(ins,outs,k) in enumerate(zip([d,h,w],sizes,ks))]
    shape=(n,c,*sizes)
    values=torch.full(shape,float('-inf'),dtype=dtype)
    indices=torch.zeros(shape,dtype=torch.int64)
    ni=torch.arange(n)[:,None,None,None,None];ci=torch.arange(c)[None,:,None,None,None]
    for kd in range(ks[0]):
        for kh in range(ks[1]):
            for kw in range(ks[2]):
                dd=sd[...,None,None]+kd;hh=sh[...,None,:,None]+kh;ww=sw[...,None,None,:]+kw
                current=x[ni,ci,dd,hh,ww]
                index=(dd*h+hh)*w+ww
                if kd==kh==kw==0:indices=index.expand(shape).clone()
                choose=(current>values)|torch.isnan(current)
                values=torch.where(choose,current,values)
                indices=torch.where(choose,index,indices)
    return values,indices

rows=[]
cases=[json.loads(line) for phase in ['correctness','timing'] for line in (leaf/(phase+'.jsonl')).read_text().splitlines() if line.strip()]
for case in cases:
    args=oracle.gen_inputs(case,'cpu');expected=oracle.run(**args)
    row=dict(case=case['name'],dtype=str(args['input'].dtype),models={})
    for rounded,label in [(False,'fp32_intervals'),(True,'scalar_dtype_intervals')]:
        got=model(args,rounded)
        row['models'][label]=dict(values_equal=torch.equal(got[0],expected[0]),indices_equal=torch.equal(got[1],expected[1]),
            value_mismatches=int((got[0]!=expected[0]).sum()),index_mismatches=int((got[1]!=expected[1]).sum()))
    rows.append(row)
result=dict(scope='CPU数学与准确参考源码对照；不是NPU编译、执行或性能证明',cases=rows,
            total=len(rows),passed=sum(r['models']['scalar_dtype_intervals']['values_equal'] and r['models']['scalar_dtype_intervals']['indices_equal'] for r in rows))
target=batch/'knowledge/fractional-interval-cpu-result.json'
assert not target.exists();target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False),flush=True)
