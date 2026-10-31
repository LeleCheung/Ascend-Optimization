"""候选补充语义、动态指针／stream及调用路径检查；不修改候选。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from probe import load,capture_call,paired_measure


def check(operator,source,seed,output):
    if output.exists() or os.environ.get('ASCEND_RT_VISIBLE_DEVICES')!='7':raise ValueError('输出或设备范围错误')
    import torch
    import torch_npu
    torch.npu.set_device(0)
    assert torch.npu.device_count()==1
    m=load(source);records=[];keep=[]
    for dt in ['float16','bfloat16','float32']:
        for shape in [(),(0,),(17,),(1025,),(7,9),(2,3,5),(65537,)]:
            row=dict(dtype=dt,shape=list(shape),status='failed');records.append(row)
            try:
                dtype=getattr(torch,dt);n=1
                for d in shape:n*=d
                g=torch.Generator().manual_seed(826715)
                original=torch.randn(shape,generator=g).to(dtype)
                if n>=17:
                    original.reshape(-1)[:8]=torch.tensor([0.,-0.,float('inf'),-float('inf'),float('nan'),1.,-1.,.01],dtype=dtype)
                reference=getattr(torch,operator[:-1])(original)
                streams=[torch.npu.default_stream(),torch.npu.Stream()]
                for rep in [0,1]:
                    # 不同指针同时存活，防止缓存输入或复用旧输出蒙混过关。
                    with torch.npu.stream(streams[rep]):
                        step=2 if len(shape)>1 else 1
                        base=torch.full((n*step+8,),19.,device='npu:0',dtype=dtype);keep.append(base)
                        x=base[4:4+n*step:step].view(shape)
                        base[4:4+n*step:step].copy_(original.reshape(-1).to('npu:0'))
                        actual=m.run(x)
                    torch.npu.synchronize()
                    assert actual is x,'原地返回别名错误'
                    tol=1e-4 if dt=='float32' else .015
                    torch.testing.assert_close(x.cpu(),reference,atol=tol,rtol=tol,equal_nan=True)
                    assert bool((base[:4]==19).all()) and bool((base[-4:]==19).all())
                    if step==2:assert bool((base[5:4+n*2:2]==19).all())
                row['status']='passed'
            except Exception as e:row['error']=type(e).__name__+': '+str(e)
    old=load(seed);path=[]
    for shape in [[1024],[1048576]] if operator=='selu_' else [[1537],[1572864]]:
        original=torch.randn(shape,generator=torch.Generator().manual_seed(194401)).to(torch.float32)
        x=original.to('npu:0');y=x.clone()
        for _ in range(3):old.run(x);m.run(y)
        before,_=capture_call(old.run,x)
        after,_=capture_call(m.run,y)
        measurements=paired_measure(dict(seed_us=lambda:old.run(x),candidate_us=lambda:m.run(y)),
                                   lambda:(x.copy_(original),y.copy_(original)),trials=9,batch=12)
        path.append(dict(shape=shape,seed_warm_jit_calls=len(before),candidate_warm_jit_calls=len(after),timings=measurements,
            scope='调用路径干预检查；零JIT调用只说明绕过JIT，不证明设备指令未变'))
    result=dict(operator=operator,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=records,path_interventions=path,
        passed=sum(r['status']=='passed' for r in records),total=len(records))
    result['status']='passed' if result['passed']==result['total'] else 'failed'
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','path_interventions']},ensure_ascii=False),flush=True)
    return result['status']!='passed'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['source','seed','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--operator',required=True);a=p.parse_args()
    raise SystemExit(check(a.operator,a.source,a.seed,a.output))
