"""额外语义门禁与同输入设备配对确认；不接触预登记独立形状。"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.append(str(HERE.parent/'mechanism'))
from probe import load,save,compiled_evidence
from measure import pair


def check(op,source,seed,out,semantic_only=False):
    if out.exists():raise ValueError('使用新目录')
    import torch
    import torch_npu
    from triton.backends.ascend.testing import do_bench_npu
    helpers=load(HERE.parent/'mechanism/probe.py','device_check_helpers') if (HERE.parent/'mechanism/probe.py').exists() else load(HERE/'launch-probe.py','device_check_helpers')
    candidate=load(source,'device_check_candidate');parent=load(seed,'device_check_parent')
    records=[];paths=[];out.mkdir(parents=True)
    for dt in ['float16','bfloat16','float32']:
        dtype=getattr(torch,dt)
        for shape in ([[29,35,67],[64,128,64],[256,256,256]] if op=='matmul_bias_activation' else [[29,35],[5,37,41],[128,256]]):
            for strided in [False,True]:
                row=dict(dtype=dt,shape=shape,strided=strided,status='failed');records.append(row)
                generator=torch.Generator().manual_seed(71917+len(records))
                def tensor(s):
                    v=torch.randn(s,generator=generator).to(device='npu:0',dtype=dtype)
                    if strided:
                        storage=torch.empty((*s,2),dtype=dtype,device='npu:0');storage[...,0]=v;return storage[...,0]
                    return v
                try:
                    if op=='matmul_bias_activation':
                        m,n,k=shape;a=tensor([m,k]);b=tensor([k,n]);bias=tensor([n]);inputs=dict(input=a,weight=b,bias=bias)
                        expected=torch.relu(a@b+bias);atol=1e-4*k;rtol={'float16':1e-3,'bfloat16':.016,'float32':1.3e-6}[dt]
                    else:
                        x=tensor(shape)
                        if not strided:
                            x.reshape(-1)[0]=float('-inf');x.reshape(-1)[-1]=float('nan')
                        inputs=dict(inp=x,dim=1,keepdim=strided);expected=torch.amin(x,dim=1,keepdim=strided);atol=rtol=0
                    saved={k:v.clone() for k,v in inputs.items() if isinstance(v,torch.Tensor)}
                    original={k:v for k,v in inputs.items() if isinstance(v,torch.Tensor)}
                    # 同时存活的第二组输入，防止错误缓存旧指针。
                    candidate.run(**inputs)
                    with torch.npu.stream(torch.npu.Stream()):
                        torch.npu.current_stream().wait_stream(torch.npu.default_stream())
                        actual=candidate.run(**{k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in inputs.items()})
                    torch.npu.synchronize()
                    torch.testing.assert_close(actual,expected,atol=atol,rtol=rtol,equal_nan=True)
                    for key,tensor_value in original.items():torch.testing.assert_close(tensor_value,saved[key],atol=0,rtol=0,equal_nan=True)
                    row['status']='passed'
                except Exception as error:row['error']=type(error).__name__+': '+str(error)[-1800:];torch.npu.synchronize()
    if semantic_only or any(row['status']!='passed' for row in records):
        result=dict(operator=op,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),seed_sha256=hashlib.sha256(seed.read_bytes()).hexdigest(),
            checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=records,
            passed=sum(r['status']=='passed' for r in records),total=len(records),confirmations=[],device_geomean=None,semantic_only=semantic_only)
        result['status']='passed' if result['passed']==result['total'] else 'failed';save(out/'check.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='cases'},ensure_ascii=False),flush=True)
        return result['status']!='passed'
    for ci,shape in enumerate([[256,256,256],[1024,1024,1024]] if op=='matmul_bias_activation' else [[4096,4096],[64,512,512]]):
        for dt in ['float16','float32']:
            dtype=getattr(torch,dt);generator=torch.Generator().manual_seed(72001+ci)
            def random(s):return torch.randn(s,generator=generator).to(dtype=dtype,device='npu:0')
            inputs=(dict(input=random([shape[0],shape[2]]),weight=random([shape[2],shape[1]]),bias=random([shape[1]])) if op=='matmul_bias_activation' else
                    dict(inp=random(shape),dim=1,keepdim=False))
            before,old=helpers.capture_call(lambda _:parent.run(**inputs),None)
            after,new=helpers.capture_call(lambda _:candidate.run(**inputs),None)
            if len(before)!=1 or len(after)!=1:raise ValueError('配对确认只支持捕获到的单kernel路径')
            tol=1e-4*shape[-1] if op=='matmul_bias_activation' else 0
            torch.testing.assert_close(new,old,atol=tol,rtol={'float16':1e-3,'bfloat16':.016,'float32':1.3e-6}[dt] if tol else 0,equal_nan=True)
            functions=dict(parent_ms=helpers.direct_call(before),candidate_ms=helpers.direct_call(after));times={key:[] for key in functions}
            for repeat in range(5):
                keys=list(functions)
                if repeat%2:keys.reverse()
                measured=pair({key:functions[key] for key in keys},dict(parent_ms=before[0]['jit'].fn.__name__,candidate_ms=after[0]['jit'].fn.__name__),
                              out/f'benchmark-{ci}-{dt}-{repeat}')
                for key,value in measured.items():times[key].append(value)
            paths.append(dict(shape=shape,dtype=dt,timings=times,
                paired_speedup=statistics.median([a/b for a,b in zip(times['parent_ms'],times['candidate_ms'])]),
                parent_ir=compiled_evidence(before,out/f'confirm-{ci}-{dt}'/'parent'),
                candidate_ir=compiled_evidence(after,out/f'confirm-{ci}-{dt}'/'candidate')))
    result=dict(operator=op,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),seed_sha256=hashlib.sha256(seed.read_bytes()).hexdigest(),
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=records,passed=sum(r['status']=='passed' for r in records),total=len(records),
        confirmations=paths,device_geomean=statistics.geometric_mean(r['paired_speedup'] for r in paths),
        scope='开发集同输入交错设备kernel计时；非独立集，非完整调用walltime；矩阵乘合同为有限随机输入')
    result['status']='passed' if result['passed']==result['total'] else 'failed';save(out/'check.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','confirmations']},ensure_ascii=False),flush=True)
    return result['status']!='passed'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--operator',required=True)
    for name in ['source','seed','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--semantic-only',action='store_true')
    a=p.parse_args();raise SystemExit(check(a.operator,a.source,a.seed,a.output,a.semantic_only))
