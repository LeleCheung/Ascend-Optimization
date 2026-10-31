"""检查厂商参考的动态参数、独立布局和当前stream，不修改原KGS合同。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
CHECKER_BYTES=Path(__file__).read_bytes()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
    parser=argparse.ArgumentParser()
    for n in ['operator','source','leaf','output']:parser.add_argument('--'+n,required=True)
    a=parser.parse_args();source=Path(a.source);leaf=Path(a.leaf);out=Path(a.output)
    assert not out.exists() and os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7'
    import torch,torch_npu
    oracle=load('semantic_oracle',leaf/'oracle.py');candidate=load('semantic_cann',source);rows=[]
    streams=[torch.npu.current_stream(),torch.npu.Stream()]
    for dtype in [torch.float16,torch.bfloat16,torch.float32]:
        tol=next(json.loads(t)['tolerance'] for t in (leaf/'correctness.jsonl').read_text().splitlines() if json.loads(t)['inputs']['spec']['dtype']==str(dtype).split('.')[-1])
        if a.operator=='fake_quantize_per_tensor_affine_cachemask_out':
            for i,(scale,zp,qmin,qmax) in enumerate([(0.05,0,-128,127),(0.2,7,0,255),(0.3,-3,-32,31)]):
                row=dict(dtype=str(dtype),case='dynamic-qparams-'+str(i),passed=False,scale=scale,zero_point=zp,quant_min=qmin,quant_max=qmax)
                try:
                    stream=streams[i%2]
                    with torch.npu.stream(stream):
                        values=torch.tensor([-100.,-1.05,-.45,-.15,-0.,0.,.15,.45,1.05,100.],dtype=dtype)
                        backing=torch.empty((values.numel(),3),device='npu:0',dtype=dtype);x=backing[:,1];x.copy_(values.to('npu:0'))
                        saved=x.clone();yb=torch.empty((values.numel(),5),device='npu:0',dtype=dtype);out0=yb[:,2]
                        mb=torch.empty((values.numel(),2),device='npu:0',dtype=torch.bool);out1=mb[:,1]
                        expected=oracle.run(values.to('npu:0'),scale,zp,qmin,qmax,out0=torch.empty_like(x),out1=torch.empty_like(x,dtype=torch.bool))
                        actual=candidate.run(x,scale,zp,qmin,qmax,out0=out0,out1=out1)
                        assert len(actual)==len(expected)==2
                        assert actual[0] is out0 and actual[1] is out1
                        snapshots=[t.clone() for t in actual];refs=[t.clone() for t in expected]
                    stream.synchronize();torch.testing.assert_close(x,saved,atol=0,rtol=0,equal_nan=True)
                    for actual_tensor,expected_tensor in zip(snapshots,refs):torch.testing.assert_close(actual_tensor,expected_tensor,atol=tol['atol'],rtol=tol['rtol'],equal_nan=True)
                    row['passed']=True
                except Exception as e:row['error']=type(e).__name__+': '+str(e)[:2000]
                rows.append(row)
        elif a.operator=='batch_norm_with_update_functional':
            for i,(momentum,eps,missing) in enumerate([(0.,1.e-3,'weight'),(.5,1.e-4,'bias'),(1.,1.e-2,'both')]):
                row=dict(dtype=str(dtype),case='dynamic-bn-'+str(i),passed=False,momentum=momentum,eps=eps,missing=missing)
                try:
                    stream=streams[i%2]
                    with torch.npu.stream(stream):
                        g=torch.Generator().manual_seed(82913+i);cpu=torch.randn((2,3,5,7),generator=g).to(dtype)
                        backing=torch.empty((2,3,5,14),device='npu:0',dtype=dtype);x=backing[...,::2];x.copy_(cpu.to('npu:0'))
                        weight=None if missing in ['weight','both'] else torch.randn(3,generator=g).to(device='npu:0',dtype=dtype)
                        bias=None if missing in ['bias','both'] else torch.randn(3,generator=g).to(device='npu:0',dtype=dtype)
                        mean=torch.randn(3,generator=g).to(device='npu:0',dtype=dtype);var=(torch.rand(3,generator=g)+.1).to(device='npu:0',dtype=dtype)
                        originals=[t.clone() if t is not None else None for t in [x,weight,bias,mean,var]]
                        reference=oracle.run(x.clone(),weight,bias,mean.clone(),var.clone(),momentum,eps)
                        actual=candidate.run(x,weight,bias,mean,var,momentum,eps)
                        assert len(actual)==len(reference)==6
                        snapshots=[t.clone() for t in actual];refs=[t.clone() for t in reference]
                    stream.synchronize()
                    for actual_tensor,expected_tensor in zip(snapshots,refs):torch.testing.assert_close(actual_tensor,expected_tensor,atol=tol['atol'],rtol=tol['rtol'],equal_nan=True)
                    for actual_tensor,expected_tensor in zip([x,weight,bias,mean,var],originals):
                        if actual_tensor is not None:torch.testing.assert_close(actual_tensor,expected_tensor,atol=0,rtol=0,equal_nan=True)
                    row['passed']=True
                except Exception as e:row['error']=type(e).__name__+': '+str(e)[:2000]
                rows.append(row)
        else:raise ValueError(a.operator)
    digest=hashlib.sha256(CHECKER_BYTES).hexdigest();checker=out.with_name('dynamic-validator-'+digest+'.py')
    if checker.exists():assert checker.read_bytes()==CHECKER_BYTES
    else:checker.write_bytes(CHECKER_BYTES)
    result=dict(status='passed' if len(rows)==9 and all(r['passed'] for r in rows) else 'failed',operator=a.operator,source=str(source),source_sha256=sha(source),checker_sha256=digest,
        checker_source=dict(path=str(checker),sha256=digest),
        contract_sha256={n:sha(leaf/n) for n in ['definition.json','oracle.py','correctness.jsonl','timing.jsonl']},
        reproduce_command=[sys.executable,str(Path(__file__).resolve()),'--operator',a.operator,'--source',str(source),'--leaf',str(leaf),'--output',str(out)],physical_device=7,
        passed=sum(r['passed'] for r in rows),total=len(rows),cases=rows,scope='动态参数、非连续独立布局和调用者stream的附加参考检查；不替代原合同')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+chr(10),encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k!='cases'},ensure_ascii=False))
if __name__=='__main__':main()
