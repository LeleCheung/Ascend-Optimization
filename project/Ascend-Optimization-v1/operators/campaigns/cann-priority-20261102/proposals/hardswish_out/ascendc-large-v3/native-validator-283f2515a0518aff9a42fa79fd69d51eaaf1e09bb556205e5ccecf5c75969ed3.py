"""验证对齐native路径的特殊值、分块边界、out别名及当前stream。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import torch
import torch_npu


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+chr(10),encoding='utf8')


def imported(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    p=argparse.ArgumentParser()
    for n in ['operator','source','leaf','output']:p.add_argument('--'+n,type=str if n=='operator' else Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    if a.output.exists():raise FileExistsError(a.output)
    candidate=imported('native_pointwise_candidate',a.source)
    oracle=imported('native_pointwise_oracle',a.leaf/'oracle.py')
    rows=[]
    streams=[torch.npu.current_stream(),torch.npu.Stream()]
    for dtype in [torch.float16,torch.bfloat16,torch.float32]:
        edges=torch.tensor([0.,-0.,float('nan'),float('inf'),-float('inf'),-3.,3.,-6.,6.,
            -2.5,-1.5,-0.5,0.5,1.5,2.5,1.e-7,-1.e-7,65504.,-65504.],dtype=dtype)
        lengths = [1024,8176,8192,8208,65536] if a.operator=='rad2deg_' else [1024,16368,16384,16400,65536]
        for n in lengths:
            cpu=edges.repeat((n+edges.numel()-1)//edges.numel())[:n]
            for offset in [0,1]:
                for sid,stream in enumerate(streams):
                    row=dict(dtype=str(dtype),elements=n,offset=offset,stream=sid,passed=False)
                    try:
                        with torch.npu.stream(stream):
                            storage=torch.empty(n+offset+16,dtype=dtype,device='npu:0')
                            storage.fill_(37.)
                            x=storage[offset:offset+n];x.copy_(cpu.to('npu:0'));before=x.clone()
                            if a.operator=='hardswish_out':
                                # 同一输入同时覆盖独立out和原地别名out。
                                for alias in [False,True]:
                                    x.copy_(before)
                                    out_storage=torch.empty(n+16,dtype=dtype,device='npu:0').fill_(37.)
                                    out=x if alias else out_storage[:n]
                                    ref=x.clone();expected=oracle.run(x=ref,out=torch.empty_like(ref))
                                    actual=candidate.run(x=x,out=out);torch.npu.synchronize()
                                    tol=1.e-4 if dtype==torch.float32 else .015
                                    torch.testing.assert_close(actual,expected,atol=tol,rtol=tol,equal_nan=True)
                                    assert torch._C._is_alias_of(actual,out)
                                    if not alias:torch.testing.assert_close(x,before,atol=0,rtol=0,equal_nan=True)
                                    if not alias:assert bool((out_storage[n:]==37.).all())
                                    zero=expected==0
                                    assert torch.equal(torch.signbit(actual[zero]),torch.signbit(expected[zero]))
                            elif a.operator=='rad2deg_':
                                expected=oracle.run(A=x.clone());actual=candidate.run(A=x);torch.npu.synchronize()
                                tol=1.e-4 if dtype==torch.float32 else .015
                                torch.testing.assert_close(actual,expected,atol=tol,rtol=tol,equal_nan=True)
                                assert torch._C._is_alias_of(actual,x)
                                zero=expected==0
                                assert torch.equal(torch.signbit(actual[zero]),torch.signbit(expected[zero]))
                            else:
                                expected=oracle.run(x=x.clone());actual=candidate.run(x=x);torch.npu.synchronize()
                                tol=1.e-4 if dtype==torch.float32 else .015
                                torch.testing.assert_close(actual,expected,atol=tol,rtol=tol,equal_nan=True)
                                assert not torch._C._is_alias_of(actual,x)
                                torch.testing.assert_close(x,before,atol=0,rtol=0,equal_nan=True)
                                zero=expected==0
                                assert torch.equal(torch.signbit(actual[zero]),torch.signbit(expected[zero]))
                            assert bool((storage[:offset]==37.).all()) and bool((storage[offset+n:]==37.).all())
                            row.update(passed=True,input_aligned=x.data_ptr()%32==0)
                    except Exception as e:row['error']=type(e).__name__+': '+str(e)[:1800]
                    rows.append(row)
    frozen=a.output.parent/('native-validator-'+sha(__file__)+'.py')
    if frozen.exists():assert frozen.read_bytes()==Path(__file__).read_bytes()
    else:frozen.write_bytes(Path(__file__).read_bytes())
    value=dict(status='passed' if all(v['passed'] for v in rows) else 'failed',operator=a.operator,
        source_sha256=sha(a.source),checker_source=dict(path=str(frozen),sha256=sha(frozen)),
        passed=sum(v['passed'] for v in rows),total=len(rows),cases=rows,
        scope='通用16K分块边界及对齐／未对齐地址，特殊值、零符号、两条stream与out别名；正式合同及布局补测独立保留')
    save(a.output,value)
    print(value['status'],value['passed'],value['total'])
    if value['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
