"""冻结候选后验证新形状，覆盖原生路径小值精度、尾块和输出独立性。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['operator','source','output']:p.add_argument('--'+k,required=True)
    a=p.parse_args();source=Path(a.source);out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    import torch,torch_npu
    s=importlib.util.spec_from_file_location('frozen_candidate',source);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    rows=[];streams=[torch.npu.current_stream(),torch.npu.Stream()]
    for dtype in [torch.float16,torch.bfloat16,torch.float32]:
        for shape in [(1536,),(96,160),(81920,),(384,640),(1105920,)]:
            for si,stream in enumerate(streams):
                row=dict(dtype=str(dtype),shape=shape,stream=si,passed=False)
                try:
                    cpu=torch.randn(shape,generator=torch.Generator().manual_seed(31031+si),dtype=torch.float32).to(dtype)
                    flat=cpu.flatten();edges=torch.tensor([0.,-0.,1.e-7,-1.e-7,1.e-4,-1.e-4,.009,-.009,.01,-.01,.011,-.011,float('nan'),float('inf'),-float('inf')],dtype=dtype)
                    flat[:len(edges)]=edges
                    with torch.npu.stream(stream):
                        x=cpu.to('npu:0');before=x.clone();reference=x.clone()
                        expected=reference.expm1_() if a.operator=='expm1_' else torch.fix(reference)
                        actual=m.run(A=x) if a.operator=='expm1_' else m.run(self=x)
                        torch.npu.synchronize();tol=.0001 if dtype==torch.float32 else .015
                        torch.testing.assert_close(actual,expected,atol=tol,rtol=tol,equal_nan=True)
                        if a.operator=='expm1_':
                            assert torch._C._is_alias_of(actual,x)
                            observed=actual.flatten()[:len(edges)].cpu();target=torch.expm1(edges)
                            torch.testing.assert_close(observed,target,atol=0,rtol=tol,equal_nan=True)
                            zero=target==0;assert torch.equal(torch.signbit(observed[zero]),torch.signbit(target[zero]))
                        else:
                            assert not torch._C._is_alias_of(actual,x)
                            torch.testing.assert_close(x,before,atol=0,rtol=0,equal_nan=True)
                        row.update(passed=True,input_aligned=x.data_ptr()%32==0)
                except Exception as e:row['error']=type(e).__name__+': '+str(e)[:1600]
                rows.append(row)
    value=dict(status='passed' if all(r['passed'] for r in rows) else 'failed',operator=a.operator,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=rows,passed=sum(r['passed'] for r in rows),total=len(rows),scope='冻结后新形状语义检查，性能不用于选版或替代原合同')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:value[k] for k in ['operator','status','passed','total']}));raise SystemExit(0 if value['status']=='passed' else 1)

if __name__=='__main__':main()
