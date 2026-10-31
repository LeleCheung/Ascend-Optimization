"""交付前追加叉积检查：partial tile、共享strided storage、特殊值与当前stream。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-failure',action='store_true',help='仅用于明确登记的反例，不把失败当作通过')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    import torch,torch_npu
    spec=importlib.util.spec_from_file_location('cross_safety_candidate',args.source)
    candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
    rows=[]
    def check(name,body):
        row=dict(case=name,passed=False)
        try:body();row['passed']=True
        except Exception as error:row['error']=type(error).__name__+': '+str(error)[:2000]
        rows.append(row)
    for dtype in [torch.float16,torch.bfloat16,torch.float32]:
        tol=1e-4 if dtype==torch.float32 else .015
        def same_addresses(dtype=dtype,tol=tol):
            x=torch.randn((3,3),dtype=dtype,device='npu:0');y=torch.randn_like(x);out=torch.empty_like(x)
            for dim in [-1,0,-1,0]:
                expected=torch.cross(x,y,dim=dim)
                actual=candidate.run(x,y,dim,out=out);torch.npu.synchronize()
                torch.testing.assert_close(actual,expected,atol=tol,rtol=tol,equal_nan=True)
        check('same-addresses-changing-dim-'+str(dtype),same_addresses)
        for n in [1,2,17,340,341,342,681,682,683,43691]:
            def tail(n=n,dtype=dtype,tol=tol):
                x=torch.randn((n,3),dtype=dtype,device='npu:0');y=torch.randn_like(x)
                storage=torch.full((n*3+32,),37.,dtype=dtype,device='npu:0');out=storage[16:-16].view(n,3)
                expected=torch.cross(x,y,dim=-1)
                actual=candidate.run(x,y,-1,out=out);torch.npu.synchronize()
                torch.testing.assert_close(actual,expected,atol=tol,rtol=tol,equal_nan=True)
                assert torch._C._is_alias_of(actual,out)
                assert bool((storage[:16]==37.).all()) and bool((storage[-16:]==37.).all())
            check('partial-tile-'+str(dtype)+'-'+str(n),tail)
        for n in [17,129,1025]:
            for which in ['input','other']:
                def shared(n=n,which=which,dtype=dtype,tol=tol):
                    original=torch.randn((16*n+32,),dtype=dtype,device='npu:0')
                    storage=original.clone()
                    view=storage.as_strided((n,3),(16,2),0)
                    out=storage[8*n:11*n].view(n,3)
                    other=torch.randn((n,3),dtype=dtype,device='npu:0')
                    assert torch._C._is_alias_of(view,out)
                    # 逻辑numel形成的字节范围不足以检测此共享storage。
                    assert view.data_ptr()+view.numel()*view.element_size()<=out.data_ptr()
                    for _ in range(3):
                        storage.copy_(original)
                        a,b=(view,other) if which=='input' else (other,view)
                        expected=torch.cross(a.clone(),b.clone(),dim=-1)
                        actual=candidate.run(a,b,-1,out=out);torch.npu.synchronize()
                        torch.testing.assert_close(actual,expected,atol=tol,rtol=tol,equal_nan=True)
                check('shared-strided-storage-'+str(dtype)+'-'+which+'-'+str(n),shared)
        for number in range(2):
            def stream(dtype=dtype,tol=tol):
                current=torch.npu.Stream()
                with torch.npu.stream(current):
                    x=torch.tensor([[float('nan'),1.,2.],[float('inf'),0.,1.],[-0.,2.,-1.]],device='npu:0',dtype=dtype).repeat(17,1)
                    y=torch.randn_like(x);out=torch.empty_like(x)
                    expected=torch.cross(x,y,dim=-1)
                    actual=candidate.run(x,y,-1,out=out)
                current.synchronize()
                torch.testing.assert_close(actual,expected,atol=tol,rtol=tol,equal_nan=True)
            check('special-values-current-stream-'+str(dtype)+'-'+str(number),stream)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    checker_bytes=Path(__file__).read_bytes();checker_sha=hashlib.sha256(checker_bytes).hexdigest()
    archived_checker=args.output.parent/('cross-checker-'+checker_sha+'.py')
    if archived_checker.exists() and archived_checker.read_bytes()!=checker_bytes:raise FileExistsError(archived_checker)
    archived_checker.write_bytes(checker_bytes)
    value=dict(status='passed' if all(row['passed'] for row in rows) else 'failed',operator='cross_out',
        source=str(args.source),source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        checker=str(archived_checker),checker_sha256=checker_sha,
        passed=sum(row['passed'] for row in rows),total=len(rows),cases=rows,
        scope='追加语义与输出邻接存储检查；普通张量测试不能证明无越界读取，读取掩码另作源码审查')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({key:value[key] for key in ['status','passed','total','source_sha256']}))
    if value['status']!='passed' and not args.allow_failure:raise SystemExit('叉积追加检查失败')


if __name__=='__main__':main()
