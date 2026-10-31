"""让特殊值确实经过对齐Ascend C路径，同时检查换地址、stream、out和原地语义。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def imported(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['operator','source','leaf','output']:parser.add_argument('--'+name,required=True)
    a=parser.parse_args();source=Path(a.source);out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    import torch,torch_npu
    candidate=imported('native_safety_candidate',source)
    oracle=imported('native_safety_oracle',Path(a.leaf)/'oracle.py')
    op=a.operator
    streams=[torch.npu.current_stream(),torch.npu.Stream()]
    rows=[]
    for dtype in [torch.float16,torch.bfloat16,torch.float32]:
        edges=torch.tensor([0.,-0.,float('nan'),float('inf'),-float('inf'),
            -3.,3.,-6.,6.,-2.5,-1.5,-0.5,0.5,1.5,2.5,1.e-7,-1.e-7,
            0.009,-0.009,0.01,-0.01,0.011,-0.011,65504.,-65504.],dtype=dtype)
        cpu=edges.repeat((1024+len(edges)-1)//len(edges))[:1024].clone()
        if op=='log2_':cpu=cpu.abs()
        for index,offset in enumerate([0,1,3,0]):
            for stream_index,stream in enumerate(streams):
                row=dict(case=str(dtype)+'-offset-'+str(offset)+'-sequence-'+str(index)+'-stream-'+str(stream_index),
                    shape=[1024] if index%2==0 else [16,64],offset=offset,passed=False)
                try:
                    with torch.npu.stream(stream):
                        shape=tuple(row['shape'])
                        base=torch.empty(1024+offset+16,dtype=dtype,device='npu:0')
                        x=base[offset:offset+1024].view(shape);x.copy_(cpu.view(shape).to('npu:0'))
                        reference=x.clone()
                        before=x.clone()
                        kwargs={};reference_kwargs={}
                        if op in {'negative_','expm1_','log2_'}:kwargs=dict(A=x);reference_kwargs=dict(A=reference)
                        elif op=='fix':kwargs=dict(self=x);reference_kwargs=dict(self=reference)
                        elif op=='hardswish':kwargs=dict(x=x);reference_kwargs=dict(x=reference)
                        elif op in {'round_out','special_round_out'}:
                            output=torch.empty(shape,dtype=dtype,device='npu:0')
                            kwargs={('input' if op=='round_out' else 'self'):x,'out':output,'decimals':0}
                            reference_kwargs={('input' if op=='round_out' else 'self'):reference,'out':torch.empty_like(output),'decimals':0}
                        elif op=='fmod_tensor_':
                            other=torch.tensor([0.,-0.,1.,-1.,2.,-2.,float('inf'),float('nan'),0.1],dtype=dtype).repeat(114)[:1024].to('npu:0').view(shape)
                            kwargs=dict(A=x,B=other);reference_kwargs=dict(A=reference,B=other.clone())
                        else:raise ValueError(op)
                        expected=oracle.run(**reference_kwargs);actual=candidate.run(**kwargs)
                        torch.npu.synchronize()
                        tol=.0001 if dtype==torch.float32 else .015
                        torch.testing.assert_close(actual,expected,atol=tol,rtol=tol,equal_nan=True)
                        if op in {'negative_','expm1_','log2_','fmod_tensor_'}:
                            assert torch._C._is_alias_of(actual,x),'原地别名丢失'
                        elif op in {'round_out','special_round_out'}:
                            assert torch._C._is_alias_of(actual,output),'out别名丢失'
                            torch.testing.assert_close(x,before,atol=0,rtol=0,equal_nan=True)
                        else:
                            assert not torch._C._is_alias_of(actual,x),'输出错误复用输入'
                            torch.testing.assert_close(x,before,atol=0,rtol=0,equal_nan=True)
                        zero=(expected==0)&(actual==0)
                        if op=='expm1_':
                            target=torch.expm1(cpu).view(shape).to('npu:0');zero=(target==0)&(actual==0)
                            assert torch.equal(torch.signbit(actual[zero]),torch.signbit(target[zero])),'零符号与CPU数学参考不同'
                        else:
                            assert torch.equal(torch.signbit(actual[zero]),torch.signbit(expected[zero])),'零符号与NPU参考不同'
                        row.update(passed=True,input_aligned=(x.data_ptr()%32==0),elements=x.numel())
                except Exception as error:row['error']=type(error).__name__+': '+str(error)[:2200]
                rows.append(row)
    value=dict(operator=op,status='passed' if all(r['passed'] for r in rows) else 'failed',source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        passed=sum(r['passed'] for r in rows),total=len(rows),cases=rows,
        scope='1024元素特殊值确实覆盖连续对齐native路径；原始KGS与原补测独立保留')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:value[k] for k in ['operator','status','passed','total']}))
    raise SystemExit(0 if value['status']=='passed' else 1)


if __name__=='__main__':main()
