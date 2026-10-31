"""核验来源可见的Ascend C补充参考与原生库绑定，结果独立于正式速度成绩。"""
import argparse
import json
import os
from pathlib import Path


def check(output, library):
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7'
    if output.exists():raise FileExistsError(output)
    import torch
    import torch_npu
    from ascendc_reference import Negative
    from cann_reference import Native
    reference=Negative(library)
    rows=[]
    for n in [0,8,1024,65536]:
        row=dict(n=n,status='failed')
        try:
            cpu=torch.randn(n,generator=torch.Generator().manual_seed(1729))
            if n>=8:cpu[:7]=torch.tensor([0.,-0.,float('nan'),float('inf'),-float('inf'),1.,-1.])
            x=cpu.to('npu:0');expected=x.neg();actual=reference.run(x);torch.npu.synchronize()
            assert torch._C._is_alias_of(actual,x)
            torch.testing.assert_close(actual,expected,atol=0,rtol=0,equal_nan=True)
            a,e=actual.cpu(),expected.cpu();assert torch.equal(torch.signbit(a[e==0]),torch.signbit(e[e==0]))
            row['status']='passed'
        except Exception as error:row['error']=type(error).__name__+': '+str(error)
        rows.append(row)
    rejected=[]
    for kind,x in [('strided',torch.empty((8,2),device='npu:0')[:,0]),('float16',torch.empty(8,dtype=torch.float16,device='npu:0')),('unaligned',torch.empty(9,device='npu:0'))]:
        try:reference.run(x);rejected.append(dict(kind=kind,passed=False))
        except ValueError:rejected.append(dict(kind=kind,passed=True))
    native={}
    for op in ['expm1_','native_group_norm']:
        binding=Native(op);checked=[]
        for dtype in [torch.float16,torch.float32]:
            for variant in [False,True]:
                row=dict(dtype=str(dtype),variant='strided' if variant and op=='expm1_' else 'no_affine' if variant else 'default',status='failed')
                try:
                    generator=torch.Generator().manual_seed(1729)
                    if op=='expm1_':
                        cpu=torch.randn((7,9),generator=generator,dtype=torch.float32).to(dtype)
                        cpu.flatten()[:7]=torch.tensor([0.,-0.,float('nan'),float('inf'),-float('inf'),-1.,1.],dtype=dtype)
                        x=cpu.to('npu:0')
                        if variant:
                            base=torch.empty((7,9,2),device=x.device,dtype=dtype);base[...,0]=x;x=base[...,0]
                        expected=torch.expm1(x);actual=binding.run(A=x)
                        assert torch._C._is_alias_of(actual,x)
                    else:
                        x=torch.randn((2,8,3,5),generator=generator).to(device='npu:0',dtype=dtype)
                        weight=None if variant else torch.ones(8,device=x.device,dtype=dtype)
                        bias=None if variant else torch.zeros(8,device=x.device,dtype=dtype)
                        expected=torch.native_group_norm(x,weight,bias,2,8,15,4,1e-5)
                        actual=binding.run(input=x,weight=weight,bias=bias,N=2,C=8,HxW=15,group=4,eps=1e-5)
                    torch.npu.synchronize()
                    torch.testing.assert_close(actual,expected,atol=.015 if dtype==torch.float16 else 1e-4,rtol=.015 if dtype==torch.float16 else 1e-4,equal_nan=True)
                    row['status']='passed'
                except Exception as error:row['error']=type(error).__name__+': '+str(error)[:1800]
                checked.append(row)
        native[op]=dict(provenance=binding.provenance(),cases=checked,status='passed' if all(r['status']=='passed' for r in checked) else 'failed')
    d=dict(status='passed' if all(r['status']=='passed' for r in rows) and all(r['passed'] for r in rejected) else 'failed',
           ascendc=dict(provenance=reference.provenance(),cases=rows,unsupported_rejected=rejected),
           cann_bindings=native,
           scope='仅参考资格核验；不作为优化候选或厂商最优速度成绩')
    if any(r['status']!='passed' for r in native.values()):d['status']='failed'
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return d


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--library',type=Path,default=Path('/data/hanle/ascend-optimization/runtime/next25-ascendc-build/lib/libhgkda_negative.so'));a=p.parse_args()
    result=check(a.output,a.library);print(json.dumps(dict(status=result['status']),ensure_ascii=False))
    if result['status']!='passed':raise SystemExit(1)
