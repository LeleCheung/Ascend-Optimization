#!/usr/bin/env python3
"""剩余五项的独立门禁：全部计分输入验数值，补布局/别名/stream语义。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True

TOOLS=Path(__file__).resolve().parent
def load(name,path,package=None):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    if package:m.__package__=package
    spec.loader.exec_module(m);return m

def check(op,source,output):
    if output.exists():raise FileExistsError(output)
    if os.environ.get('ASCEND_RT_VISIBLE_DEVICES')!='7':raise ValueError('仅允许物理卡7')
    import torch
    import torch_npu
    import torch.nn.functional as F
    assert torch.npu.device_count()==1
    helper=load('remaining_native_inputs',TOOLS/'check-endurance.py')
    leaf,timing=helper.native_timing_contract(source,op)
    oracle=load('remaining_oracle',leaf/'oracle.py')
    candidate=load('remaining_candidate',source,'flag_gems.ops' if op=='scatter_add' else None)
    correctness=[json.loads(line) for line in (leaf/'correctness.jsonl').read_text().splitlines() if line.strip()]
    stream=torch.npu.Stream();rows=[]
    def cpu(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu().clone(memory_format=torch.preserve_format)
        if isinstance(value,tuple):return tuple(cpu(v) for v in value)
        if isinstance(value,list):return [cpu(v) for v in value]
        if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
        return value
    def verify(expected,actual,tol):
        if isinstance(expected,torch.Tensor):
            assert isinstance(actual,torch.Tensor),'输出类型不同'
            assert actual.device.type=='npu','输出离开NPU'
            torch.testing.assert_close(cpu(actual),cpu(expected),atol=tol['atol'],rtol=tol['rtol'],check_dtype=True,equal_nan=True)
        elif isinstance(expected,(tuple,list)):
            assert isinstance(actual,type(expected)) and len(actual)==len(expected),'输出结构不同'
            for e,a in zip(expected,actual):verify(e,a,tol)
        else:assert expected==actual
    def one(name,factory,ref,tol,phase):
        row=dict(case=name,phase=phase,status='failed',repetitions=2,stream_switch=True)
        try:
            for repeat in range(2):
                args=factory();before=cpu(args)
                expected=ref(cpu(args))
                if repeat:
                    with torch.npu.stream(stream):actual=candidate.run(**args)
                else:actual=candidate.run(**args)
                torch.npu.synchronize()
                verify(expected,actual,tol)
                for key,value in args.items():
                    if isinstance(value,torch.Tensor):
                        torch.testing.assert_close(cpu(value),before[key],atol=0,rtol=0,equal_nan=True)
                if op=='_conj_physical':
                    x=args['input']
                    assert not actual.is_conj(),'输出不应带lazy conjugate bit'
                    if x.numel():assert actual.untyped_storage().data_ptr()!=x.untyped_storage().data_ptr(),'输出必须独立分配'
                    assert actual.stride()==expected.stride(),'输出布局与参考不同'
            row['status']='passed'
        except Exception as error:row['error']=type(error).__name__+': '+str(error)[:3500]
        rows.append(row)
    for phase,cases in [('correctness',correctness),('timing',timing)]:
        for case in cases:
            one(case['name'],lambda case=case:helper.native_timing_inputs(oracle,case,torch),
                lambda args:oracle.run(**args),case['tolerance'],phase)
    # 新三项的独立非计分覆盖。scatter/scan另调用上一轮专用门禁。
    if op=='_conj_physical':
        for dtype in [torch.float16,torch.bfloat16,torch.float32,torch.complex64]:
            for layout in ['transpose','stride2','offset','empty','conj']:
                def factory(dtype=dtype,layout=layout):
                    g=torch.Generator().manual_seed(11377)
                    base=torch.randn((5,14),generator=g,dtype=dtype).to('npu:0')
                    x=base.t() if layout=='transpose' else base[:,::2] if layout=='stride2' else base[1:,1:10] if layout=='offset' else base[:0] if layout=='empty' else base.conj()
                    return dict(input=x)
                one(str(dtype)+'-'+layout,factory,lambda a:torch.ops.aten._conj_physical(a['input']),dict(atol=0.,rtol=0.),'supplementary')
    elif op=='upsample_bilinear2d':
        for dtype in [torch.float16,torch.bfloat16,torch.float32]:
            for index in range(8):
                def factory(dtype=dtype,index=index):
                    g=torch.Generator().manual_seed(11990+index)
                    x=torch.randn((2,3,9,14),generator=g).to(device='npu:0',dtype=dtype)
                    if index%2:x=x[...,::2]
                    output=[[1,1],[1,13],[11,1],[17,19]][index//2]
                    return dict(input=x,output_size=output,align_corners=bool(index%2),scales_h=None,scales_w=None)
                one(str(dtype)+'-'+str(index),factory,
                    lambda a:torch.ops.aten.upsample_bilinear2d(a['input'],a['output_size'],a['align_corners'],a['scales_h'],a['scales_w']),
                    dict(atol=1e-4 if dtype==torch.float32 else .015,rtol=1e-4 if dtype==torch.float32 else .015),'supplementary')
    elif op=='fractional_max_pool3d':
        for dtype in [torch.float16,torch.bfloat16,torch.float32]:
            for index in range(6):
                def factory(dtype=dtype,index=index):
                    g=torch.Generator().manual_seed(92333+index)
                    shape=(2,3,7,9,11) if index<4 else (3,7,9,11)
                    x=torch.randn(shape,generator=g).to(device='npu:0',dtype=dtype)
                    if index==1:x=x[...,::2]
                    if index==2:x=torch.ones_like(x)
                    if index==3:x[...,0,0,0]=float('nan')
                    samples=torch.rand((2 if x.ndim==5 else 1,3,3),generator=g).to(device='npu:0',dtype=dtype)
                    return dict(input=x,kernel_size=[2,2,2],output_size=[1,2,2] if index==4 else [3,4,2],return_indices=index!=5,_random_samples=samples)
                one(str(dtype)+'-'+str(index),factory,lambda a:F.fractional_max_pool3d(**a),dict(atol=0.,rtol=0.),'supplementary')
    result=dict(operator=op,status='passed' if all(r['status']=='passed' for r in rows) else 'failed',
        total=len(rows),passed=sum(r['status']=='passed' for r in rows),cases=rows,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        physical_device=7,contract_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in leaf.iterdir() if p.is_file()},
        scope='独立CPU数值参考；全部正式正确性和性能输入各两次、换stream；不作性能计分或原生NPU参考证明')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','contract_sha256']},ensure_ascii=False),flush=True)
    return int(result['status']!='passed')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--operator',required=True)
    p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();raise SystemExit(check(a.operator,a.source,a.output))
