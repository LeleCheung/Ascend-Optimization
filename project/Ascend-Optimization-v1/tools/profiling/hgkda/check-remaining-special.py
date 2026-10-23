#!/usr/bin/env python3
"""补充池化极值/采样stride、上采样显式scale/布局；不改变正式合同。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True


def check(op,source,output):
    if output.exists():raise FileExistsError(output)
    if os.environ.get('ASCEND_RT_VISIBLE_DEVICES')!='7':raise ValueError('仅物理卡7')
    import torch
    import torch_npu
    import torch.nn.functional as F
    assert torch.npu.device_count()==1
    spec=importlib.util.spec_from_file_location('fractional_special_candidate',source)
    candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
    rows=[];stream=torch.npu.Stream()
    for dtype in [torch.float16,torch.bfloat16,torch.float32] if op=='fractional_max_pool3d' else []:
        for ndim in [4,5]:
            for pattern in ['negative_inf','mixed_inf','multiple_nan','ties','strided_samples']:
                row=dict(dtype=str(dtype),ndim=ndim,pattern=pattern,status='failed',repetitions=2,stream_switch=True)
                try:
                    for repeat in range(2):
                        generator=torch.Generator().manual_seed(61237+repeat)
                        x=torch.randn((2,3,7,9,11) if ndim==5 else (3,7,9,11),generator=generator).to(dtype)
                        if pattern=='negative_inf':x.fill_(-float('inf'))
                        elif pattern=='mixed_inf':x[...,0]=float('inf');x[...,1]=-float('inf')
                        elif pattern=='multiple_nan':x.fill_(float('nan'))
                        elif pattern=='ties':x.zero_()
                        samples=torch.rand((2 if ndim==5 else 1,3,6),generator=generator).to(dtype)
                        samples=samples[...,::2] if pattern=='strided_samples' else samples[...,:3].contiguous()
                        args=dict(input=x,kernel_size=[2,2,2],output_size=[3,4,2],return_indices=True,_random_samples=samples)
                        expected=F.fractional_max_pool3d(**args)
                        # 保留采样张量的非连续stride；整块搬到NPU后再切片。
                        if pattern=='strided_samples':
                            buffer=torch.zeros((samples.shape[0],3,6),dtype=dtype)
                            buffer[...,::2]=samples
                            device_samples=buffer.to('npu:0')[...,::2]
                        else:device_samples=samples.to('npu:0')
                        args.update(input=x.to('npu:0'),_random_samples=device_samples)
                        before=args['input'].cpu().clone();before_samples=device_samples.cpu().clone()
                        torch.npu.synchronize()
                        with torch.npu.stream(stream if repeat else torch.npu.default_stream()):actual=candidate.run(**args)
                        torch.npu.synchronize()
                        assert isinstance(actual,tuple) and len(actual)==2
                        for got,want in zip(actual,expected):
                            assert got.device.type=='npu'
                            torch.testing.assert_close(got.cpu(),want,atol=0,rtol=0,equal_nan=True)
                        torch.testing.assert_close(args['input'].cpu(),before,atol=0,rtol=0,equal_nan=True)
                        torch.testing.assert_close(device_samples.cpu(),before_samples,atol=0,rtol=0)
                    row['status']='passed'
                except Exception as error:row['error']=type(error).__name__+': '+str(error)[:2500]
                rows.append(row)
    if op=='upsample_bilinear2d':
        for dtype in [torch.float16,torch.bfloat16,torch.float32]:
            for case in range(12):
                row=dict(dtype=str(dtype),case=case,status='failed',repetitions=2,stream_switch=True)
                try:
                    for repeat in range(2):
                        generator=torch.Generator().manual_seed(99931+case+repeat)
                        x=torch.randn((2,3,5,9),generator=generator).to(dtype)
                        if case%3==0:x=x.transpose(2,3)
                        elif case%3==1:x=x.contiguous(memory_format=torch.channels_last)
                        else:x=x[...,1:8:2]
                        size=[x.shape[2]*2,x.shape[3]*2] if case<6 else [1,1] if case<9 else [7,11]
                        sh,sw=[(.75,1.25),(1.,1.),(2.,2.)][case%3]
                        align=bool(case%2)
                        expected=torch.ops.aten.upsample_bilinear2d(x,size,align,sh,sw)
                        if case%3==0:device=x.transpose(2,3).contiguous().to('npu:0').transpose(2,3)
                        elif case%3==1:device=x.to('npu:0').contiguous(memory_format=torch.channels_last)
                        else:
                            buffer=torch.zeros((2,3,5,9),dtype=dtype);buffer[...,1:8:2]=x
                            device=buffer.to('npu:0')[...,1:8:2]
                        before=device.cpu().clone();torch.npu.synchronize()
                        with torch.npu.stream(stream if repeat else torch.npu.default_stream()):
                            actual=candidate.run(device,output_size=size,align_corners=align,scales_h=sh,scales_w=sw)
                        torch.npu.synchronize()
                        assert actual.device==device.device
                        assert not torch._C._is_alias_of(actual,device)
                        tol=1e-4 if dtype==torch.float32 else .015
                        torch.testing.assert_close(actual.cpu(),expected,atol=tol,rtol=tol,equal_nan=True)
                        torch.testing.assert_close(device.cpu(),before,atol=0,rtol=0,equal_nan=True)
                    row['status']='passed'
                except Exception as error:row['error']=type(error).__name__+': '+str(error)[:2500]
                rows.append(row)
    data=dict(operator=op,status='passed' if all(r['status']=='passed' for r in rows) else 'failed',
        total=len(rows),passed=sum(r['status']=='passed' for r in rows),cases=rows,physical_device=7,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='独立CPU参考补充特殊值与采样stride；不计分，不替换正式原样参考')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in data.items() if k!='cases'},ensure_ascii=False))
    return int(data['status']!='passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operator',required=True,choices=['upsample_bilinear2d','fractional_max_pool3d'])
    parser.add_argument('--source',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();raise SystemExit(check(args.operator,args.source,args.output))
