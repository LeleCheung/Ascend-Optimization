#!/usr/bin/env python3
"""补查连续大归约维的跨列块写入与非2次幂尾部，不参与性能计分。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

if os.environ.get('ASCEND_RT_VISIBLE_DEVICES','7')!='7':raise RuntimeError('仅允许物理卡7')
os.environ['ASCEND_RT_VISIBLE_DEVICES']='7'
import torch
import torch_npu


def check(source,output):
    assert torch.npu.device_count()==1
    spec=importlib.util.spec_from_file_location('dense_candidate',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    run=getattr(module,'value_selecting_reduction_backward',None) or module.run
    cases=[]
    for dtype in [torch.float16,torch.bfloat16,torch.float32]:
        for sizes in [[1,8209],[3,16387],[2,7,8209]]:
            for keep in [False,True]:
                shape=sizes[:-1]+([1] if keep else [])
                generator=torch.Generator().manual_seed(61021)
                grad=torch.randn(shape,generator=generator).to(dtype=dtype,device='npu:0')
                boundaries=[0,8191,8192,sizes[-1]-1]
                values=[boundaries[i%len(boundaries)] for i in range(grad.numel())]
                if grad.numel()==1:values=[sizes[-1]-1]
                indices=torch.tensor(values,dtype=torch.int64,device='npu:0').reshape(shape)
                saved_grad=grad.clone();saved_indices=indices.clone()
                record={'dtype':str(dtype),'sizes':sizes,'keepdim':keep,'repeats':5,'status':'failed'}
                try:
                    expected=torch.ops.aten.value_selecting_reduction_backward(grad,-1,indices,sizes,keep)
                    for _ in range(5):
                        actual=run(grad,-1,indices,sizes,keep)
                        torch.testing.assert_close(actual,expected,rtol=0.,atol=0.,check_stride=False)
                        assert not torch._C._is_alias_of(actual,grad)
                    torch.testing.assert_close(grad,saved_grad,rtol=0.,atol=0.)
                    torch.testing.assert_close(indices,saved_indices,rtol=0.,atol=0.)
                    torch.npu.synchronize();record['status']='passed'
                except Exception as error:record['error']=str(error)
                cases.append(record)
    value={'operator':'value_selecting_reduction_backward','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
           'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'physical_device':7,'logical_device':'npu:0','total':len(cases),
           'passed':sum(r['status']=='passed' for r in cases),'cases':cases,
           'scope':'连续布局，跨8192列边界和非2次幂尾部；每项重复5次，不参与性能计分'}
    value['status']='passed' if value['passed']==value['total'] else 'failed'
    if output.exists():raise RuntimeError('保留已有证据，使用新的尝试目录')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode())
    print(json.dumps({k:v for k,v in value.items() if k!='cases'},ensure_ascii=False),flush=True)
    if value['status']!='passed':raise RuntimeError('跨列块补测失败')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();check(args.source,args.output)
