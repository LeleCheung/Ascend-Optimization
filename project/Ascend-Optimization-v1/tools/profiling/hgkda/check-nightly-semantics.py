#!/usr/bin/env python3
"""独立检查两个重点算子的未见形状、stride与参数语义，不参与性能计分。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

# 此补测仅用于本项目获准的物理卡7，必须在torch_npu初始化前限定可见设备。
if os.environ.get('ASCEND_RT_VISIBLE_DEVICES','7')!='7':
    raise RuntimeError('补测仅允许物理卡7')
os.environ['ASCEND_RT_VISIBLE_DEVICES']='7'
import torch
import torch_npu


def check(operator,source,output):
    assert torch.npu.device_count()==1,'物理卡7应映射为唯一逻辑卡0'
    spec=importlib.util.spec_from_file_location('candidate',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    run=getattr(module,operator,None) or module.run
    records=[]
    for dtype in [torch.float16,torch.bfloat16,torch.float32]:
        count=16 if operator=='value_selecting_reduction_backward' else 12
        for i in range(count):
            generator=torch.Generator().manual_seed(91813+i)
            def tensor(shape):return torch.randn(shape,generator=generator).to(dtype=dtype,device='npu:0')
            def strided(value):
                base=torch.full((*value.shape,2),17.,dtype=value.dtype,device=value.device)
                base[...,0]=value
                return base[...,0],base
            def permuted(value):
                # 反向物理布局再转回相同逻辑shape，产生不能简单合并维度的stride。
                axes=tuple(reversed(range(value.ndim)))
                base=torch.empty(tuple(reversed(value.shape)),dtype=value.dtype,device=value.device)
                view=base.permute(axes);view.copy_(value)
                return view
            record={'dtype':str(dtype),'case':i,'status':'failed'}
            try:
                if operator=='value_selecting_reduction_backward':
                    sizes=[[11],[5,7],[3,7,11],[2,3,5,7]][i%4]
                    axis=i%len(sizes);dim=axis-len(sizes) if i%2 else axis;keep=bool(i%3)
                    if i>=12:
                        # 检查原生评审指出的未覆盖分支，不改变正式计分合同。
                        sizes,dim,keep=[([17],0,False),([0,7],1,True),
                                        ([2,1,3],1,False),([2,8209],-1,False)][i-12]
                        axis=dim%len(sizes)
                    shape=list(sizes)
                    if keep:shape[axis]=1
                    else:shape.pop(axis)
                    grad=tensor(shape)
                    indices=torch.randint(sizes[axis],shape,generator=generator).to('npu:0')
                    guard=None
                    if i%2:
                        grad,guard=strided(grad)
                        base=torch.zeros((*shape,2),dtype=torch.int64,device='npu:0')
                        base[...,0]=indices;indices=base[...,0]
                    layout='stride2' if i%2 else 'contiguous'
                    if i%3==2 and len(shape)>=2:
                        grad=permuted(grad);indices=permuted(indices);guard=None;layout='permuted'
                    saved_grad=grad.clone();saved_indices=indices.clone()
                    record.update(sizes=sizes,dim=dim,keepdim=keep,layout=layout,grad_stride=list(grad.stride()),indices_stride=list(indices.stride()))
                    expected=torch.ops.aten.value_selecting_reduction_backward(grad,dim,indices,sizes,keep)
                    actual=run(grad,dim,indices,sizes,keep)
                    # 此算子只复制梯度和补零，应逐元素精确一致。
                    torch.testing.assert_close(actual,expected,atol=0.,rtol=0.,check_stride=False)
                    assert not torch._C._is_alias_of(actual,grad)
                    torch.testing.assert_close(grad,saved_grad,atol=0.,rtol=0.)
                    torch.testing.assert_close(indices,saved_indices,atol=0.,rtol=0.)
                    if guard is not None:assert bool((guard[...,1]==17.).all())
                else:
                    shape=([1,7,65,129][i%4],[1,2,3,5,8,16][i%6],[1,31,257,1025][(i//3)%4])
                    scale=[0.,-1.25,.37,1.414][i%4]
                    value=tensor(shape);guard=None
                    if i%2:value,guard=strided(value)
                    layout='stride2' if i%2 else 'contiguous'
                    if i%4==2:value=permuted(value);guard=None;layout='permuted'
                    before=value.clone()
                    record.update(shape=list(shape),scale=scale,layout=layout,input_stride=list(value.stride()))
                    expected=value.float().sum(dim=1).mul(scale).to(dtype)
                    actual=run(value,scale)
                    assert actual.shape==expected.shape and actual.dtype==expected.dtype
                    assert not torch._C._is_alias_of(actual,value)
                    atol,rtol=(1e-4,1e-4) if dtype==torch.float32 else (.015,.015)
                    torch.testing.assert_close(actual,expected,atol=atol,rtol=rtol)
                    torch.testing.assert_close(value,before,atol=0.,rtol=0.)
                    if guard is not None:assert bool((guard[...,1]==17.).all())
                torch.npu.synchronize();record['status']='passed'
            except Exception as error:
                record['error']=str(error)
            records.append(record)
    result={'operator':operator,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'physical_device':7,'logical_device':'npu:0',
            'status':'passed' if all(r['status']=='passed' for r in records) else 'failed',
            'passed':sum(r['status']=='passed' for r in records),'total':len(records),
            'scope':'独立未见形状/seed/stride及参数检查；不修改KGS合同、不参与加速比计分','cases':records}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},ensure_ascii=False),flush=True)
    if result['status']!='passed':raise RuntimeError('补充语义检查失败')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operator',choices=['value_selecting_reduction_backward','moe_sum_reduce'],required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();check(args.operator,args.source,args.output)
