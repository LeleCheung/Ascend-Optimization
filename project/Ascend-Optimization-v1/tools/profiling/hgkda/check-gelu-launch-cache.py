#!/usr/bin/env python3
"""验证GELU先缓存对齐输入、再执行有存储偏移输入时的正确性与stream。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode=True


def check(source,output):
    if output.exists():raise FileExistsError(output)
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7'
    import torch
    import torch_npu
    import torch.nn.functional as F
    assert torch.npu.device_count()==1
    spec=importlib.util.spec_from_file_location('gelu_launch_identity',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[]
    for dtype in [torch.float16,torch.bfloat16,torch.float32]:
        for shape in [(2,128),(4,128),(2,66)]:
            row=dict(dtype=str(dtype),shape=shape,status='failed',storage_offset=1)
            try:
                values=torch.randn(shape,generator=torch.Generator().manual_seed(90371)).to(dtype=dtype,device='npu:0')
                module.run(values);torch.npu.synchronize()
                base=torch.full((values.numel()+2,),17.,dtype=dtype,device='npu:0')
                x=base[1:-1].view(shape);x.copy_(values);before=base.clone()
                d=shape[-1]//2
                expected=(F.gelu(x[:,:d].float(),approximate='none')*x[:,d:].float()).to(dtype)
                row['input_alignment_mod16']=x.data_ptr()&15
                tol=(1e-4,1e-4) if dtype==torch.float32 else (.015,.015)
                for repetition in range(2):
                    torch.npu.synchronize()
                    with torch.npu.stream(torch.npu.Stream() if repetition else torch.npu.default_stream()):
                        actual=module.run(x)
                    torch.npu.synchronize()
                    assert not torch._C._is_alias_of(actual,x)
                    torch.testing.assert_close(actual,expected,atol=tol[0],rtol=tol[1])
                    torch.testing.assert_close(base,before,atol=0.,rtol=0.)
                row['status']='passed'
            except Exception as error:row['error']=type(error).__name__+': '+str(error)
            rows.append(row)
    passed=sum(x['status']=='passed' for x in rows)
    result=dict(status='passed' if passed==len(rows) else 'failed',passed=passed,total=len(rows),cases=rows,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='对齐输入预热后切换相同几何的存储偏移输入，检查全部输出、输入保持与当前stream；不计性能')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},ensure_ascii=False))
    return passed==len(rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();sys.exit(0 if check(a.source,a.output) else 1)
