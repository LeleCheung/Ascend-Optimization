#!/usr/bin/env python3
"""检查不同目标间的大数累加是否吞掉小结果；CPU模式仅作算法反例。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode=True


def check(source,output,cpu_model=False):
    import torch
    if cpu_model:
        device='cpu';module=None
    else:
        if os.environ.get('ASCEND_RT_VISIBLE_DEVICES')!='7':raise ValueError('只允许物理卡7')
        import torch_npu
        assert torch.npu.device_count()==1
        device='npu:0'
        spec=importlib.util.spec_from_file_location('scatter_range_candidate',source)
        module=importlib.util.module_from_spec(spec);module.__package__='flag_gems.ops'
        spec.loader.exec_module(module)
    rows=[]
    for dtype in [torch.float16,torch.bfloat16,torch.float32]:
        for width in [257,1024,4096]:
            for sign in [1,-1]:
                row=dict(dtype=str(dtype),width=width,sign=sign,status='failed')
                try:
                    for repetition in range(1 if cpu_model else 2):
                        inp=torch.zeros((2,width),dtype=dtype,device=device)
                        src=torch.zeros_like(inp)
                        index=torch.zeros((2,width),dtype=torch.int64,device=device)
                        if dtype==torch.float16:
                            src[:,:-1]=sign*65504.
                        else:src[:,0]=sign*float(2**27)
                        src[:,-1]=.25;index[:,-1]=1
                        saved=(inp.clone(),src.clone(),index.clone())
                        expected=torch.scatter_add(inp,-1,index,src)
                        if cpu_model:
                            # 排序后所有target0在前，唯一target1在后。
                            prefix=src.float().cumsum(-1)
                            got=prefix[:,-1]-prefix[:,-2]
                            row.update(expected_small=expected[:,1].tolist(),prefix_difference=got.tolist())
                            assert torch.equal(got,expected[:,1].float()),'无关目标的大前缀吞掉了target1的0.25'
                        else:
                            torch.npu.synchronize()
                            stream=torch.npu.Stream() if repetition else torch.npu.default_stream()
                            with torch.npu.stream(stream):actual=module.run(inp,-1,index,src)
                            torch.npu.synchronize()
                            row.update(expected_small=expected[:,1].cpu().tolist(),actual_small=actual[:,1].cpu().tolist())
                            tol=1e-4 if dtype==torch.float32 else .015
                            torch.testing.assert_close(actual,expected,atol=tol,rtol=tol,equal_nan=True)
                            assert not torch._C._is_alias_of(actual,inp)
                            for value,before in zip((inp,src,index),saved):
                                torch.testing.assert_close(value,before,atol=0.,rtol=0.,equal_nan=True)
                    row.update(status='passed',repetitions=1 if cpu_model else 2,stream_switch=not cpu_model)
                except Exception as error:row['error']=type(error).__name__+': '+str(error)
                rows.append(row)
    result=dict(operator='scatter_add',status='passed' if all(r['status']=='passed' for r in rows) else 'failed',
                total=len(rows),passed=sum(r['status']=='passed' for r in rows),cases=rows,
                source_sha256=None if cpu_model else hashlib.sha256(source.read_bytes()).hexdigest(),
                checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                physical_device=None if cpu_model else 7,
                scope='CPU前缀差算法反例，不是候选NPU实测' if cpu_model else 'NPU不同目标间的大数/小数独立累加，含正负大数与换stream')
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},ensure_ascii=False))
    return int(result['status']!='passed' and not cpu_model)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpu-model',action='store_true')
    args=parser.parse_args()
    if not args.cpu_model and not args.source:parser.error('NPU模式需要--source')
    raise SystemExit(check(args.source,args.output,args.cpu_model))
