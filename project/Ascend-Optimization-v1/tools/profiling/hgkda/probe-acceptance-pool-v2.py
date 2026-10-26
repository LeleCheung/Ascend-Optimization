#!/usr/bin/env python3
"""验证独立纯NPU池化分解的数值、索引和设备路径，不替代默认参考。"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys
sys.dont_write_bytecode=True

TOOLS=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/tools/profiling/hgkda')
PROJECT=TOOLS.parents[2]
BATCH=PROJECT/'operators/campaigns/acceptance-three-20261026'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    import torch
    import torch_npu
    import torch.nn.functional as F
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7' and torch.npu.device_count()==1
    out=BATCH/'research/pool-decomposition-v2'
    out.mkdir(parents=True,exist_ok=False)
    ref=load('npu_decomp',TOOLS/'npu-reference-decompositions-v2.py')
    common=load('acceptance_common',TOOLS/'probe-acceptance-three.py')
    parent=PROJECT/'operators/campaigns/remaining-five-20261023-v2'
    oracle=load('pool_oracle',parent/'catalog/ops/fractional_max_pool3d/oracle.py')
    candidate=load('pool_candidate',parent/'deliverables/reviewed/fractional_max_pool3d/main.py')
    leaf=parent/'catalog/ops/fractional_max_pool3d'
    rows=[]
    def one(name,args,profiling=False):
        row=dict(case=name,status='failed')
        try:
            cpu={k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in args.items()}
            expected=F.fractional_max_pool3d(**cpu)
            got=ref.fractional_pool(**args)
            torch.npu.synchronize()
            for actual,want in zip(got,expected):
                torch.testing.assert_close(actual.cpu(),want,atol=0,rtol=0,equal_nan=True)
            row.update(status='passed',npu_decomposition=common.latency(lambda:ref.fractional_pool(**args),5),
                current=common.latency(lambda:candidate.run(**args),5))
            if profiling:
                row['default_profile']=common.profile(name+'_default',lambda:oracle.run(**args),out)
                row['npu_profile']=common.profile(name+'_npu',lambda:ref.fractional_pool(**args),out)
        except Exception as error:
            row['error']=type(error).__name__+': '+str(error)[-3500:]
        rows.append(row)
        common.save(out/'results.json',dict(scope='自建纯NPU分解补充参考；原PyTorch参考不变',rows=rows,
            reference_sha256=hashlib.sha256((TOOLS/'npu-reference-decompositions-v2.py').read_bytes()).hexdigest()))
        print(json.dumps({k:v for k,v in row.items() if 'profile' not in k}),flush=True)
    for phase in ['correctness','timing']:
        cases=[json.loads(l) for l in (leaf/(phase+'.jsonl')).read_text().splitlines()]
        for index,case in enumerate(cases):
            one(case['name'],oracle.gen_inputs(case,'npu:0'),phase=='timing' and index==0)
    for dtype in [torch.float16,torch.bfloat16,torch.float32]:
        for pattern in ['ties','nan','infinity','strided']:
            x=torch.randn((2,3,7,9,11),dtype=dtype).to('npu:0')
            if pattern=='ties':x.zero_()
            elif pattern=='nan':x.fill_(float('nan'))
            elif pattern=='infinity':x.fill_(-float('inf'))
            elif pattern=='strided':x=x.transpose(-1,-2)
            samples=torch.rand((2,3,6),dtype=dtype).to('npu:0')[...,::2]
            one(str(dtype)+'-'+pattern,dict(input=x,kernel_size=[2,2,2],output_size=[3,4,2],return_indices=True,_random_samples=samples))
    assert all(row['status']=='passed' for row in rows),'纯NPU分解未通过全部语义'


if __name__=='__main__':
    main()
