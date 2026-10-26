#!/usr/bin/env python3
"""池化改为每program处理连续输出向量；保存原舍入与NaN/索引语义。"""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import os
import sys
sys.dont_write_bytecode=True

ROOT=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1')
BATCH=ROOT/'operators/campaigns/acceptance-three-20261026'
TOOLS=ROOT/'tools/profiling/hgkda'


KERNEL=r'''
@libentry()
@triton.jit
def _pool_flat_kernel(
    in_ptr,out_ptr,idx_ptr,samp_ptr,s_n,s_c,s_d,s_h,s_w,ss_n,ss_c,
    in_c,in_d,in_h,in_w,out_d,out_h,out_w,alpha_d,alpha_h,alpha_w,
    KD:tl.constexpr,KH:tl.constexpr,KW:tl.constexpr,
    BLOCK_H:tl.constexpr,BLOCK_W:tl.constexpr,WRITE_IDX:tl.constexpr):
    nc=tl.program_id(0)
    linear=tl.program_id(1)*BLOCK_W+tl.arange(0,BLOCK_W)
    mask=linear<out_d*out_h*out_w
    wo=linear%out_w
    ho=(linear//out_w)%out_h
    do=linear//(out_h*out_w)
    ni=nc//in_c
    ci=nc%in_c
    samp=samp_ptr+ni*ss_n+ci*ss_c
    sd=tl.load(samp).to(tl.float32)
    sh=tl.load(samp+1).to(tl.float32)
    sw=tl.load(samp+2).to(tl.float32)
    dt:tl.constexpr=samp_ptr.dtype.element_ty
    ad=_round_coordinate(_round_coordinate(tl.cast(in_d-KD,tl.float32),dt)/_round_coordinate(tl.cast(tl.maximum(out_d-1,1),tl.float32),dt),dt)
    ah=_round_coordinate(_round_coordinate(tl.cast(in_h-KH,tl.float32),dt)/_round_coordinate(tl.cast(tl.maximum(out_h-1,1),tl.float32),dt),dt)
    aw=_round_coordinate(_round_coordinate(tl.cast(in_w-KW,tl.float32),dt)/_round_coordinate(tl.cast(tl.maximum(out_w-1,1),tl.float32),dt),dt)
    ds=_round_coordinate(_round_coordinate(do.to(tl.float32)+sd,dt)*ad,dt).to(tl.int32)-_round_coordinate(sd*ad,dt).to(tl.int32)
    hs=_round_coordinate(_round_coordinate(ho.to(tl.float32)+sh,dt)*ah,dt).to(tl.int32)-_round_coordinate(sh*ah,dt).to(tl.int32)
    ws=_round_coordinate(_round_coordinate(wo.to(tl.float32)+sw,dt)*aw,dt).to(tl.int32)-_round_coordinate(sw*aw,dt).to(tl.int32)
    ds=tl.where(do==out_d-1,in_d-KD,ds)
    hs=tl.where(ho==out_h-1,in_h-KH,hs)
    ws=tl.where(wo==out_w-1,in_w-KW,ws)
    best=tl.full((BLOCK_W,),float('-inf'),tl.float32)
    index=ds*in_h*in_w+hs*in_w+ws
    base=in_ptr+ni*s_n+ci*s_c
    for di in tl.static_range(KD):
        for hi in tl.static_range(KH):
            for wi in tl.static_range(KW):
                d=ds+di
                h=hs+hi
                w=ws+wi
                valid=mask&(d>=0)&(d<in_d)&(h>=0)&(h<in_h)&(w>=0)&(w<in_w)
                value=tl.load(base+d*s_d+h*s_h+w*s_w,mask=valid,other=float('-inf')).to(tl.float32)
                take=(value>best)|(value!=value)
                best=tl.where(take,value,best)
                index=tl.where(take,d*in_h*in_w+h*in_w+w,index)
    offset=nc*out_d*out_h*out_w+linear
    tl.store(out_ptr+offset,best,mask)
    if WRITE_IDX:
        tl.store(idx_ptr+offset,index.to(tl.int64),mask)

_fractional_max_pool3d_general_kernel=_pool_flat_kernel
'''


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    import torch
    import torch_npu
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7' and torch.npu.device_count()==1
    out=BATCH/'research/pool-variants'
    out.mkdir(parents=True,exist_ok=False)
    parent=ROOT/'operators/campaigns/remaining-five-20261023-v2'
    source=parent/'deliverables/reviewed/fractional_max_pool3d/main.py'
    oracle=load('pool_oracle',parent/'catalog/ops/fractional_max_pool3d/oracle.py')
    common=load('pool_common',TOOLS/'probe-acceptance-three.py')
    cases=[json.loads(l) for l in (parent/'catalog/ops/fractional_max_pool3d/timing.jsonl').read_text().splitlines()]
    candidates=[('parent',source)]
    for block in [128,256,512,1024]:
        text=source.read_text()
        text=text.replace('    block_h = 4','    block_h = 1').replace('    block_w = 8','    block_w = '+str(block))
        text=text.replace('out_d * triton.cdiv(out_h, block_h) * triton.cdiv(out_w, block_w)',
            'triton.cdiv(out_d*out_h*out_w, block_w)')
        path=out/('flat-'+str(block)+'.py')
        path.write_text(text+KERNEL)
        candidates.append(('flat-'+str(block),path))
    rows=[]
    for label,path in candidates:
        row=dict(label=label,source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),status='passed',cases=[])
        try:
            module=load('pool_'+label.replace('-','_'),path)
            for case in cases:
                args=oracle.gen_inputs(case,'npu:0')
                expected=oracle.run(**args)
                actual=module.run(**args)
                torch.npu.synchronize()
                for got,want in zip(actual,expected):
                    torch.testing.assert_close(got.cpu(),want.cpu(),atol=0,rtol=0,equal_nan=True)
                row['cases'].append(dict(case=case['name'],**common.latency(lambda:module.run(**args),11)))
        except Exception as error:
            row.update(status='failed',error=type(error).__name__+': '+str(error)[-4500:])
        if row['status']=='passed':
            row['geometric_latency_ms']=math.exp(sum(math.log(x['median_ms']) for x in row['cases'])/len(cases))
        rows.append(row)
        common.save(out/'results.json',dict(scope='同计时输入诊断选版，不替代KGS完整评测',rows=rows))
        print(json.dumps(row),flush=True)
    valid=[r for r in rows if r['status']=='passed']
    common.save(out/'selection.json',dict(best=min(valid,key=lambda r:r['geometric_latency_ms']),all_valid=valid))


if __name__=='__main__':
    main()
