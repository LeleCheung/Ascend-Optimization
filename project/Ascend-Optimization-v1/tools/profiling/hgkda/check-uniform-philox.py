#!/usr/bin/env python3
"""独立核验手写Philox的逐位结果、64位参数传递与counter跨界；不计性能。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
MASK = (1 << 32)-1


def reference(seed, counter):
    c0,c1,c2,c3 = counter&MASK,(counter>>32)&MASK,0,0
    k0,k1 = seed&MASK,(seed>>32)&MASK
    for _ in range(10):
        a,b = c0*0xD2511F53,c2*0xCD9E8D57
        c0,c1,c2,c3 = ((b>>32)^c1^k0)&MASK,b&MASK,((a>>32)^c3^k1)&MASK,a&MASK
        k0,k1 = (k0+0x9E3779B9)&MASK,(k1+0xBB67AE85)&MASK
    return c0,c1,c2,c3


def check(source, output):
    if output.exists():
        raise FileExistsError(output)
    if os.environ.get('ASCEND_RT_VISIBLE_DEVICES') != '7':
        raise RuntimeError('必须仅使用物理卡7')
    import torch
    import torch_npu
    import triton
    import triton.language as tl
    if torch.npu.device_count() != 1:
        raise RuntimeError('可见设备数必须为1')
    spec = importlib.util.spec_from_file_location('uniform_philox_under_test',source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module,'_philox4'):
        raise ValueError('待测源码没有_philox4；不猜测其他随机算法')
    # JIT解析本文件全局符号，避免把Python模块对象当成constexpr传入kernel。
    global CANDIDATE_PHILOX
    CANDIDATE_PHILOX = module._philox4
    globals().update(tl=tl,triton=triton)

    @triton.jit(do_not_specialize=['seed','offset'])
    def raw(out, echo, state, seed, offset, FROM_TENSOR: tl.constexpr, BUILTIN: tl.constexpr, BLOCK: tl.constexpr):
        if FROM_TENSOR:
            s = tl.load(state).to(tl.uint64)
            o = tl.load(state+1).to(tl.uint64)
        else:
            s = seed.to(tl.uint64)
            o = offset.to(tl.uint64)
        lane = tl.arange(0,BLOCK)
        counter = o+lane.to(tl.uint64)
        c0 = counter.to(tl.uint32)
        c1 = (counter>>32).to(tl.uint32)
        if BUILTIN:
            z = c0*0
            r0,r1,r2,r3 = tl.philox(s,c0,c1,z,z)
        else:
            r0,r1,r2,r3 = CANDIDATE_PHILOX(s,c0,c1)
        tl.store(out+lane,r0.to(tl.int32,bitcast=True))
        tl.store(out+BLOCK+lane,r1.to(tl.int32,bitcast=True))
        tl.store(out+2*BLOCK+lane,r2.to(tl.int32,bitcast=True))
        tl.store(out+3*BLOCK+lane,r3.to(tl.int32,bitcast=True))
        tl.store(echo,s.to(tl.uint32).to(tl.int64))
        tl.store(echo+1,(s>>32).to(tl.uint32).to(tl.int64))
        tl.store(echo+2,o.to(tl.uint32).to(tl.int64))
        tl.store(echo+3,(o>>32).to(tl.uint32).to(tl.int64))

    tests = [(0,0),(0,1),(7,0),(1<<31,0),(1<<32,0),(1<<40,1<<40),
             (0x123456789ABCDEF0,1<<32),(0x7123456789ABCDEF,(1<<32)-16)]
    rows = []
    block = 64
    for seed,offset in tests:
        matrix = [reference(seed,offset+lane) for lane in range(block)]
        expected = [matrix[lane][word] for word in range(4) for lane in range(block)]
        echo_expected = [seed&MASK,seed>>32,offset&MASK,offset>>32]
        for from_tensor in [False,True]:
            for builtin in [False,True]:
                row = dict(seed=seed,offset=offset,from_tensor=from_tensor,builtin=builtin,status='failed')
                try:
                    state = torch.tensor([seed,offset],dtype=torch.int64,device='npu:0')
                    out = torch.empty(block*4,dtype=torch.int32,device='npu:0')
                    echo = torch.empty(4,dtype=torch.int64,device='npu:0')
                    raw[(1,)](out,echo,state,seed,offset,FROM_TENSOR=from_tensor,BUILTIN=builtin,BLOCK=block)
                    torch.npu.synchronize()
                    actual = [int(v)&MASK for v in out.cpu().tolist()]
                    row.update(echo=echo.cpu().tolist(),echo_expected=echo_expected,
                               mismatches=sum(a!=b for a,b in zip(actual,expected)),
                               actual_first8=actual[:8],expected_first8=expected[:8])
                    row['status'] = 'passed' if row['mismatches']==0 and row['echo']==echo_expected else 'failed'
                except Exception as error:
                    row['error'] = type(error).__name__+': '+str(error)
                rows.append(row)
    manual = [r for r in rows if not r['builtin']]
    passed = sum(r['status']=='passed' for r in manual)
    data = dict(status='passed' if passed==len(manual) else 'failed',passed=passed,total=len(manual),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=rows,
                scope='手写Philox与独立Python整数参考逐位比较；SoA布局一致；同时测动态标量、设备状态输入、高位seed/counter及counter低32位跨界。tl.philox仅作另列对照，不作为正确参考。')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in data.items() if k!='cases'},ensure_ascii=False))
    return data['status']=='passed'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    sys.exit(0 if check(a.source,a.output) else 1)
