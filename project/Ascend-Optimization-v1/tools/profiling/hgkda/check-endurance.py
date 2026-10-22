#!/usr/bin/env python3
"""非计分的夜间语义门禁：新形状、布局、参数、特殊值与stream。"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode=True


def cumulative_timing_contract(source):
    """从候选所属批次读取冻结合同，避免仅小形状正确而计时分支未验数值。"""
    for batch in source.resolve().parents:
        plan=batch/'plan.json'
        if not plan.is_file():continue
        records=json.loads(plan.read_text(encoding='utf-8')).get('records',[])
        record=next((r for r in records if r.get('operator')=='cumulative_trapezoid'),None)
        if record is None:continue
        leaf=batch/record['catalog_leaf']
        for name in ['oracle.py','timing.jsonl']:
            if hashlib.sha256((leaf/name).read_bytes()).hexdigest()!=record['contract_sha256'][name]:
                raise ValueError('累计积分性能输入合同身份改变: '+name)
        rows=[json.loads(line) for line in (leaf/'timing.jsonl').read_text().splitlines() if line.strip()]
        if len(rows)!=record['timing_cases']:raise ValueError('累计积分性能输入不完整')
        return leaf,rows
    raise ValueError('找不到候选所属批次的冻结累计积分合同')


def check(op,source,output):
    if os.environ.get('ASCEND_RT_VISIBLE_DEVICES')!='7':raise ValueError('只允许物理卡7')
    import torch
    import torch_npu
    import torch.nn.functional as F
    assert torch.npu.device_count()==1
    s=importlib.util.spec_from_file_location('endurance_candidate',source)
    m=importlib.util.module_from_spec(s)
    if op=='scatter_add':m.__package__='flag_gems.ops'
    s.loader.exec_module(m)
    records=[]
    for dtype in [torch.float16,torch.bfloat16,torch.float32]:
        for case in range(6):
            row=dict(dtype=str(dtype),case=case,status='failed')
            try:
                generator=torch.Generator().manual_seed(73137+case)
                def tensor(shape):return torch.randn(shape,generator=generator).to(device='npu:0',dtype=dtype)
                def stride(x):
                    base=torch.full((*x.shape,2),17.,device=x.device,dtype=x.dtype)
                    base[...,0]=x
                    return base[...,0]
                tol=(1e-4,1e-4) if dtype==torch.float32 else (.015,.015)
                stream=torch.npu.Stream()
                if op=='rad2deg_' and case==0:
                    # 裸启动缓存不得沿用第一次整除输入的n%16假设。先以整除长度
                    # 填满三个分块档位，再执行后续奇数长度，不只验证"先小后大"顺序。
                    for warm_n in [2048,65536,1048576]:
                        warm=tensor((warm_n,));want=torch.rad2deg(warm)
                        got=m.run(warm);torch.npu.synchronize()
                        assert got is warm
                        torch.testing.assert_close(got,want,atol=tol[0],rtol=tol[1],equal_nan=True)
                    row['cache_primed_with_divisible_lengths']=[2048,65536,1048576]
                for repetition in range(2):
                    # 所有输入在默认stream建立；同步后再交给另一个stream，避免检查器自己的竞争。
                    if op=='rad2deg_':
                        x=tensor([(0,1,1031,2050,35,11)[case]])
                        if case==4:x=stride(x.view(5,7))
                        if case==5:x=torch.tensor([float('nan'),float('inf'),-float('inf'),0.,-0.,1.,-1.,.001,100.,-100.,1e4],dtype=dtype,device='npu:0')
                        before=x.clone();expected=torch.rad2deg(x)
                        args=(x,)
                    elif op=='cumulative_trapezoid':
                        x=tensor([(3,1),(3,2),(5,37),(7,19),(4,4097),(3,5,17)][case])
                        if case==3:x=x.t()
                        if case==5:x=stride(x)
                        dim=(-1,0,-1,0,-1,1)[case];dx=(1.,.25,-.5,1.3,.125,-2.)[case]
                        before=x.clone();expected=torch.cumulative_trapezoid(x,dx=dx,dim=dim);args=(x,dx,dim)
                    elif op=='scatter_add':
                        shape=[5,7,9] if case<4 else [17,131]
                        dim=(case%len(shape))-len(shape)
                        x=tensor(shape);src=tensor(shape)
                        index=torch.randint(shape[dim],shape,generator=generator).to('npu:0')
                        if case==2:index.zero_()
                        if case==3:x=stride(x);src=stride(src);index=stride(index)
                        if case==5:
                            src=src[:,:63];index=index[:,:63]
                        before=x.clone();saved_src=src.clone();saved_index=index.clone()
                        expected=torch.scatter_add(x,dim,index,src);args=(x,dim,index,src)
                    elif op=='uniform':
                        x=tensor((65537,) if case<4 else (127,513))
                        if case==5:x=stride(x)
                        low,high=[(0.,1.),(-3.,2.),(.5,.5),(-.01,.02),(-100.,-10.),(.1,.9)][case]
                        before=x.clone();expected=None;args=(x,low,high)
                    elif op=='gelu_and_mul':
                        shape=[(1,2),(3,34),(7,2050),(2,8194),(3,130),(2,66)][case]
                        x=tensor(shape)
                        if case==4:x=stride(x)
                        if case==5:
                            values=torch.linspace(-12,12,shape[-1],device='npu:0',dtype=dtype)
                            x[:]=values
                        before=x.clone();d=shape[-1]//2
                        expected=(F.gelu(x[...,:d].float(),approximate='none')*x[...,d:].float()).to(dtype);args=(x,)
                    else:raise ValueError(op)
                    torch.npu.synchronize()
                    with torch.npu.stream(stream if repetition else torch.npu.default_stream()):
                        actual=m.run(args[0],dx=args[1],dim=args[2]) if op=='cumulative_trapezoid' else m.run(*args)
                    torch.npu.synchronize()
                    if op=='rad2deg_':assert actual is x and actual.stride()==x.stride()
                    else:
                        assert not torch._C._is_alias_of(actual,x)
                        torch.testing.assert_close(x,before,atol=0.,rtol=0.,equal_nan=True)
                    if op=='uniform':
                        assert actual.shape==x.shape and actual.dtype==dtype and actual.device==x.device
                        assert torch.isfinite(actual).all()
                        if low==high:assert (actual==low).all()
                        else:
                            assert ((actual>=low)&(actual<high)).all()
                            z=(actual.float()-low)/(high-low)
                            assert abs(z.mean().item()-.5)<.025
                            assert abs(z.var(unbiased=False).item()-1/12)<.012
                            nxt=m.run(*args);torch.npu.synchronize()
                            assert not torch.equal(nxt,actual),'连续两次调用复用了随机输出'
                    else:
                        torch.testing.assert_close(actual,expected,atol=tol[0],rtol=tol[1],check_stride=False,equal_nan=True)
                    if op=='scatter_add':
                        torch.testing.assert_close(src,saved_src,atol=0.,rtol=0.)
                        assert torch.equal(index,saved_index)
                row.update(status='passed',repetitions=2,stream_switch_checked=True)
            except Exception as error:row['error']=type(error).__name__+': '+str(error)
            records.append(row)
    timing_checks=[]
    timing_contract={}
    if op=='cumulative_trapezoid':
        leaf,rows=cumulative_timing_contract(source)
        spec=importlib.util.spec_from_file_location('endurance_timing_oracle',leaf/'oracle.py')
        oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
        timing_contract={name:hashlib.sha256((leaf/name).read_bytes()).hexdigest() for name in ['oracle.py','timing.jsonl']}
        for case in rows:
            row=dict(case=case['name'],status='failed',spec=case['inputs']['spec'])
            try:
                for repetition in range(2):
                    args=oracle.gen_inputs(case,'npu:0');before=args['y'].clone()
                    expected=oracle.run(**args);torch.npu.synchronize()
                    stream=torch.npu.Stream()
                    with torch.npu.stream(stream if repetition else torch.npu.default_stream()):
                        actual=m.run(**args)
                    torch.npu.synchronize()
                    assert not torch._C._is_alias_of(actual,args['y'])
                    torch.testing.assert_close(args['y'],before,atol=0.,rtol=0.,equal_nan=True)
                    tol=case['tolerance']
                    torch.testing.assert_close(actual,expected,atol=tol['atol'],rtol=tol['rtol'],check_stride=False,equal_nan=True)
                row.update(status='passed',repetitions=2,stream_switch_checked=True)
            except Exception as error:row['error']=type(error).__name__+': '+str(error)
            timing_checks.append(row)
    result=dict(operator=op,status='passed' if all(r['status']=='passed' for r in records+timing_checks) else 'failed',
        total=len(records),passed=sum(r['status']=='passed' for r in records),cases=records,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),physical_device=7,
        timing_input_correctness=dict(total=len(timing_checks),passed=sum(r['status']=='passed' for r in timing_checks),
            cases=timing_checks,contract_sha256=timing_contract),
        scope='独立非计分语义；18项各两次，含当前stream检查；累计积分另核对全部冻结性能输入的数值，不替代完整上游pytest')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','timing_input_correctness']},ensure_ascii=False),flush=True)
    return int(result['status']!='passed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operator',required=True,choices=['rad2deg_','cumulative_trapezoid','scatter_add','uniform','gelu_and_mul'])
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();raise SystemExit(check(a.operator,a.source,a.output))
