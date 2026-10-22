#!/usr/bin/env python3
"""同一Native输入的参考/候选真实设备任务采集与实际IR；不作正式计分。"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

# Native目录严格限制文件集合，导入oracle不能在冻结合同中制造__pycache__。
sys.dont_write_bytecode=True


def load(name,path,package=None):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s)
    if package:m.__package__=package
    s.loader.exec_module(m);return m


def observe(op,source,leaf,output,index):
    if os.environ.get('ASCEND_RT_VISIBLE_DEVICES')!='7':raise ValueError('只允许物理卡7')
    import torch
    import torch_npu
    assert torch.npu.device_count()==1
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    oracle=load('endurance_oracle',leaf/'oracle.py')
    candidate=load('endurance_source',source,'flag_gems.ops' if op=='scatter_add' else None)
    rows=[json.loads(l) for l in (leaf/'timing.jsonl').read_text().splitlines() if l.strip()]
    case=rows[index]
    def inputs():
        if hasattr(oracle,'gen_inputs'):return oracle.gen_inputs(case,'npu:0')
        g=torch.Generator().manual_seed(case.get('seed',0));result={}
        for k,v in case['inputs'].items():
            if isinstance(v,dict) and v.get('type')=='random':
                result[k]=torch.randn(v['shape'],generator=g).to(device='npu:0',dtype=getattr(torch,v['dtype']))
            else:raise ValueError('未实现此Native输入，请保留失败而不猜测: '+k)
        return result
    facts=dict(operator=op,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),case=case,
        physical_device=7,scope='同输入真实执行路径；独立采集时间不替代KGS完整调用成绩',sides={})
    for side,m in [('reference',oracle),('candidate',candidate)]:
        warm=inputs();m.run(**warm);torch.npu.synchronize()
        value=inputs();torch.npu.synchronize()
        directory=output/side;directory.mkdir()
        try:
            with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True,with_stack=False,
                    schedule=torch_npu.profiler.schedule(wait=0,warmup=0,active=1,repeat=1),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(directory/'raw'))) as prof:
                with torch.profiler.record_function('HGKDA_TARGET_'+op):
                    m.run(**value);torch.npu.synchronize()
                prof.step()
            trace=directory/'trace.json';prof.export_chrome_trace(str(trace))
            data=json.loads(trace.read_text());events=data if isinstance(data,list) else data.get('traceEvents',[])
            # 保存全部设备事件；kernel命名依赖本机版本，不凭名称猜测DMA或计算路径。
            device=[{k:e[k] for k in ['name','cat','dur','ts','pid','tid','args'] if k in e}
                    for e in events if e.get('ph')=='X' and
                    ('kernel' in str(e.get('cat','')).lower() or 'memcpy' in str(e.get('name','')).lower()
                     or 'Task Type' in e.get('args',{}) or 'hcc' in str(e.get('cat','')).lower())]
            facts['sides'][side]=dict(status='completed',trace=str(trace),
                trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),device_events=device,
                note='含调用内同步；原始trace保留完整CPU和设备事件；DMA判断须核对原始设备任务')
        except Exception as e:
            facts['sides'][side]=dict(status='failed',error=type(e).__name__+': '+str(e))
    ir=output/'ir';ir.mkdir();entries=[]
    for name,obj in vars(candidate).items():
        caches=getattr(obj,'device_caches',None)
        if not isinstance(caches,dict):continue
        for device,cache_tuple in caches.items():
            for compiled in cache_tuple[0].values():
                if hasattr(compiled,'result'):compiled=compiled.result()
                artifacts=[]
                for kind,content in compiled.asm.items():
                    if not isinstance(content,(str,bytes)):continue
                    data=content.encode() if isinstance(content,str) else content
                    path=ir/(name+'-'+str(len(entries))+'.'+kind);path.write_bytes(data)
                    artifacts.append(dict(kind=kind,path=str(path),sha256=hashlib.sha256(data).hexdigest()))
                entries.append(dict(symbol=name,device=device,metadata=str(compiled.metadata),artifacts=artifacts))
    facts['ir']=entries
    facts['status']='completed' if all(v['status']=='completed' for v in facts['sides'].values()) else 'partial'
    (output/'facts.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(operator=op,status=facts['status'],device_events={k:len(v.get('device_events',[])) for k,v in facts['sides'].items()},ir=len(entries))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operator',required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--leaf',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--index',type=int,default=0)
    a=p.parse_args();observe(a.operator,a.source,a.leaf,a.output,a.index)
