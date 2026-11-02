"""独立诊断宿主调用与设备任务；不把两次采集相减，不用于KGS验收排名。"""
import argparse
from decimal import Decimal
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.dont_write_bytecode=True


def imported(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def summary(values):return dict(median=statistics.median(values),mean=statistics.mean(values),min=min(values),max=max(values),count=len(values))


def interval_union(tasks):
    intervals=sorted((Decimal(str(e['ts'])),Decimal(str(e['ts']))+Decimal(str(e['dur']))) for e in tasks)
    if not intervals:return 0.
    total=Decimal(0);start,end=intervals[0]
    for left,right in intervals[1:]:
        if left>end:total+=end-start;start,end=left,right
        else:end=max(end,right)
    return float(total+end-start)


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--resume-after-parse-failure',action='store_true');a=p.parse_args()
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7'
    trace=a.output/'trace.json';sample_file=a.output/'host-samples.json'
    if a.resume_after_parse_failure:
        assert a.output.is_dir() and trace.exists() and not (a.output/'result.json').exists(), '只能恢复已有trace的未完成解析'
        # 先确认原trace确实可解析，不能把采集失败伪装成解析恢复。
        raw=json.loads(trace.read_text());raw=raw.get('traceEvents',[]) if isinstance(raw,dict) else raw
        assert any(e.get('ph')=='X' and 'Task Type' in e.get('args',{}) for e in raw)
    else:
        assert not a.output.exists(),'已有诊断原件，不覆盖重跑'
        a.output.mkdir(parents=True)
    spec=json.loads(a.spec.read_text());source=Path(spec['source'])
    assert hashlib.sha256(source.read_bytes()).hexdigest()==spec['source_sha256']
    import torch
    import torch_npu
    torch.set_num_threads(1)
    candidate=imported('component_candidate',source);oracle=imported('component_oracle',Path(spec['leaf'])/'oracle.py')
    host=[];sync=[];total=[]
    for i in range(10):
        # 原始gen_inputs在计时范围之外；原地算子也不靠重复变换后的数据采样。
        inputs=oracle.gen_inputs(spec['case'],'npu:0');torch.npu.synchronize()
        start=time.perf_counter_ns();output=candidate.run(**inputs);submitted=time.perf_counter_ns()
        torch.npu.synchronize();finished=time.perf_counter_ns()
        if i>=3:
            host.append((submitted-start)/1000);sync.append((finished-submitted)/1000);total.append((finished-start)/1000)
    plan=[]
    for key,value in getattr(candidate,'_ENTRIES',{}).items():
        if isinstance(value,(tuple,list)) and len(value)>=8:
            plan.append(dict(key=str(key),grid=value[0],mode=value[7],raw_launch_available=value[2] is not None,
                             packed_argument_tail=list(value[10]) if len(value)>10 else
                             list(value[9]) if len(value)>9 and isinstance(value[9],(tuple,list)) else None))
    samples=dict(host_call_us=summary(host),sync_tail_us=summary(sync),synchronized_call_us=summary(total),dispatch_plan=plan,
        recovery_note='原trace保持字节不变；此前解析异常前的宿主样本未落盘，本次仅重新采集宿主诊断，不重复正式评测或trace' if a.resume_after_parse_failure else None)
    assert not sample_file.exists(),'已有宿主样本不可覆盖'
    sample_file.write_text(json.dumps(samples,ensure_ascii=False,indent=2)+'\n')
    inputs=oracle.gen_inputs(spec['case'],'npu:0');torch.npu.synchronize()
    if not a.resume_after_parse_failure:
        with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
                record_shapes=False,with_stack=False,schedule=torch_npu.profiler.schedule(wait=0,warmup=0,active=1,repeat=1),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(a.output/'raw'))) as prof:
            with torch.profiler.record_function('HGKDA_HOST_CALL'):output=candidate.run(**inputs)
            with torch.profiler.record_function('HGKDA_SYNC_WAIT'):torch.npu.synchronize()
            prof.step()
        prof.export_chrome_trace(str(trace))
    data=json.loads(trace.read_text());events=data.get('traceEvents',[]) if isinstance(data,dict) else data
    tasks=[e for e in events if e.get('ph')=='X' and e.get('dur') is not None and
           ('Task Type' in e.get('args',{}) or 'kernel' in str(e.get('cat','')).lower())]
    assert tasks,'设备任务为空，不能作成功分解'
    result=dict(status='completed',spec=spec,probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        **samples,
        device_trace=str(trace),device_trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
        device_task_union_us=interval_union(tasks),tasks=[{k:e.get(k) for k in ['name','ts','dur','args']} for e in tasks],
        limitations=['宿主调用包括分配、参数、启动及可能的同步；不是纯Python指令耗时',
            'sync_tail是调用返回后的等待；设备工作可能已与宿主调用重叠',
            '设备任务来自独立带profiler采样，不与无profiler宿主时间相减或相加',
            '独立诊断只解释执行路径，不作为正式加速比或额外独立复验'])
    (a.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','host_call_us','dispatch_plan','device_task_union_us']}))


if __name__=='__main__':main()
