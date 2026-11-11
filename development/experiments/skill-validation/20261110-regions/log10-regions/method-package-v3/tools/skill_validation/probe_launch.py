"""补测当前候选的小/大输入调用和真机任务轨迹，并记录实际编译启动器接口。"""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import statistics
import time


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def save(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--leaf',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--trace',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    import torch,torch_npu
    from triton.runtime.jit import JITFunction
    torch.set_num_threads(1);torch.npu.set_device(0)
    candidate=load('_probe_candidate',args.source)
    oracle=load('_probe_reference',args.leaf/'oracle.py')
    cases=[json.loads(line) for line in (args.leaf/'timing.jsonl').read_text().splitlines() if line.strip()]
    selected=[r for i,r in enumerate(cases) if i in [0,3,4,7,8,11]]
    original=JITFunction.run;entries={}
    def capture(kernel,*values,**options):
        compiled=original(kernel,*values,**options)
        if compiled is not None and hasattr(compiled,'asm'):
            key=(kernel.__name__,str(compiled.metadata))
            if key not in entries:
                folder=args.output/('compiled-'+str(len(entries)));folder.mkdir()
                artifacts=[]
                for kind,value in compiled.asm.items():
                    if kind not in ['ttir','bcmlir','npubin']:continue
                    raw=value.encode() if isinstance(value,str) else value
                    path=folder/('kernel.'+kind);path.write_bytes(raw)
                    artifacts.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest()))
                launcher=compiled.run
                try:signature=str(inspect.signature(launcher.launch))
                except (ValueError,TypeError):signature=getattr(launcher.launch,'__doc__',None)
                entries[key]=dict(kernel=kernel.__name__,metadata=str(compiled.metadata),
                    launcher_class=type(launcher).__module__+'.'+type(launcher).__name__,
                    launch_signature=signature,launcher_fields=sorted(vars(launcher)) if hasattr(launcher,'__dict__') else [],
                    compiled_fields=[key for key in ['function','packed_metadata','metadata','run'] if hasattr(compiled,key)],
                    artifacts=artifacts)
        return compiled
    measurements=[]
    try:
        for case in selected:
            for side,fn in [('candidate',candidate.run),('reference',oracle.run)]:
                inputs=oracle.gen_inputs(case,'npu:0')
                # 记录钩子只在预热期间启用，正式诊断计时恢复原调度器。
                JITFunction.run=capture if side=='candidate' else original
                for _ in range(5):fn(**inputs)
                JITFunction.run=original
                torch.npu.synchronize()
                records=[]
                for _ in range(3):
                    torch.npu.synchronize();start=time.perf_counter_ns()
                    for _ in range(50):fn(**inputs)
                    submitted=time.perf_counter_ns();torch.npu.synchronize();finished=time.perf_counter_ns()
                    records.append(dict(submit_us=(submitted-start)/50/1000,wall_us=(finished-start)/50/1000))
                measurements.append(dict(case=case['name'],spec=case['inputs']['spec'],side=side,
                    submit_us=statistics.median(r['submit_us'] for r in records),
                    wall_us=statistics.median(r['wall_us'] for r in records),samples=records))
    finally:JITFunction.run=original
    if args.trace:
        traces=[]
        for case in selected[:2]:
            for side,fn in [('candidate',candidate.run),('reference',oracle.run)]:
                inputs=oracle.gen_inputs(case,'npu:0')
                for _ in range(5):fn(**inputs)
                torch.npu.synchronize()
                with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU],record_shapes=False,with_stack=False) as prof:
                    with torch.profiler.record_function(side+'-'+case['name']):
                        for _ in range(3):fn(**inputs)
                        torch.npu.synchronize()
                path=args.output/(case['name']+'-'+side+'-trace.json')
                prof.export_chrome_trace(str(path))
                traces.append(dict(case=case['name'],side=side,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        save(args.output/'traces.json',traces)
    save(args.output/'probe.json',dict(source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        measurements=measurements,compiled=list(entries.values()),
        scope='独立诊断窗口；submit包含Python提交及可能的背压，wall包含同步，不作为正式KGS成绩。设备本体从任务轨迹另行统计，不用wall减submit推算。'))
    print(json.dumps(dict(cases=len(selected),measurements=[{k:r[k] for k in ['case','side','submit_us','wall_us']} for r in measurements])))


if __name__=='__main__':main()
