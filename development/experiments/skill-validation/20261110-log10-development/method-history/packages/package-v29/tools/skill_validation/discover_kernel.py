"""在调用区间采集真实设备任务名称；输入生成在采集区间外。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--side',choices=['candidate','reference'],required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    spec=json.loads(a.spec.read_bytes())
    import torch,torch_npu
    torch.npu.set_device(0);torch.set_num_threads(1)
    oracle=load('_discover_oracle',Path(spec['leaf'])/'oracle.py')
    fn=oracle.run if a.side=='reference' else getattr(load('_discover_candidate',spec['source']),spec.get('entrypoint','run'))
    for _ in range(5):fn(**oracle.gen_inputs(spec['case'],'npu:0'))
    inputs=oracle.gen_inputs(spec['case'],'npu:0');torch.npu.synchronize()
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=False,with_stack=False) as prof:
        fn(**inputs);torch.npu.synchronize()
    trace=a.output.with_name(a.output.stem+'-trace.json');prof.export_chrome_trace(str(trace))
    document=json.loads(trace.read_bytes());events=document.get('traceEvents',[]) if isinstance(document,dict) else document
    tasks=[r for r in events if r.get('ph')=='X' and r.get('args',{}).get('Task Type')]
    assert tasks,'未找到真实设备任务'
    result=dict(source_sha256=spec['source_sha256'],side=a.side,kernels=sorted({r['name'] for r in tasks}),
        tasks=[dict(name=r['name'],duration_us=r['dur'],task_type=r['args']['Task Type']) for r in tasks],
        trace=dict(path=str(trace),sha256=hashlib.sha256(trace.read_bytes()).hexdigest()))
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(side=a.side,kernels=result['kernels'])))


if __name__=='__main__':main()
