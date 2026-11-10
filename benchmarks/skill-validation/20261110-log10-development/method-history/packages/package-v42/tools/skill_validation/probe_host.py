"""对候选宿主热路径取样；只用于函数开销定位，不作为正式性能成绩。"""
import argparse
import cProfile
import hashlib
import importlib.util
import json
from pathlib import Path
import pstats
import statistics
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    import torch,torch_npu
    torch.npu.set_device(0);torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('_profile_host_candidate',args.source)
    candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
    x=torch.full((1024,),-.25,dtype=torch.float16,device='npu:0')
    for _ in range(5):candidate.run(x)
    torch.npu.synchronize()
    def measure(fn):
        values=[]
        for _ in range(3):
            start=time.perf_counter_ns()
            for _ in range(1000):fn()
            values.append((time.perf_counter_ns()-start)/1000/1000)
            torch.npu.synchronize()
        return statistics.median(values)
    plain=measure(lambda:candidate.run(x))
    probe=cProfile.Profile();probe.enable()
    for _ in range(1000):candidate.run(x)
    probe.disable();torch.npu.synchronize()
    rows=[]
    for (file,line,name),(primitive,total,self_time,cumulative,callers) in pstats.Stats(probe).stats.items():
        rows.append(dict(file=file,line=line,function=name,calls=total,self_us=self_time*1e6/1000,cumulative_us=cumulative*1e6/1000))
    rows.sort(key=lambda r:r['cumulative_us'],reverse=True)
    result=dict(source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        uninstrumented_submit_us=plain,profiled_top_functions=rows[:30],
        cache_entries=len(getattr(candidate,'_LAUNCH_CACHE',{})),
        current_device_us=measure(torch_npu.npu.current_device),
        current_stream_us=measure(lambda:torch_npu.npu.current_stream(x.device).npu_stream),
        scope='函数级宿主诊断；cProfile有额外开销，累计时间不能相加，提交可能包含背压，不替代KGS成绩')
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
