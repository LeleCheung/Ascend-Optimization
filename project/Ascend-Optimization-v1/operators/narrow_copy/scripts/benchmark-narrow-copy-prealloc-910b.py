#!/usr/bin/env python3
"""测量正式 run 与实验性预分配 run_into 的同卡差异。"""
from __future__ import annotations
import argparse, importlib.util, json, statistics, time
from pathlib import Path
import torch
import torch_npu  # noqa: F401

def load(path):
    spec=importlib.util.spec_from_file_location("candidate",path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def timed(fn):
    torch.npu.synchronize(); b=time.perf_counter_ns(); v=fn(); e=time.perf_counter_ns(); torch.npu.synchronize(); f=time.perf_counter_ns(); del v
    return (e-b)/1000,(f-b)/1000

def summary(xs):
    return {"enqueue_median_us":statistics.median(x[0] for x in xs),"wall_median_us":statistics.median(x[1] for x in xs),
            "wall_min_us":min(x[1] for x in xs),"wall_max_us":max(x[1] for x in xs)}

def main():
    p=argparse.ArgumentParser(); p.add_argument("candidate",type=Path); p.add_argument("output",type=Path); p.add_argument("--device",default="npu:0"); p.add_argument("--repeats",type=int,default=30); a=p.parse_args()
    torch.npu.set_device(a.device); m=load(a.candidate); cases=[]
    for label,shape,dtype,dim,start,length in (("small-f32",(64,64),torch.float32,0,16,32),("large-f16",(1024,65536),torch.float16,0,256,512)):
        x=torch.ones(shape,device=a.device,dtype=dtype); out=torch.empty((*shape[:dim],length,*shape[dim+1:]),device=a.device,dtype=dtype)
        # warm-up establishes compiled specialization and validates output.
        y=m.run_into(out,x,dim,start,length); torch.npu.synchronize(); assert torch.equal(y,torch.narrow_copy(x,dim,start,length))
        samples={"run":[],"run_into":[],"pytorch":[]}
        for i in range(a.repeats+5):
            order=(("run",lambda:m.run(x,dim,start,length)),("run_into",lambda:m.run_into(out,x,dim,start,length)),("pytorch",lambda:torch.narrow_copy(x,dim,start,length)))
            if i%2: order=tuple(reversed(order))
            for name,fn in order:
                v=timed(fn)
                if i>=5:samples[name].append(v)
        cases.append({"case":label,"shape":list(shape),"dtype":str(dtype),"preallocated_output_shape":list(out.shape),"results":{k:summary(v) for k,v in samples.items()}})
    result={"schema_version":"1.0","device":a.device,"repeats":a.repeats,"scope":"同一进程、显式同步；run_into 是实验性 ABI，不是 KGS Definition","cases":cases}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(result,ensure_ascii=False))

if __name__=="__main__":main()
