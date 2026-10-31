"""完整评测后的宿主调用诊断；只分解Python调用，不代替正式KGS成绩。"""
import argparse
import cProfile
import hashlib
import importlib.util
import json
import os
import pstats
import sys
from pathlib import Path

sys.dont_write_bytecode=True


def imported(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['source','leaf','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expected-sha',required=True)
    parser.add_argument('--calls',type=int,default=100)
    args=parser.parse_args()
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7'
    if args.output.exists():raise FileExistsError(args.output)
    if not 1<=args.calls<=1000:raise ValueError('调用数应在1到1000之间')
    digest=hashlib.sha256(args.source.read_bytes()).hexdigest()
    if digest!=args.expected_sha:raise ValueError('候选源码身份不同')
    import torch
    import torch_npu
    oracle=imported('host_oracle',args.leaf/'oracle.py')
    candidate=imported('host_candidate',args.source)
    case=json.loads((args.leaf/'timing.jsonl').read_text().splitlines()[0])
    inputs=oracle.gen_inputs(case,'npu:0')
    for _ in range(5):output=candidate.run(**inputs)
    torch.npu.synchronize()
    profiler=cProfile.Profile()
    profiler.enable()
    for _ in range(args.calls):output=candidate.run(**inputs)
    profiler.disable()
    torch.npu.synchronize()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    raw=args.output.with_suffix('.pstats');profiler.dump_stats(str(raw))
    rows=[]
    for (file,line,function),(primitive,calls,total,cumulative,_) in pstats.Stats(profiler).stats.items():
        rows.append(dict(file=file,line=line,function=function,calls=calls,primitive_calls=primitive,
                         self_seconds=total,cumulative_seconds=cumulative))
    rows.sort(key=lambda row:row['self_seconds'],reverse=True)
    caches={}
    for name in ['_CK_CACHE','_ENTRIES','_FAST']:
        value=getattr(candidate,name,None)
        if isinstance(value,dict):
            caches[name]=dict(entries=len(value),disabled=sum(v is False or v is None for v in value.values()),
                              value_types=sorted({type(v).__name__ for v in value.values()}))
    result=dict(status='completed',source=str(args.source),source_sha256=digest,case=case,calls=args.calls,
                top_self_time=rows[:30],top_cumulative_time=sorted(rows,key=lambda row:row['cumulative_seconds'],reverse=True)[:20],
                compiled_caches=caches,profile=str(raw),profile_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='同一代表输入，预热5次后cProfile记录宿主Python调用；计时含诊断扰动，不能代替KGS或据此推导设备耗时')
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status=result['status'],source_sha256=digest,compiled_caches=caches,
                         top=[dict(function=row['function'],calls=row['calls'],self_seconds=row['self_seconds']) for row in rows[:6]])))


if __name__=='__main__':main()
