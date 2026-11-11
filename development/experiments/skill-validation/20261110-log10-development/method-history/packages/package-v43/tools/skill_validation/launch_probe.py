"""诊断实际候选冷／热调用的JIT路径与缓存状态；不改候选或正式计时合同。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def imported(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['source','leaf','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--case',action='append',required=True)
    p.add_argument('--exercise-hooks',action='store_true');a=p.parse_args();assert not a.output.exists()
    import torch,torch_npu,triton
    from triton.runtime.jit import JITFunction
    torch.set_num_threads(1);torch.npu.set_device(0)
    candidate=imported('_launch_probe_candidate',a.source)
    oracle=imported('_launch_probe_inputs',a.leaf/'oracle.py')
    cases=[json.loads(line) for line in (a.leaf/'timing.jsonl').read_text().splitlines() if line.strip()]
    original=JITFunction.run;calls=[];returned=[]

    def counted(kernel,*args,**kwargs):
        calls.append(kernel.__name__)
        compiled=original(kernel,*args,**kwargs)
        if compiled is not None:
            returned.append(dict(kernel=kernel.__name__,type=type(compiled).__name__,
                has_function=hasattr(compiled,'function'),function_is_none=getattr(compiled,'function',None) is None,
                has_packed_metadata=hasattr(compiled,'packed_metadata'),has_run=hasattr(compiled,'run')))
        return compiled

    def state():
        globals_={name:len(value) for name,value in vars(candidate).items()
                  if name.startswith('_') and not name.startswith('__') and isinstance(value,dict)}
        jit=[]
        for name,value in vars(candidate).items():
            if not isinstance(value,JITFunction):continue
            fields={}
            for key,item in vars(value).items():
                if 'cache' not in key:continue
                row=dict(type=type(item).__name__)
                try:row['length']=len(item)
                except TypeError:pass
                if isinstance(item,dict):
                    row['entry_types']=[type(v).__name__ for v in item.values()]
                    row['entry_shapes']=[dict(type=type(v).__name__,length=len(v),
                        component_types=[type(c).__name__ for c in v],
                        first_dict_value_types=sorted({type(c).__name__ for c in v[0].values()}) if v and isinstance(v[0],dict) else [])
                        for v in item.values() if isinstance(v,(tuple,list))]
                fields[key]=row
            jit.append(dict(name=name,cache_fields=fields))
        return dict(candidate_dict_sizes=globals_,jit=jit)

    records=[];hook_records=[];JITFunction.run=counted
    try:
        for case_name in a.case:
            case=next(c for c in cases if c['name']==case_name)
            for iteration in range(3):
                inputs=oracle.gen_inputs(case,'npu:0');torch.npu.synchronize()
                before=len(calls);candidate.run(**inputs);torch.npu.synchronize()
                records.append(dict(case=case_name,iteration=iteration,jit_calls=calls[before:],state=state()))
        if a.exercise_hooks:
            runtime=triton.knobs.runtime
            enter,leave=runtime.launch_enter_hook,runtime.launch_exit_hook
            assert isinstance(enter.calls,list) and isinstance(leave.calls,list)
            seen=dict(enter=0,exit=0)
            def on_enter(*args,**kwargs):seen['enter']+=1
            def on_exit(*args,**kwargs):seen['exit']+=1
            enter.calls.append(on_enter);leave.calls.append(on_exit)
            try:
                case=next(c for c in cases if c['name']==a.case[0])
                inputs=oracle.gen_inputs(case,'npu:0');torch.npu.synchronize()
                before=len(calls);candidate.run(**inputs);torch.npu.synchronize()
                hook_records.append(dict(case=a.case[0],jit_calls=calls[before:],callbacks=seen.copy()))
                assert calls[before:] and seen==dict(enter=1,exit=1),'活动钩子没有通过普通Triton路径执行'
            finally:
                enter.calls.remove(on_enter);leave.calls.remove(on_exit)
    finally:JITFunction.run=original
    knobs=getattr(getattr(triton,'knobs',None),'runtime',None)
    output=dict(source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),records=records,returned_compiled=returned,
        hook_records=hook_records,
        runtime_hooks={k:dict(is_none=getattr(knobs,k,None) is None,type=type(getattr(knobs,k,None)).__name__,
                       callback_count=len(getattr(getattr(knobs,k,None),'calls',[])))
                       for k in ['launch_enter_hook','launch_exit_hook']},
        scope='调用区间只计数JIT路径并读取缓存元数据；输入每次重新构造，非正式性能或因果耗时测量')
    a.output.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status='completed',records=len(records),jit_calls=len(calls))))


if __name__=='__main__':main()
