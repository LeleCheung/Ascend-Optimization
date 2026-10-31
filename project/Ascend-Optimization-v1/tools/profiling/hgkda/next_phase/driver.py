"""同一冻结Native输入的Triton/PyTorch/CANN直接参考驱动。"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent


def imported(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--side',choices=['candidate','pytorch','cann','ascendc'],required=True);p.add_argument('--trace',type=Path);p.add_argument('--ir',type=Path);p.add_argument('--semantic',type=Path);a=p.parse_args()
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7' or os.environ.get('HGKDA_SIMULATOR')=='1'
    if os.environ.get('HGKDA_SIMULATOR')=='1':
        os.environ['ASCEND_RT_VISIBLE_DEVICES']='0'
    import torch
    import torch_npu
    spec=json.loads(a.spec.read_text(encoding='utf8'))
    source=Path(spec['source']);assert hashlib.sha256(source.read_bytes()).hexdigest()==spec['source_sha256']
    oracle=imported('phase_oracle',Path(spec['leaf'])/'oracle.py')
    module=imported('phase_candidate',source) if a.side=='candidate' else oracle
    if a.side=='cann':
        from cann_reference import Native
        module=Native(spec['operator'])
    elif a.side=='ascendc':
        from ascendc_reference import Negative
        assert spec['operator']=='negative_'
        module=Negative('/data/hanle/ascend-optimization/runtime/next25-ascendc-build/lib/libhgkda_negative.so')
    case=spec['case'];value=oracle.gen_inputs(case,'npu:0')
    if a.semantic:
        assert not a.semantic.exists()
        expected_inputs=oracle.gen_inputs(case,'npu:0');expected=oracle.run(**expected_inputs)
        actual=module.run(**value);torch.npu.synchronize()
        refs=list(expected) if isinstance(expected,(list,tuple)) else [expected]
        outs=list(actual) if isinstance(actual,(list,tuple)) else [actual]
        checks=[]
        for i,(r,c) in enumerate(zip(refs,outs)):
            tol=case['tolerance'];torch.testing.assert_close(c,r,atol=tol['atol'],rtol=tol['rtol'],equal_nan=True)
            checks.append(dict(output=i,shape=list(c.shape),dtype=str(c.dtype),passed=True))
        assert len(refs)==len(outs)
        definition=json.loads((Path(spec['leaf'])/'definition.json').read_text(encoding='utf8'))
        for name,param in definition.get('effects',{}).get('returns_alias_of',{}).items():
            assert torch._C._is_alias_of(outs[definition['outputs'].index(name)],value[param])
        result=dict(status='passed',side=a.side,spec=spec,checks=checks)
        if a.side in {'cann','ascendc'}:result['reference']=module.provenance()
        a.semantic.parent.mkdir(parents=True,exist_ok=True);a.semantic.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8');return
    # 直接捕获实际JIT调用；动态生成的FlagGems包装函数不一定能由模块属性遍历找到。
    captured=[]
    if a.ir:
        from triton.runtime.jit import JITFunction
        original_run=JITFunction.run
        def capture(jit,*args,**kwargs):
            compiled=original_run(jit,*args,**kwargs)
            if hasattr(compiled,'result'):compiled=compiled.result()
            if compiled is not None and hasattr(compiled,'asm'):
                captured.append((jit.fn.__name__,0,compiled,0))
            return compiled
        JITFunction.run=capture
        try:module.run(**value);torch.npu.synchronize()
        finally:JITFunction.run=original_run
    else:
        module.run(**value);torch.npu.synchronize()
    value=oracle.gen_inputs(case,'npu:0');torch.npu.synchronize()
    if a.trace:
        assert not a.trace.exists();a.trace.parent.mkdir(parents=True,exist_ok=True)
        with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
                record_shapes=True,with_stack=False,schedule=torch_npu.profiler.schedule(wait=0,warmup=0,active=1,repeat=1),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(a.trace.parent/'raw'))) as prof:
            with torch.profiler.record_function('HGKDA_TARGET_'+spec['operator']):
                module.run(**value);torch.npu.synchronize()
            prof.step()
        prof.export_chrome_trace(str(a.trace));return
    module.run(**value);torch.npu.synchronize()
    if a.ir:
        assert a.side=='candidate' and not a.ir.exists();a.ir.mkdir(parents=True)
        helper=imported('phase_ir_helper',HERE.parent/'observe-endurance.py')
        entries=[]
        modules=[module]
        for obj in vars(module).values():
            dependent=inspect.getmodule(obj) if inspect.isfunction(obj) else None
            if dependent is not None and dependent.__name__.startswith('flag_gems.ops.') and dependent not in modules:modules.append(dependent)
        compiled_rows=[];seen=set()
        for row in captured:
            if id(row[2]) not in seen:seen.add(id(row[2]));compiled_rows.append(row)
        for dependent in modules:
            for row in helper.compiled_kernels(dependent):
                if id(row[2]) in seen:continue
                seen.add(id(row[2]));compiled_rows.append(row)
        for name,device,compiled,depth in compiled_rows:
            artifacts=[]
            for kind,data in compiled.asm.items():
                if not isinstance(data,(str,bytes)):continue
                raw=data.encode() if isinstance(data,str) else data
                f=a.ir/(name+'-'+str(len(entries))+'.'+kind);f.write_bytes(raw)
                artifacts.append(dict(kind=kind,path=str(f),sha256=hashlib.sha256(raw).hexdigest()))
            entries.append(dict(symbol=name,device=device,metadata=str(compiled.metadata),wrapper_depth=depth,artifacts=artifacts))
        (a.ir/'manifest.json').write_text(json.dumps(dict(spec=spec,entries=entries),ensure_ascii=False,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()
