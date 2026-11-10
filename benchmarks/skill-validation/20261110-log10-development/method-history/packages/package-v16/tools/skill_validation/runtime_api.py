"""只读提取本机Triton后端启动接口及版本；不生成或执行目标算子实现。"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    import torch,torch_npu,triton
    import triton.backends.ascend.driver as driver
    import triton.backends.ascend.backend_register as backend
    import triton.compiler.compiler as compiler
    records=[]
    for module in [driver,backend,compiler]:
        path=Path(inspect.getfile(module));raw=path.read_bytes();lines=raw.decode('utf8').splitlines()
        selected=set()
        for i,line in enumerate(lines):
            if any(key in line for key in ['def make_launcher','PyArg_ParseTuple(args','def get_current_stream',
                                           '_npu_getCurrentRawStream','OpCommand','SetCustomHandler','TRITON_ENABLE_TASKQUEUE',
                                           'def _init_handles','self.run =','self.packed_metadata =','def __getitem__',
                                           'def format_of','args_format =','ty in ("constexpr")']):
                selected.update(range(max(0,i-5),min(len(lines),i+24)))
        records.append(dict(module=module.__name__,path=str(path),sha256=hashlib.sha256(raw).hexdigest(),
             excerpts=[dict(line=i+1,text=lines[i]) for i in sorted(selected)]))
    result=dict(torch=torch.__version__,triton=triton.__version__,sources=records,
        scope='当前运行环境的只读接口事实；不包含目标算子候选实现，ABI仍需当前候选预检')
    from triton.runtime.jit import JITFunction
    callables=[]
    for function in [JITFunction.run,JITFunction.__getitem__]:
        lines=inspect.getsource(function).splitlines()
        selected=sorted(set(list(range(min(16,len(lines))))+list(range(max(0,len(lines)-24),len(lines)))))
        callables.append(dict(name=function.__qualname__,source_file=inspect.getsourcefile(function),
            excerpts=[dict(relative_line=i+1,text=lines[i]) for i in selected]))
    result['jit_callables']=callables
    result['knobs_locations']={name:hasattr(module,'knobs') for name,module in [('triton',triton),('triton.runtime',triton.runtime)]}
    runtime=getattr(getattr(triton,'knobs',None),'runtime',None)
    hooks={}
    for name in ['launch_enter_hook','launch_exit_hook']:
        value=getattr(runtime,name,None)
        record=dict(type=type(value).__name__,is_none=value is None,truthy=bool(value))
        callbacks=getattr(value,'calls',None)
        if isinstance(callbacks,list):record['callback_count']=len(callbacks)
        if value is not None:
            try:record['length']=len(value)
            except TypeError:pass
            try:
                cls=type(value);record['class_source']=inspect.getsource(cls)
                file=Path(inspect.getfile(cls));record['source_sha256']=hashlib.sha256(file.read_bytes()).hexdigest()
            except (OSError,TypeError):pass
        hooks[name]=record
    result['runtime_hooks']=hooks
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status='completed',versions=dict(torch=result['torch'],triton=result['triton']))))


if __name__=='__main__':main()
