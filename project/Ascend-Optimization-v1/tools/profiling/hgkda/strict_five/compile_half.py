"""单CPU编译FP16原生类型提案；禁止运行时初始化、加载及设备提交。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def forbidden(*args,**kwargs):raise RuntimeError('离线编译禁止设备初始化、加载或提交')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});os.environ['OMP_NUM_THREADS']='1'
    os.environ['TRITON_ASCEND_ARCH']='Ascend910B4';os.environ['TRITON_CACHE_DIR']=str(a.output/'compiler-cache')
    import torch
    import torch_npu
    torch.npu._lazy_init=forbidden
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.compiler.compiler import CompiledKernel
    class NoRuntime:
        def __getattr__(self,name):forbidden()
    triton.runtime.driver.set_active(NoRuntime());CompiledKernel._init_handles=forbidden
    spec=importlib.util.spec_from_file_location('half_proposal',a.source)
    candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
    rows=[]
    for dtype in ['fp16','bf16','fp32']:
        for symbol in ['_expm1_flat','_expm1_streaming']:
            row=dict(dtype=dtype,symbol=symbol)
            try:
                compiled=triton.compile(ASTSource(getattr(candidate,symbol),dict(a_ptr='*'+dtype,n='i32'),dict(BLOCK=4096)),
                    target=GPUTarget('npu','Ascend910B4',0),options=dict(num_warps=32))
                assert compiled.module is None and compiled.function is None and compiled._run is None
                folder=a.output/(dtype+'-'+symbol);folder.mkdir();artifacts={}
                for kind,raw in compiled.asm.items():
                    if not isinstance(raw,(str,bytes)):continue
                    f=folder/('kernel.'+kind);f.write_bytes(raw.encode() if isinstance(raw,str) else raw)
                    artifacts[kind]=dict(path=str(f),sha256=sha(f))
                row.update(status='compiled_without_runtime',artifacts=artifacts)
            except Exception as error:row.update(status='compile_failed',error=type(error).__name__+': '+str(error))
            rows.append(row)
            value=dict(source=str(a.source),source_sha256=sha(a.source),checker_sha256=sha(__file__),
                device_executed=False,performance_measured=False,rows=rows)
            (a.output/'result.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
            print(json.dumps({k:v for k,v in row.items() if k!='artifacts'}),flush=True)


if __name__=='__main__':main()
