"""单CPU编译真实第二轮整块核，核对毫秒级退化是否来自被吞掉的编译失败。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def forbidden(*args, **kwargs):
    raise RuntimeError('离线检查禁止设备初始化、加载和提交')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists(), '保留已有离线编译原件'
    args.output.mkdir(parents=True)
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    os.environ['OMP_NUM_THREADS']='1'
    os.environ['TRITON_ASCEND_ARCH']='Ascend910B4'
    os.environ['TRITON_CACHE_DIR']=str(args.output/'compiler-cache')
    import torch
    import torch_npu
    torch.npu._lazy_init=forbidden
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.compiler.compiler import CompiledKernel
    class NoRuntime:
        def __getattr__(self,name): forbidden()
    triton.runtime.driver.set_active(NoRuntime())
    CompiledKernel._init_handles=forbidden
    spec=importlib.util.spec_from_file_location('expm1_exact_cpu',args.source)
    candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
    rows=[]
    for dtype in ['fp16','bf16','fp32']:
        started=time.monotonic()
        row=dict(dtype=dtype,block=8192,narrow=int(dtype!='fp32'))
        try:
            compiled=triton.compile(ASTSource(candidate._expm1_exact,
                dict(a_ptr='*'+dtype,n='i32'),dict(NARROW=row['narrow'],BLOCK=8192)),
                target=GPUTarget('npu','Ascend910B4',0),options=dict(num_warps=32))
            assert compiled.module is None and compiled.function is None and compiled._run is None
            folder=args.output/dtype;folder.mkdir()
            artifacts={}
            for kind,raw in compiled.asm.items():
                if not isinstance(raw,(str,bytes)):continue
                path=folder/('kernel.'+kind)
                path.write_bytes(raw.encode() if isinstance(raw,str) else raw)
                artifacts[kind]=dict(path=str(path),sha256=sha(path))
            row.update(status='compiled_without_runtime',artifacts=artifacts)
        except Exception as error:
            row.update(status='compile_failed',error=type(error).__name__+': '+str(error))
        row['elapsed_seconds']=time.monotonic()-started
        rows.append(row)
        value=dict(source=str(args.source),source_sha256=sha(args.source),
            checker_sha256=sha(__file__),rows=rows,device_executed=False,performance_measured=False,
            scope='显式910B4目标、单CPU、独立缓存、禁止设备操作；只核验编译路径，不能直接充当12ms宿主耗时的因果测量。')
        (args.output/'result.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(json.dumps({k:v for k,v in row.items() if k not in {'artifacts','error'}},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
