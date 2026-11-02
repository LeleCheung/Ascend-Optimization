"""单CPU交叉编译最小数值探测，禁止设备初始化与加载。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    os.environ['OMP_NUM_THREADS']='1'
    os.environ['TRITON_ASCEND_ARCH']='Ascend910B4'
    os.environ['TRITON_CACHE_DIR']=str(args.output/'compiler-cache')
    import torch
    import torch_npu
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.compiler.compiler import CompiledKernel

    def forbidden(*args,**kwargs):raise RuntimeError('离线编译禁止设备初始化、加载或提交')
    class NoRuntime:
        def __getattr__(self,name):return forbidden()
    torch.npu._lazy_init=forbidden
    triton.runtime.driver.set_active(NoRuntime())
    CompiledKernel._init_handles=forbidden
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    probe=args.project/'tools/profiling/hgkda/strict_night/masked_update_probe.py'
    module=load('offline_mask_probe_tool',probe)
    source=args.project/'operators/campaigns/strict-night-20261102/proposals/linalg_inv_ex/codex-masked-operands-v7/main.py'
    text=source.read_text(encoding='utf8')
    text=text[text.index('@triton.jit\ndef _compensated_fma'):text.index('@triton.jit\ndef _leftlooking')]
    kernel=args.output/'kernel.py'
    kernel.write_text('import triton\nimport triton.language as tl\n\n'+text+module.KERNELS,encoding='utf8')
    kernels=load('offline_mask_probe_kernels',kernel)
    target=GPUTarget('npu','Ascend910B4',0)
    rows=[]
    cases=[('_masked_update',dict(A='*fp32',F='*fp32',V='*fp32',T='*i32',OUT='*fp32',STEPS='i32'),dict(BLOCK=16,MODE=i)) for i in [0,1,2]]
    cases.append(('_reciprocal_forms',dict(D='*fp32',T='*i32',OLD='*fp32',VECTOR='*fp32'),dict(BLOCK=16)))
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    for i,(name,signature,constants) in enumerate(cases):
        row=dict(name=name,constants=constants)
        try:
            compiled=triton.compile(ASTSource(getattr(kernels,name),signature,constants),target=target,
                                    options=dict(num_warps=32,enable_fp_fusion=False))
            assert compiled.module is None and compiled.function is None and compiled._run is None
            artifacts={}
            for kind,data in compiled.asm.items():
                if not isinstance(data,(str,bytes)):continue
                path=args.output/(str(i)+'.'+kind)
                path.write_bytes(data.encode() if isinstance(data,str) else data)
                artifacts[kind]=dict(path=str(path),sha256=sha(path))
            row.update(status='compiled_without_runtime',artifacts=artifacts)
        except Exception as error:row.update(status='compile_failed',error=type(error).__name__+': '+str(error)[:4000])
        rows.append(row)
        (args.output/'result.json').write_text(json.dumps(dict(checker_sha256=sha(__file__),
            probe_sha256=sha(probe),source_sha256=sha(source),kernel_sha256=sha(kernel),rows=rows,
            device_executed=False,performance_measured=False),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(json.dumps({k:v for k,v in row.items() if k!='artifacts'}),flush=True)

if __name__=='__main__':main()
