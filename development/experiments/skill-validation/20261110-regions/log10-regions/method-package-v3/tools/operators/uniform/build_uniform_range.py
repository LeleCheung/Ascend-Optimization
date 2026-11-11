"""在独立目录单线程编译 CANN Uniform＋Clamp 参考桥；不启动 NPU 任务。"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    os.environ['MAX_JOBS']='1'
    if hasattr(os,'sched_setaffinity'):
        os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    import torch
    import torch_npu
    from torch.utils.cpp_extension import load
    root=Path(torch_npu.__file__).parent
    source=Path(__file__).with_name('cann_uniform_range.cpp')
    frozen=args.output/source.name
    frozen.write_bytes(source.read_bytes())
    name='_hgkda_cann_uniform_range_'+sha(frozen)[:12]
    module=load(name=name,sources=[str(frozen)],build_directory=str(args.output),
                extra_include_paths=[str(root/'include'),str(args.sdk/'include')],
                extra_cflags=['-O2','-g0'],extra_ldflags=[
                    '-L'+str(root/'lib'),'-ltorch_npu','-Wl,-rpath,'+str(root/'lib'),
                    '-L'+str(args.sdk/'lib64'),'-lopapi','-lascendcl','-Wl,-rpath,'+str(args.sdk/'lib64')],
                with_cuda=False,verbose=True)
    cpu_checks=[]
    for dtype in ['float16','bfloat16','float32']:
        for low,high in [(-.01,.02),(.1,.9),(-1.,1.),(0.,1.),(-1.e-6,1.e-6)]:
            lower,upper=module.bounds(dtype,low,high)
            lo=torch.tensor(lower,dtype=getattr(torch,dtype))
            hi=torch.tensor(upper,dtype=getattr(torch,dtype))
            assert lo.item()==lower and hi.item()==upper and lower>=low and upper<high
            assert torch.nextafter(lo,torch.tensor(-float('inf'),dtype=lo.dtype)).item()<low
            assert torch.nextafter(hi,torch.tensor(float('inf'),dtype=hi.dtype)).item()>=high
            cpu_checks.append(dict(dtype=dtype,low=low,high=high,lower=lower,upper=upper,passed=True))
    binding='''import hashlib,importlib.util
from pathlib import Path
import torch,torch_npu
'''
    binding+='p=Path(__file__).with_name('+repr(Path(module.__file__).name)+')\n'
    binding+='assert hashlib.sha256(p.read_bytes()).hexdigest()=='+repr(sha(module.__file__))+'\n'
    binding+='s=importlib.util.spec_from_file_location('+repr(name)+',p)\n'
    binding+='_cpp=importlib.util.module_from_spec(s);s.loader.exec_module(_cpp)\n'
    binding+='assert torch.__version__=='+repr(torch.__version__)+' and torch._C._GLIBCXX_USE_CXX11_ABI=='+repr(torch._C._GLIBCXX_USE_CXX11_ABI)+'\n'
    binding+='def run(self,from_=0.,to=1.,*,generator=None):\n    return _cpp.uniform(self,from_,to,generator)\n'
    (args.output/'reference.py').write_text(binding,encoding='utf8')
    value=dict(module_name=name,binary=str(Path(module.__file__).resolve()),binary_sha256=sha(module.__file__),
               source=str(frozen.resolve()),source_sha256=sha(frozen),torch=torch.__version__,
               torch_npu=torch_npu.__version__,abi=torch._C._GLIBCXX_USE_CXX11_ABI,
               reference_kind='CANN_Uniform_Clamp_range_compatibility',
               reference=str((args.output/'reference.py').resolve()),reference_sha256=sha(args.output/'reference.py'),
               cpu_bounds_checks=cpu_checks,
               qualified_CANN=False,device_executed=False,
               headers={str(args.sdk/'include/aclnnop'/name):sha(args.sdk/'include/aclnnop'/name)
                        for name in ['aclnn_uniform.h','aclnn_clamp.h']})
    (args.output/'build.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(value,ensure_ascii=False))


if __name__=='__main__':
    main()
