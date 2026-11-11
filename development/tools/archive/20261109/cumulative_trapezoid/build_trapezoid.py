"""构建FP32累加的CANN积分参考，并生成保留中间舍入的独立Triton候选。"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parent-source', type=Path, required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['MAX_JOBS']='1'
    if hasattr(os,'sched_setaffinity'):
        os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    import torch
    import torch_npu
    from torch.utils.cpp_extension import load
    root=Path(torch_npu.__file__).parent
    frozen=args.output/'cann_trapezoid.cpp'
    frozen.write_bytes(Path(__file__).with_name(frozen.name).read_bytes())
    name='_hgkda_cann_trapezoid_'+sha(frozen)[:12]
    module=load(name=name, sources=[str(frozen)], build_directory=str(args.output),
                extra_include_paths=[str(root/'include'),str(args.sdk/'include')],
                extra_cflags=['-O2','-g0'],extra_ldflags=[
                    '-L'+str(root/'lib'),'-ltorch_npu','-Wl,-rpath,'+str(root/'lib'),
                    '-L'+str(args.sdk/'lib64'),'-lopapi','-lascendcl','-Wl,-rpath,'+str(args.sdk/'lib64')],
                with_cuda=False,verbose=True)
    binding='import hashlib,importlib.util\nfrom pathlib import Path\nimport torch,torch_npu\n'
    binding+='p=Path(__file__).with_name('+repr(Path(module.__file__).name)+')\n'
    binding+='assert hashlib.sha256(p.read_bytes()).hexdigest()=='+repr(sha(module.__file__))+'\n'
    binding+='assert torch.__version__=='+repr(torch.__version__)+'\n'
    binding+='s=importlib.util.spec_from_file_location('+repr(name)+',p)\n'
    binding+='_cpp=importlib.util.module_from_spec(s);s.loader.exec_module(_cpp)\n'
    binding+='def run(y,*,dx=1,dim=-1):\n    return _cpp.trapezoid(y,float(dx),int(dim))\n'
    (args.output/'reference.py').write_text(binding,encoding='utf8')
    parent=args.parent_source.read_bytes().decode('utf8')
    old='inc = ((a + b) * SCALE).to(Y.dtype.element_ty)'
    assert parent.count(old)==2
    # 两个函数的缩进不同，逐行保留源缩进。
    lines=[]
    for line in parent.splitlines(keepends=True):
        if old in line:
            indent=line[:len(line)-len(line.lstrip())]
            lines.extend([indent+'pair = (a + b).to(Y.dtype.element_ty).to(tl.float32)\n',
                          indent+'inc = (pair * SCALE).to(Y.dtype.element_ty)\n'])
        else:
            lines.append(line)
    candidate=''.join(lines)
    old_low='low = 1 if y.dtype == torch.float16 else 0'
    assert candidate.count(old_low)==1
    candidate=candidate.replace(old_low,'low = 0  # FP32前缀累加；加法及缩放仍保留输入类型舍入。')
    (args.output/'candidate.py').write_bytes(candidate.encode('utf8'))
    value=dict(reference_kind='CANN_Add_Muls_CumsumFP32_Cast_mathematical_contract',
               module_name=name,binary=str(Path(module.__file__).resolve()),binary_sha256=sha(module.__file__),
               source=str(frozen),source_sha256=sha(frozen),
               reference=str(args.output/'reference.py'),reference_sha256=sha(args.output/'reference.py'),
               parent_source=str(args.parent_source),parent_source_sha256=sha(args.parent_source),
               candidate=str(args.output/'candidate.py'),candidate_sha256=sha(args.output/'candidate.py'),
               torch=torch.__version__,torch_npu=torch_npu.__version__,device_executed=False,qualified=False,
               policy='独立数学参考合同；原NPU默认参考及失败证据保留，不声称原合同已经通过')
    (args.output/'build.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(value,ensure_ascii=False))


if __name__=='__main__':
    main()
