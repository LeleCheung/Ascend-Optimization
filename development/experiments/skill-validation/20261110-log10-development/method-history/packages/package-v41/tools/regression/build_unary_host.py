"""单线程构建仅处理元数据与Triton启动的宿主缓存；设备计算沿用冻结候选。"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    os.environ['MAX_JOBS']='1'
    if hasattr(os,'sched_setaffinity'):
        os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    import torch
    from torch.utils.cpp_extension import load
    source=Path(__file__).with_name('unary_host_cache.cpp')
    frozen=args.output/source.name
    frozen.write_bytes(source.read_bytes())
    name='_hgkda_unary_host_'+sha(frozen)[:12]
    module=load(name=name,sources=[str(frozen)],build_directory=str(args.output),
                extra_cflags=['-O2','-g0'],with_cuda=False,verbose=True)
    base=args.source.read_bytes().decode('utf8')
    wrapper='''
# 宿主桥只组织元数据及调用Triton启动器；计算核与全部回退保持父版本。
import hashlib as _bridge_hash
import importlib.util as _bridge_import
from pathlib import Path as _BridgePath
'''
    wrapper+='_bridge_path=_BridgePath('+repr(str(Path(module.__file__).resolve()))+')\n'
    wrapper+='assert _bridge_hash.sha256(_bridge_path.read_bytes()).hexdigest()=='+repr(sha(module.__file__))+'\n'
    wrapper+='assert torch.__version__=='+repr(torch.__version__)+' and torch._C._GLIBCXX_USE_CXX11_ABI=='+repr(torch._C._GLIBCXX_USE_CXX11_ABI)+'\n'
    wrapper+='_bridge_spec=_bridge_import.spec_from_file_location('+repr(name)+',_bridge_path)\n'
    wrapper+='_bridge_module=_bridge_import.module_from_spec(_bridge_spec)\n_bridge_spec.loader.exec_module(_bridge_module)\n'
    wrapper+='''_parent_run=run
def _bridge_fallback(A):
    result=_parent_run(A)
    if A.numel() and A.is_contiguous() and not (A.data_ptr()&15):
        entry=_FAST.get(A.data_ptr())
        if entry is not None and entry[2] is not None:
            arguments=(entry[4],1,1,None,entry[5],entry[6],None,None,None,None)
            if entry[2]:arguments+=(entry[1],)
            arguments+=(None,)
            _BRIDGE.register_plan(A,entry[3],arguments,entry[7])
    return result
_BRIDGE=_bridge_module.UnaryLaunchCache(_bridge_fallback)
def run(A):
    return _BRIDGE.run(A)
'''
    (args.output/'candidate.py').write_bytes((base+wrapper).encode('utf8'))
    value=dict(module_name=name,binary=str(Path(module.__file__).resolve()),binary_sha256=sha(module.__file__),
               source=str(frozen.resolve()),source_sha256=sha(frozen),torch=torch.__version__,abi=torch._C._GLIBCXX_USE_CXX11_ABI,
               parent_source=str(args.source.resolve()),parent_source_sha256=sha(args.source),
               candidate=str((args.output/'candidate.py').resolve()),candidate_sha256=sha(args.output/'candidate.py'),
               device_computation='inherited_Triton_only',device_executed=False,validated=False)
    (args.output/'build.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(value,ensure_ascii=False))


if __name__=='__main__':
    main()
