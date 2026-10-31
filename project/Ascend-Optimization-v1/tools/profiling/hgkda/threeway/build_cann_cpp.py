"""只构建CANN C++调度封装，不执行NPU负载。"""
import hashlib
import argparse
import json
import os
from pathlib import Path
import platform
import sys

os.environ['MAX_JOBS']='1'
import torch
import torch_npu
from torch.utils.cpp_extension import load

here=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',default='cann_cpp.cpp')
parser.add_argument('--name',default='_hgkda_cann_cpp_v1')
parser.add_argument('--record',default='cann-cpp-build.json')
args=parser.parse_args()
if any('/' in s or '\\' in s for s in [args.source,args.name,args.record]):
    raise ValueError('构建参数只接受文件名或模块名')
build=Path('/data/hanle/ascend-optimization/runtime')/('threeway-cann-cpp-build' if args.name=='_hgkda_cann_cpp_v1' else args.name)
build.mkdir(parents=True,exist_ok=True)
npu=Path(torch_npu.__file__).parent
source=here/args.source
module=load(name=args.name,sources=[str(source)],build_directory=str(build),
    extra_include_paths=[str(npu/'include'),str(npu/'include/third_party/op-plugin'),'/usr/local/Ascend/cann-9.0.0/include'],
    extra_cflags=['-O2','-g0'],extra_ldflags=['-L'+str(npu/'lib'),'-ltorch_npu','-Wl,-rpath,'+str(npu/'lib'),
       '-L/usr/local/Ascend/cann-9.0.0/lib64','-lopapi','-lascendcl','-Wl,-rpath,/usr/local/Ascend/cann-9.0.0/lib64'],
    with_cuda=False,verbose=True)
binary=Path(module.__file__)
headers=[Path('/usr/local/Ascend/cann-9.0.0/include/aclnn/acl_meta.h'),
         npu/'include/torch_npu/csrc/framework/OpCommand.h']
record=dict(status='built',module_name=args.name,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    binary=str(binary),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
    headers={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in headers},
    python=sys.version,torch=torch.__version__,torch_npu=torch_npu.__version__,architecture=platform.machine(),abi=torch._C._GLIBCXX_USE_CXX11_ABI,
    scope='直接CANN算子库；复用torch_npu的元数据、任务队列和执行器生命周期，不调用PyTorch目标算子')
path=here.parents[3]/'operators/campaigns/threeway-20261030'/args.record
path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
(build/'build-provenance.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(record,ensure_ascii=False))
