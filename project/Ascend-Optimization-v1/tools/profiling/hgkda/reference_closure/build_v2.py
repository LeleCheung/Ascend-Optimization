"""构建独立CANN参考；只编译，不使用NPU。"""
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
os.environ['MAX_JOBS']='1'
import torch
import torch_npu
from torch.utils.cpp_extension import load
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[3]
BATCH=PROJECT/'operators/campaigns/reference-closure-20261031'
BUILD=Path('/data/hanle/ascend-optimization/runtime/reference-closure-cann-v2')
BUILD.mkdir(parents=True,exist_ok=True)
npu=Path(torch_npu.__file__).parent
source=HERE/'cann_baselines_v2.cpp'
module=load(name='_hgkda_reference_closure_v2',sources=[str(source)],build_directory=str(BUILD),
    extra_include_paths=[str(npu/'include'),str(npu/'include/third_party/op-plugin'),'/usr/local/Ascend/cann-9.0.0/include'],
    extra_cflags=['-O2','-g0'],extra_ldflags=['-L'+str(npu/'lib'),'-ltorch_npu','-Wl,-rpath,'+str(npu/'lib'),
      '-L/usr/local/Ascend/cann-9.0.0/lib64','-lopapi','-lascendcl','-Wl,-rpath,/usr/local/Ascend/cann-9.0.0/lib64'],
    with_cuda=False,verbose=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
binary=Path(module.__file__)
out=BATCH/'references/artifacts';out.mkdir(parents=True,exist_ok=True)
frozen=out/binary.name;frozen.write_bytes(binary.read_bytes())
headers={n:Path('/usr/local/Ascend/cann-9.0.0/include/aclnnop')/n for n in ['aclnn_trunc.h','aclnn_expm1.h','aclnn_inverse.h','aclnn_batch_norm.h','aclnn_fake_quant_per_tensor_affine_cachemask.h']}
for n,p in headers.items():(out/n).write_bytes(p.read_bytes())
record=dict(status='built',module_name='_hgkda_reference_closure_v2',source=str(source),source_sha256=sha(source),binary=str(frozen),binary_sha256=sha(frozen),
    headers={str(p):sha(p) for p in headers.values()},torch=torch.__version__,torch_npu=torch_npu.__version__,abi=torch._C._GLIBCXX_USE_CXX11_ABI,architecture=platform.machine(),
    scope='厂商CANN目标库直接调用；不是HG-KDA自主实现；不调用PyTorch目标算子，输出分配和必要copy明确计入')
(BATCH/'cann-build-v2.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(record,ensure_ascii=False))
