"""构建Ascend C宿主任务队列封装，不执行NPU负载。"""
import hashlib
import json
import os
from pathlib import Path
import platform
os.environ['MAX_JOBS']='1'
import torch
import torch_npu
from torch.utils.cpp_extension import load

here=Path(__file__).resolve().parent
build=Path('/data/hanle/ascend-optimization/runtime/threeway-ascendc-host-build')
build.mkdir(parents=True,exist_ok=True)
npu=Path(torch_npu.__file__).parent
source=here/'ascendc_host.cpp'
module=load(name='_hgkda_ascendc_host_v2',sources=[str(source)],build_directory=str(build),
    extra_include_paths=[str(npu/'include'),'/usr/local/Ascend/cann-9.0.0/include'],
    extra_cflags=['-O2','-g0'],extra_ldflags=['-L'+str(npu/'lib'),'-ltorch_npu','-Wl,-rpath,'+str(npu/'lib')],
    with_cuda=False,verbose=True)
binary=Path(module.__file__)
record=dict(status='built',source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            binary=str(binary),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            torch=torch.__version__,torch_npu=torch_npu.__version__,architecture=platform.machine(),abi=torch._C._GLIBCXX_USE_CXX11_ABI)
(here.parents[3]/'operators/campaigns/threeway-20261030/ascendc-host-build.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(record,ensure_ascii=False))
