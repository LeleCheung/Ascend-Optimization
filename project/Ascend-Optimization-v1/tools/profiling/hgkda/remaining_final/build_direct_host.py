"""直接启动设备函数；分配和等待队列时释放GIL，显式登记stream内存生命周期。"""
import hashlib
import json
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[3];ROOT=PROJECT.parents[2]
BATCH=PROJECT/'operators/campaigns/remaining-final-20261031'
OLD=PROJECT/'operators/campaigns/remaining-nine-20261030'

def main():
    os.environ['MAX_JOBS']='1'
    import torch,torch_npu
    from torch.utils.cpp_extension import load
    folder=BATCH/'native-host-direct-v11';folder.mkdir(parents=True,exist_ok=True)
    manifest=BATCH/'host-build-direct-v11.json'
    if manifest.exists():raise FileExistsError(manifest)
    body=(OLD/'native-build-v4/ascendc_host.cpp').read_text()
    body=body.replace('#include <cstdint>', '#include <cstdint>\n#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"')
    start=body.index('    at::Tensor out=supplied?')
    end=body.index('    return op==6?',start)
    body=body[:start]+'''    at::Tensor out;
    {
        // Python侧的分配通常会释放GIL；原C++直接分配路径也显式释放。
        py::gil_scoped_release release;
        out=supplied?THPVariable_Unpack(output.ptr()):((op<2 || op==5)?input:at::empty_like(input));
        auto stream=c10_npu::getCurrentNPUStream(input.device().index());
        // 默认stream()先排空已有宿主任务队列，保证先前copy/计算先于本次直接启动。
        auto rawStream=stream.stream();
        c10_npu::NPUCachingAllocator::recordStream(input.storage().data_ptr(),stream);
        c10_npu::NPUCachingAllocator::recordStream(out.storage().data_ptr(),stream);
        uint32_t blocks=static_cast<uint32_t>(std::min<int64_t>(40,(n+4095)/4096));
        kernels[op](blocks,rawStream,static_cast<uint8_t*>(input.data_ptr()),static_cast<uint8_t*>(out.data_ptr()),static_cast<uint32_t>(n),dtype);
    }
'''+body[end:]
    source=folder/'ascendc_host.cpp';source.write_text(body,encoding='utf8')
    build=ROOT/'runtime/remaining-final-host-v11';build.mkdir(parents=True,exist_ok=True)
    npu=Path(torch_npu.__file__).parent
    module=load(name='_hgkda_remaining_host_v11',sources=[str(source)],build_directory=str(build),
        extra_include_paths=[str(npu/'include'),'/usr/local/Ascend/cann-9.0.0/include'],extra_cflags=['-O2','-g0'],
        extra_ldflags=['-L'+str(npu/'lib'),'-ltorch_npu','-Wl,-rpath,'+str(npu/'lib')],with_cuda=False,verbose=True)
    binary=Path(module.__file__)
    record=dict(status='built',binary=str(binary),torch=torch.__version__,abi=torch._C._GLIBCXX_USE_CXX11_ABI,
        files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,binary]})
    manifest.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(record))

if __name__=='__main__':main()
