"""保留同stream任务队列，在C++输出分配期间释放GIL。"""
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
    folder=BATCH/'native-host-queue-gil-v12';folder.mkdir(parents=True,exist_ok=True)
    manifest=BATCH/'host-build-queue-gil-v12.json'
    if manifest.exists():raise FileExistsError(manifest)
    body=(OLD/'native-build-v4/ascendc_host.cpp').read_text()
    old='    at::Tensor out=supplied?THPVariable_Unpack(output.ptr()):((op<2 || op==5)?input:at::empty_like(input));'
    assert body.count(old)==1
    body=body.replace(old,'''    at::Tensor out;
    if(supplied) out=THPVariable_Unpack(output.ptr());
    else if(op<2 || op==5) out=input;
    else {
        py::gil_scoped_release release;
        out=at::empty_like(input);
    }''')
    source=folder/'ascendc_host.cpp';source.write_text(body,encoding='utf8')
    build=ROOT/'runtime/remaining-final-host-v12';build.mkdir(parents=True,exist_ok=True)
    npu=Path(torch_npu.__file__).parent
    module=load(name='_hgkda_remaining_host_v12',sources=[str(source)],build_directory=str(build),
        extra_include_paths=[str(npu/'include'),'/usr/local/Ascend/cann-9.0.0/include'],extra_cflags=['-O2','-g0'],
        extra_ldflags=['-L'+str(npu/'lib'),'-ltorch_npu','-Wl,-rpath,'+str(npu/'lib')],with_cuda=False,verbose=True)
    binary=Path(module.__file__)
    value=dict(status='built',binary=str(binary),torch=torch.__version__,abi=torch._C._GLIBCXX_USE_CXX11_ABI,
        files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,binary]})
    manifest.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(value))

if __name__=='__main__':main()
