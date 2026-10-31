"""合并Python输出分配与C++启动入口，避免进入已挂起的at::empty_like路径。"""
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
    folder=BATCH/'native-host-python-allocation-v9';folder.mkdir(parents=True,exist_ok=True)
    manifest=BATCH/'host-build-python-allocation-v9.json'
    if manifest.exists():raise FileExistsError(manifest)
    body=(OLD/'native-build-v4/ascendc_host.cpp').read_text()
    marker='PYBIND11_MODULE(TORCH_EXTENSION_NAME,module) {'
    body=body.replace(marker,'''static py::object allocate_output;
py::object execute_allocated(py::object input,py::object fallback) {
    py::object output=allocate_output(input);
    return execute(input,output,2,fallback);
}

'''+marker)
    body=body.replace('    module.def("configure_library",&configure_library);','    allocate_output=py::module_::import("torch").attr("empty_like");\n    module.def("execute_allocated",&execute_allocated);\n    module.def("configure_library",&configure_library);')
    source=folder/'ascendc_host.cpp';source.write_text(body,encoding='utf8')
    build=ROOT/'runtime/remaining-final-host-v9';build.mkdir(parents=True,exist_ok=True)
    npu=Path(torch_npu.__file__).parent
    module=load(name='_hgkda_remaining_host_v9',sources=[str(source)],build_directory=str(build),
        extra_include_paths=[str(npu/'include'),'/usr/local/Ascend/cann-9.0.0/include'],extra_cflags=['-O2','-g0'],
        extra_ldflags=['-L'+str(npu/'lib'),'-ltorch_npu','-Wl,-rpath,'+str(npu/'lib')],with_cuda=False,verbose=True)
    binary=Path(module.__file__)
    record=dict(status='built',binary=str(binary),torch=torch.__version__,abi=torch._C._GLIBCXX_USE_CXX11_ABI,
        files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,binary]})
    manifest.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(record))

if __name__=='__main__':main()
