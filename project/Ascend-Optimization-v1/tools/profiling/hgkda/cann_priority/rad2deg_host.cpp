// 元数据、当前stream及同stream任务提交，数学计算由自写设备核完成。
#include <torch/csrc/autograd/python_variable.h>
#include <pybind11/pybind11.h>
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include <dlfcn.h>
#include <cstdint>
namespace py=pybind11;
using Launch=void(*)(uint32_t,void*,uint8_t*,uint32_t,int);
static void* handle=nullptr;
static Launch launch=nullptr;
void configure_library(const std::string& path) {
    if(handle)throw py::value_error("kernel library already configured");
    handle=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);
    if(!handle)throw py::value_error(dlerror());
    launch=reinterpret_cast<Launch>(dlsym(handle,"hgkda_rad2deg_launch"));
    if(!launch)throw py::value_error(dlerror());
}
py::object execute(py::object object,py::object fallback) {
    if(!THPVariable_Check(object.ptr()))throw py::type_error("expected tensor");
    const auto& input=THPVariable_Unpack(object.ptr());
    auto n=input.numel();
    int dtype=input.scalar_type()==at::kFloat?0:input.scalar_type()==at::kHalf?1:input.scalar_type()==at::kBFloat16?27:-1;
    if(!launch || dtype<0 || input.device().type()!=c10::DeviceType::PrivateUse1 || n<=0 || n>=UINT32_MAX ||
       n%16 || reinterpret_cast<uintptr_t>(input.data_ptr())%32 || !input.is_contiguous())return fallback(object);
    auto stream=c10_npu::getCurrentNPUStream(input.device().index());
    uint32_t blocks=static_cast<uint32_t>(std::min<int64_t>(40,(n+8191)/8192));
    at::Tensor held=input;
    auto callback=[held,stream,blocks,n,dtype]() -> int {
        launch(blocks,stream.stream(false),static_cast<uint8_t*>(held.data_ptr()),static_cast<uint32_t>(n),dtype);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("HGKDA_AscendC_Rad2deg",callback);
    return object;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("configure_library",&configure_library);m.def("execute",&execute);}
