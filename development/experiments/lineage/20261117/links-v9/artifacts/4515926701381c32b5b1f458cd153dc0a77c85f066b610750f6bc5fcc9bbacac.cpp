// 来源可见Ascend C kernel的宿主任务队列封装；不计算目标，不缓存张量值。
#include <torch/csrc/autograd/python_variable.h>
#include <pybind11/pybind11.h>
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include <dlfcn.h>
#include <cstdint>

namespace py=pybind11;
using Launch=void(*)(uint32_t,void*,uint8_t*,uint32_t,int);
static void* library_handle=nullptr;
static Launch launch_kernel=nullptr;

void configure_library(const std::string& path) {
    if (library_handle) throw py::value_error("kernel library already configured");
    library_handle=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);
    if (!library_handle) throw py::value_error(dlerror());
    launch_kernel=reinterpret_cast<Launch>(dlsym(library_handle,"hgkda_log10_launch"));
    if (!launch_kernel) throw py::value_error(dlerror());
}

py::object run_log10(py::object object,py::object fallback) {
    if (!THPVariable_Check(object.ptr())) throw py::type_error("expected tensor");
    const auto& input=THPVariable_Unpack(object.ptr());
    auto n=input.numel();
    if (!n) return object;
    int dtype=input.scalar_type()==at::kFloat ? 0 : input.scalar_type()==at::kHalf ? 1 : input.scalar_type()==at::kBFloat16 ? 27 : -1;
    auto pointer=reinterpret_cast<uintptr_t>(input.data_ptr());
    if (!launch_kernel || input.device().type()!=c10::DeviceType::PrivateUse1 ||
        dtype<0 || n>=static_cast<int64_t>(UINT32_MAX) || n%16 || pointer%32 || !input.is_contiguous())
        return fallback(object);
    auto stream=c10_npu::getCurrentNPUStream(input.device().index());
    uint32_t blocks=static_cast<uint32_t>(std::min<int64_t>(40,(n+16383)/16384));
    at::Tensor held=input;
    // 仅在宿主工作线程执行实际launch；前序PyTorch任务顺序由同一任务队列保持。
    auto callback=[held,stream,blocks,n,dtype]() -> int {
        launch_kernel(blocks,stream.stream(false),static_cast<uint8_t*>(held.data_ptr()),static_cast<uint32_t>(n),dtype);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("HGKDA_AscendC_Log10",callback);
    return object;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,module) {
    module.def("configure_library",&configure_library);
    module.def("run_log10",&run_log10);
}
