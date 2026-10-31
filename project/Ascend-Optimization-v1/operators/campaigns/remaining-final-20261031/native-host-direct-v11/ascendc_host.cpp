// 只做元数据、分配、当前stream及任务队列；目标计算在自写Ascend C源码内。
#include <torch/csrc/autograd/python_variable.h>
#include <pybind11/pybind11.h>
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include <dlfcn.h>
#include <cstdint>
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

namespace py=pybind11;
using Launch=void(*)(uint32_t,void*,uint8_t*,uint8_t*,uint32_t,int);
static void* handle=nullptr;
static Launch kernels[7]={};

void configure_library(const std::string& path) {
    if(handle) throw py::value_error("kernel library already configured");
    handle=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);
    if(!handle) throw py::value_error(dlerror());
    const char* names[]={"hgkda_negative_launch","hgkda_log2_launch","hgkda_fix_launch","hgkda_round_launch","hgkda_hardswish_launch","hgkda_expm1_launch","hgkda_fmod_launch"};
    for(int i=0;i<7;i++) {
        kernels[i]=reinterpret_cast<Launch>(dlsym(handle,names[i]));
        if(!kernels[i]) throw py::value_error(dlerror());
    }
}

py::object execute(py::object object,py::object output,int op,py::object fallback) {
    if(!THPVariable_Check(object.ptr())) throw py::type_error("expected input tensor");
    const auto& input=THPVariable_Unpack(object.ptr());
    bool supplied=!output.is_none();
    if(supplied && !THPVariable_Check(output.ptr())) throw py::type_error("expected output tensor");
    auto n=input.numel();
    int dtype=input.scalar_type()==at::kFloat?0:input.scalar_type()==at::kHalf?1:input.scalar_type()==at::kBFloat16?27:-1;
    auto pointer=reinterpret_cast<uintptr_t>(input.data_ptr());
    bool eligible=handle && op>=0 && op<7 && dtype>=0 && input.device().type()==c10::DeviceType::PrivateUse1 && n>0 && n<UINT32_MAX && n%16==0 && pointer%32==0 && input.is_contiguous();
    if(supplied) {
        const auto& out=THPVariable_Unpack(output.ptr());
        eligible=eligible && out.sizes()==input.sizes() && out.scalar_type()==input.scalar_type() && out.device()==input.device() && out.is_contiguous() && reinterpret_cast<uintptr_t>(out.data_ptr())%32==0;
        // 部分重叠交由原路径；相同起址的逐元素原地操作是安全的。
        if(out.is_alias_of(input) && out.data_ptr()!=input.data_ptr()) eligible=false;
    }
    if(op==6 && !supplied) eligible=false;
    if(!eligible) return supplied ? fallback(object,output) : fallback(object);
    at::Tensor out;
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
    return op==6?object:(supplied?output:((op<2 || op==5)?object:py::reinterpret_steal<py::object>(THPVariable_Wrap(out))));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,module) {
    module.def("configure_library",&configure_library);
    module.def("execute",&execute);
}
