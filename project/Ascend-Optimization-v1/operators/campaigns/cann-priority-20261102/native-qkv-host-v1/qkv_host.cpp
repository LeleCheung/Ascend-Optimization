// 保留既有Triton计算；将元数据、分配、指针和参数打包移至C++，不调用CANN目标算子。
#include <torch/csrc/autograd/python_variable.h>
#include <pybind11/pybind11.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include <unordered_map>
#include <sstream>
namespace py=pybind11;
static std::unordered_map<std::string,py::object>* entries=new std::unordered_map<std::string,py::object>();

py::object execute(py::object xo,py::object bo,int64_t heads,py::object fallback,py::object compiler) {
    if(!THPVariable_Check(xo.ptr()) || !THPVariable_Check(bo.ptr()))return fallback(xo,bo,heads);
    const auto& x=THPVariable_Unpack(xo.ptr());const auto& bias=THPVariable_Unpack(bo.ptr());
    if(x.dim()!=3 || bias.dim()!=1 || heads<=0 || x.size(2)%(3*heads) ||
       bias.numel()!=x.size(2) || x.device()!=bias.device() || x.scalar_type()!=bias.scalar_type() ||
       x.device().type()!=c10::DeviceType::PrivateUse1 ||
       (x.scalar_type()!=at::kFloat && x.scalar_type()!=at::kHalf && x.scalar_type()!=at::kBFloat16))
        return fallback(xo,bo,heads);
    int64_t batch=x.size(0),seq=x.size(1),dim=x.size(2)/(3*heads);
    if(!batch || !seq || !dim || reinterpret_cast<uintptr_t>(x.data_ptr())%16 ||
       reinterpret_cast<uintptr_t>(bias.data_ptr())%16 || (batch*heads*seq*dim*x.element_size())%16)
        return fallback(xo,bo,heads);
    auto packed=at::empty({3,batch,heads,seq,dim},x.options());
    auto q=packed.select(0,0),k=packed.select(0,1),v=packed.select(0,2);
    py::object qo=py::reinterpret_steal<py::object>(THPVariable_Wrap(q));
    py::object ko=py::reinterpret_steal<py::object>(THPVariable_Wrap(k));
    py::object vo=py::reinterpret_steal<py::object>(THPVariable_Wrap(v));
    int64_t bd=1;while(bd<dim)bd*=2;
    py::tuple constants=py::make_tuple(batch,seq,heads,dim,x.stride(0),x.stride(1),x.stride(2),bias.stride(0),64,bd);
    std::ostringstream key;
    key<<int(x.scalar_type())<<':'<<x.device().index();
    for(auto value:constants)key<<':'<<py::cast<int64_t>(value);
    auto found=entries->find(key.str());
    if(found==entries->end()) {
        py::object entry=compiler(xo,bo,qo,ko,vo,constants);
        found=entries->emplace(key.str(),entry).first;
    }
    if(found->second.is_none())return fallback(xo,bo,heads);
    py::tuple entry=py::cast<py::tuple>(found->second);
    // 等待当前stream的host任务提交，保持上游运算的stream顺序；不缓存地址或Tensor。
    auto stream=c10_npu::getCurrentNPUStream(x.device().index()).stream();
    py::tuple args(24);
    args[0]=py::int_((seq+63)/64);args[1]=py::int_(batch*heads);args[2]=py::int_(1);
    args[3]=py::int_(reinterpret_cast<uintptr_t>(stream));args[4]=entry[1];args[5]=entry[2];
    for(int i=6;i<9;++i)args[i]=py::none();
    args[9]=py::int_(reinterpret_cast<uintptr_t>(x.data_ptr()));
    args[10]=py::int_(reinterpret_cast<uintptr_t>(bias.data_ptr()));
    args[11]=py::int_(reinterpret_cast<uintptr_t>(q.data_ptr()));
    args[12]=py::int_(reinterpret_cast<uintptr_t>(k.data_ptr()));
    args[13]=py::int_(reinterpret_cast<uintptr_t>(v.data_ptr()));
    for(int i=0;i<10;++i)args[14+i]=constants[i];
    PyObject* result=PyObject_CallObject(entry[0].ptr(),args.ptr());
    if(!result)throw py::error_already_set();
    Py_DECREF(result);
    return py::make_tuple(qo,ko,vo);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("execute",&execute);}
