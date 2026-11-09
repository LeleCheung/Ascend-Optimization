// 宿主启动缓存：检查张量元数据，获取当前地址与stream，调用Triton生成的启动器。
#include <torch/csrc/autograd/python_variable.h>
#include <pybind11/pybind11.h>
#include <array>
#include <cstdint>
#include <unordered_map>

namespace py=pybind11;

struct Key {
    int64_t elements;
    int dtype;
    int device;
    bool operator==(const Key& other) const {
        return elements==other.elements && dtype==other.dtype && device==other.device;
    }
};
struct KeyHash {
    size_t operator()(const Key& key) const {
        return std::hash<int64_t>{}(key.elements) ^ (std::hash<int>{}(key.dtype)<<1) ^
               (std::hash<int>{}(key.device)<<3);
    }
};
struct Plan {
    py::object launch;
    py::tuple arguments;
    py::object stream_getter;
};

class UnaryLaunchCache {
public:
    explicit UnaryLaunchCache(py::object fallback):fallback_(std::move(fallback)){}

    void register_plan(py::object input,py::object launch,py::tuple arguments,py::object stream_getter) {
        const auto& tensor=checked(input);
        if (!eligible(tensor)) throw py::value_error("requires aligned contiguous floating NPU tensor");
        if (arguments.size()<10 || arguments.size()>24 || !arguments[3].is_none() || !arguments[9].is_none())
            throw py::value_error("stream and tensor address must be dynamic placeholders");
        if (!PyCallable_Check(launch.ptr()) || !PyCallable_Check(stream_getter.ptr()))
            throw py::value_error("launch and stream getter must be callable");
        plans_.insert_or_assign(key(tensor),Plan{std::move(launch),std::move(arguments),std::move(stream_getter)});
    }

    py::object run(py::object input) {
        const auto& tensor=checked(input);
        if (!eligible(tensor)) return slow(input);
        auto found=plans_.find(key(tensor));
        if (found==plans_.end()) return slow(input);
        // 保留启动器、元数据和getter的强引用；缓存不持有输入张量、数据值或旧stream。
        const Plan plan=found->second;
        py::object stream=plan.stream_getter();
        py::object pointer=py::int_(reinterpret_cast<uintptr_t>(tensor.data_ptr()));
        std::array<PyObject*,24> arguments{};
        const size_t count=plan.arguments.size();
        for (size_t index=0;index<count;++index) arguments[index]=PyTuple_GET_ITEM(plan.arguments.ptr(),index);
        arguments[3]=stream.ptr();
        arguments[9]=pointer.ptr();
        PyObject* result=PyObject_Vectorcall(plan.launch.ptr(),arguments.data(),count,nullptr);
        // 启动异常直接传播，防止原地算子在异常后被再次执行。
        if (!result) throw py::error_already_set();
        Py_DECREF(result);
        ++fast_calls_;
        return input;
    }

    py::dict stats() const {
        py::dict output;
        output["plans"]=plans_.size();
        output["fast_calls"]=fast_calls_;
        output["fallback_calls"]=fallback_calls_;
        return output;
    }

private:
    static const at::Tensor& checked(const py::object& object) {
        if (!THPVariable_Check(object.ptr())) throw py::type_error("expected tensor");
        return THPVariable_Unpack(object.ptr());
    }
    static Key key(const at::Tensor& tensor) {
        return {tensor.numel(),static_cast<int>(tensor.scalar_type()),tensor.device().index()};
    }
    static bool eligible(const at::Tensor& tensor) {
        const auto dtype=tensor.scalar_type();
        return tensor.device().type()==c10::DeviceType::PrivateUse1 && tensor.numel()>0 &&
               (dtype==at::kHalf || dtype==at::kBFloat16 || dtype==at::kFloat) &&
               tensor.is_contiguous() && !(reinterpret_cast<uintptr_t>(tensor.data_ptr())&15);
    }
    py::object slow(const py::object& input) {
        ++fallback_calls_;
        return fallback_(input);
    }
    py::object fallback_;
    std::unordered_map<Key,Plan,KeyHash> plans_;
    uint64_t fast_calls_=0;
    uint64_t fallback_calls_=0;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME,module) {
    py::class_<UnaryLaunchCache>(module,"UnaryLaunchCache")
        .def(py::init<py::object>())
        .def("register_plan",&UnaryLaunchCache::register_plan)
        .def("run",&UnaryLaunchCache::run)
        .def("stats",&UnaryLaunchCache::stats);
}
