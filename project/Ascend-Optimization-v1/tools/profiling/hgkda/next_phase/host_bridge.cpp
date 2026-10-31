// 宿主启动包装；只读取张量元数据并调用已经编译的Triton启动器。
#include <torch/csrc/autograd/python_variable.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <unordered_map>

namespace py = pybind11;

struct Key {
  int64_t elements;
  int scalar_type;
  int device;
  bool operator==(const Key& other) const {
    return elements == other.elements && scalar_type == other.scalar_type && device == other.device;
  }
};

struct KeyHash {
  size_t operator()(const Key& key) const {
    return std::hash<int64_t>{}(key.elements) ^
      (std::hash<int>{}(key.scalar_type) << 1) ^ (std::hash<int>{}(key.device) << 3);
  }
};

struct Plan {
  py::object launch;
  py::tuple arguments;
  py::object stream_getter;
};

class InplaceLaunchCache {
 public:
  explicit InplaceLaunchCache(py::object fallback) : fallback_(std::move(fallback)) {}

  // 每个候选独立持有实例；计划只保存代码和元数据，不保存张量或输出值。
  void register_plan(py::object tensor, py::object launch, py::tuple arguments, py::object stream_getter) {
    const auto& t = checked(tensor);
    if (!eligible(t)) throw py::value_error("plan requires a contiguous aligned nonempty NPU tensor");
    if (arguments.size() < 10 || !arguments[3].is_none() || !arguments[9].is_none())
      throw py::value_error("stream slot 3 and tensor-pointer slot 9 must be dynamic placeholders");
    if (!PyCallable_Check(launch.ptr()) || !PyCallable_Check(stream_getter.ptr()))
      throw py::value_error("launch and stream getter must be callable");
    plans_.insert_or_assign(key(t), Plan{std::move(launch), std::move(arguments), std::move(stream_getter)});
  }

  py::object run(py::object tensor) {
    const auto& t = checked(tensor);
    if (!eligible(t)) return slow(tensor);
    auto it = plans_.find(key(t));
    if (it == plans_.end()) return slow(tensor);
    const Plan plan = it->second;
    // 每次取本次地址和当前stream；shape/dtype/device变化都会重新查找计划。
    py::object stream = plan.stream_getter();
    py::tuple arguments(plan.arguments.size());
    for (size_t i = 0; i < plan.arguments.size(); ++i) arguments[i] = plan.arguments[i];
    arguments[3] = stream;
    arguments[9] = py::int_(reinterpret_cast<uintptr_t>(t.data_ptr()));
    PyObject* result = PyObject_CallObject(plan.launch.ptr(), arguments.ptr());
    if (!result) throw py::error_already_set();
    Py_DECREF(result);
    ++fast_calls_;
    return tensor;
  }

  py::dict stats() const {
    py::dict out;
    out["plans"] = plans_.size();
    out["fast_calls"] = fast_calls_;
    out["fallback_calls"] = fallback_calls_;
    return out;
  }

 private:
  static const at::Tensor& checked(const py::object& object) {
    if (!THPVariable_Check(object.ptr())) throw py::type_error("expected a torch tensor");
    return THPVariable_Unpack(object.ptr());
  }
  static Key key(const at::Tensor& tensor) {
    return {tensor.numel(), static_cast<int>(tensor.scalar_type()), tensor.device().index()};
  }
  static bool eligible(const at::Tensor& tensor) {
    return tensor.device().type() == c10::DeviceType::PrivateUse1 && tensor.numel() > 0 &&
      tensor.is_contiguous() && !(reinterpret_cast<uintptr_t>(tensor.data_ptr()) & 15);
  }
  py::object slow(const py::object& tensor) {
    ++fallback_calls_;
    return fallback_(tensor);
  }
  py::object fallback_;
  std::unordered_map<Key, Plan, KeyHash> plans_;
  uint64_t fast_calls_ = 0;
  uint64_t fallback_calls_ = 0;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  py::class_<InplaceLaunchCache>(module, "InplaceLaunchCache")
    .def(py::init<py::object>())
    .def("register_plan", &InplaceLaunchCache::register_plan)
    .def("run", &InplaceLaunchCache::run)
    .def("stats", &InplaceLaunchCache::stats);
}
