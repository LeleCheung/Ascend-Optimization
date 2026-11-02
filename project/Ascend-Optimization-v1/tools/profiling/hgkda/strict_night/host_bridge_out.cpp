// 仅负责out张量元数据和Triton启动，不包含设备计算或数据复制。
#include <torch/csrc/autograd/python_variable.h>
#include <pybind11/pybind11.h>
#include <array>
#include <cstdint>
#include <unordered_map>

namespace py = pybind11;

struct OutKey {
  int64_t elements;
  int input_type;
  int output_type;
  int device;
  bool operator==(const OutKey& other) const {
    return elements == other.elements && input_type == other.input_type &&
      output_type == other.output_type && device == other.device;
  }
};

struct OutKeyHash {
  size_t operator()(const OutKey& key) const {
    return std::hash<int64_t>{}(key.elements) ^
      (std::hash<int>{}(key.input_type) << 1) ^
      (std::hash<int>{}(key.output_type) << 3) ^
      (std::hash<int>{}(key.device) << 5);
  }
};

struct OutPlan {
  py::object launch;
  py::tuple arguments;
  py::object stream_getter;
};

class OutLaunchCache {
 public:
  explicit OutLaunchCache(py::object fallback) : fallback_(std::move(fallback)) {}

  void register_plan(py::object input, py::object output, py::object launch,
                     py::tuple arguments, py::object stream_getter) {
    const auto& x = checked(input);
    const auto& y = checked(output);
    if (!eligible(x, y)) throw py::value_error("plan requires compatible contiguous aligned NPU tensors");
    if (arguments.size() < 11 || arguments.size() > 32 ||
        !arguments[3].is_none() || !arguments[9].is_none() || !arguments[10].is_none())
      throw py::value_error("stream and both tensor pointers must be dynamic placeholders");
    if (!PyCallable_Check(launch.ptr()) || !PyCallable_Check(stream_getter.ptr()))
      throw py::value_error("launch and stream getter must be callable");
    plans_.insert_or_assign(key(x, y), OutPlan{std::move(launch), std::move(arguments), std::move(stream_getter)});
  }

  py::object run(py::object input, py::object output) {
    const auto& x = checked(input);
    const auto& y = checked(output);
    if (!eligible(x, y)) return slow(input, output);
    auto it = plans_.find(key(x, y));
    if (it == plans_.end()) return slow(input, output);
    // 强引用在整个提交期间保活；每次重新获取实际地址和当前stream。
    const OutPlan plan = it->second;
    py::object stream = plan.stream_getter();
    py::object input_pointer = py::int_(reinterpret_cast<uintptr_t>(x.data_ptr()));
    py::object output_pointer = py::int_(reinterpret_cast<uintptr_t>(y.data_ptr()));
    std::array<PyObject*, 32> arguments{};
    const size_t count = plan.arguments.size();
    for (size_t i = 0; i < count; ++i)
      arguments[i] = PyTuple_GET_ITEM(plan.arguments.ptr(), i);
    arguments[3] = stream.ptr();
    arguments[9] = input_pointer.ptr();
    arguments[10] = output_pointer.ptr();
    PyObject* result = PyObject_Vectorcall(plan.launch.ptr(), arguments.data(), count, nullptr);
    // 提交异常直接传播；绝不在异常后重新计算。
    if (!result) throw py::error_already_set();
    Py_DECREF(result);
    ++fast_calls_;
    return output;
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
  static OutKey key(const at::Tensor& x, const at::Tensor& y) {
    return {x.numel(), static_cast<int>(x.scalar_type()),
      static_cast<int>(y.scalar_type()), x.device().index()};
  }
  static bool eligible(const at::Tensor& x, const at::Tensor& y) {
    return x.device().type() == c10::DeviceType::PrivateUse1 &&
      x.device() == y.device() && x.numel() > 0 && x.sizes() == y.sizes() &&
      x.is_contiguous() && y.is_contiguous() &&
      !((reinterpret_cast<uintptr_t>(x.data_ptr()) | reinterpret_cast<uintptr_t>(y.data_ptr())) & 15);
  }
  py::object slow(const py::object& input, const py::object& output) {
    ++fallback_calls_;
    return fallback_(input, output);
  }
  py::object fallback_;
  std::unordered_map<OutKey, OutPlan, OutKeyHash> plans_;
  uint64_t fast_calls_ = 0;
  uint64_t fallback_calls_ = 0;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  py::class_<OutLaunchCache>(module, "OutLaunchCache")
    .def(py::init<py::object>())
    .def("register_plan", &OutLaunchCache::register_plan)
    .def("run", &OutLaunchCache::run)
    .def("stats", &OutLaunchCache::stats);
}
