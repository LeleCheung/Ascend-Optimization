// 仅作CANN参考资格诊断；不进入任何HG-KDA Triton候选。
#include <torch/extension.h>
#include "op_plugin/utils/op_api_common_base.h"
#include <aclnnop/aclnn_foreach_expm1.h>
#include <array>

// 复用官方元数据转换和释放，避免自行推测aclTensorList的所有权。
struct TensorListOwner {
  aclTensorList* value;
  explicit TensorListOwner(const at::TensorList& tensors) : value(ConvertType(tensors)) {
    TORCH_CHECK(value, "CANN tensor-list conversion failed");
  }
  ~TensorListOwner() { Release(value); }
  TensorListOwner(const TensorListOwner&) = delete;
  TensorListOwner& operator=(const TensorListOwner&) = delete;
};

at::Tensor foreach_expm1_reference(at::Tensor input, bool alias_output) {
  TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1, "NPU input required");
  auto output = alias_output ? input : at::empty_like(input);
  const std::array<at::Tensor, 1> inputs{input}, outputs{output};
  TensorListOwner x(at::TensorList(inputs.data(), inputs.size()));
  TensorListOwner y(at::TensorList(outputs.data(), outputs.size()));
  uint64_t bytes = 0;
  aclOpExecutor* executor = nullptr;
  auto rc = aclnnForeachExpm1GetWorkspaceSize(x.value, y.value, &bytes, &executor);
  TORCH_CHECK(rc == 0, "aclnnForeachExpm1 query rc=", rc);
  auto workspace = bytes ? at::empty({static_cast<int64_t>(bytes)}, input.options().dtype(at::kByte))
                         : at::Tensor();
  auto stream = c10_npu::getCurrentNPUStream(input.device().index()).stream();
  rc = aclnnForeachExpm1(workspace.defined() ? workspace.data_ptr() : nullptr, bytes, executor, stream);
  TORCH_CHECK(rc == 0, "aclnnForeachExpm1 execute rc=", rc);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("expm1", &foreach_expm1_reference);
}
