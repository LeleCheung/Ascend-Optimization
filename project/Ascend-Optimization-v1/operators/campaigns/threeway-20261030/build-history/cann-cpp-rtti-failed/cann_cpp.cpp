// HG-KDA的CANN直接参考封装。仅复用torch_npu的张量、任务队列及aclnn生命周期工具。
// 目标计算直接调用aclnn，不调用aten/torch的目标算子。
#include <torch/extension.h>
#include "op_plugin/utils/op_api_common.h"

at::Tensor log10_inplace(at::Tensor self) {
    EXEC_NPU_CMD(aclnnInplaceLog10, self);
    return self;
}

at::Tensor hardswish_out(const at::Tensor& self, at::Tensor out) {
    EXEC_NPU_CMD(aclnnHardswish, self, out);
    return out;
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> group_norm(
    const at::Tensor& input, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias, int64_t n, int64_t c,
    int64_t hxw, int64_t groups, double eps) {
    auto out=at::empty(input.sizes(),input.options());
    auto mean=at::empty({n,groups},input.options());
    auto rstd=at::empty({n,groups},input.options());
    EXEC_NPU_CMD(aclnnGroupNorm,input,weight,bias,n,c,hxw,groups,eps,out,mean,rstd);
    return {out,mean,rstd};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,module) {
    module.def("log10_inplace",&log10_inplace);
    module.def("hardswish_out",&hardswish_out);
    module.def("group_norm",&group_norm);
}
