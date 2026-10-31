// CANN公开算子API参考。PyTorch仅提供张量、分配器与同stream宿主任务队列。
// 不调用aten目标算子，不复用executor，不额外全局同步。
#include <torch/extension.h>
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_log10.h>
#include <aclnnop/aclnn_hardswish.h>
#include <aclnnop/aclnn_group_norm.h>
#include <memory>
#include <vector>

struct Tensors {
    std::vector<aclTensor*> descriptors;
    std::vector<at::Tensor> held;
    ~Tensors() { for (auto* tensor:descriptors) aclDestroyTensor(tensor); }
    aclTensor* wrap(const at::Tensor& input) {
        int dtype=input.scalar_type()==at::kFloat ? 0 : input.scalar_type()==at::kHalf ? 1 : input.scalar_type()==at::kBFloat16 ? 27 : -1;
        TORCH_CHECK(dtype>=0,"unsupported reference dtype");
        std::vector<int64_t> shape(input.sizes().begin(),input.sizes().end());
        std::vector<int64_t> strides(input.strides().begin(),input.strides().end());
        int64_t span=input.numel()?1:0;
        for (size_t i=0;i<shape.size();++i) span+=(shape[i]-1)*strides[i];
        auto* descriptor=aclCreateTensor(shape.data(),shape.size(),static_cast<aclDataType>(dtype),
            strides.data(),0,ACL_FORMAT_ND,&span,1,input.data_ptr());
        TORCH_CHECK(descriptor,"aclCreateTensor failed");
        descriptors.push_back(descriptor);held.push_back(input);return descriptor;
    }
};

using Execute=aclnnStatus(*)(void*,uint64_t,aclOpExecutor*,aclrtStream);

void dispatch(const char* name,std::shared_ptr<Tensors> tensors,aclOpExecutor* executor,
              uint64_t bytes,const at::Tensor& like,Execute execute) {
    auto stream=c10_npu::getCurrentNPUStream(like.device().index());
    // 分配发生在调用者当前stream；回调持有工作区至runtime提交完成，分配器管理设备生命周期。
    at::Tensor workspace;
    if (bytes) workspace=at::empty({static_cast<int64_t>(bytes)},like.options().dtype(at::kByte));
    auto callback=[tensors,executor,bytes,workspace,execute,stream]() -> int {
        auto code=execute(workspace.defined()?workspace.data_ptr():nullptr,bytes,executor,stream.stream(false));
        TORCH_CHECK(code==0,"CANN execute failed: ",code);
        // 与厂商OpApi宏一致：execute返回后释放host描述符，不等待设备任务完成。
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2(name,callback);
}

at::Tensor log10_inplace(at::Tensor self) {
    auto tensors=std::make_shared<Tensors>();
    uint64_t bytes=0;aclOpExecutor* executor=nullptr;
    auto code=aclnnInplaceLog10GetWorkspaceSize(tensors->wrap(self),&bytes,&executor);
    TORCH_CHECK(code==0,"aclnnInplaceLog10GetWorkspaceSize failed: ",code);
    dispatch("aclnnInplaceLog10",tensors,executor,bytes,self,aclnnInplaceLog10);
    return self;
}

at::Tensor hardswish_out(const at::Tensor& self,at::Tensor out) {
    auto tensors=std::make_shared<Tensors>();
    uint64_t bytes=0;aclOpExecutor* executor=nullptr;
    auto code=aclnnHardswishGetWorkspaceSize(tensors->wrap(self),tensors->wrap(out),&bytes,&executor);
    TORCH_CHECK(code==0,"aclnnHardswishGetWorkspaceSize failed: ",code);
    dispatch("aclnnHardswish",tensors,executor,bytes,self,aclnnHardswish);
    return out;
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> group_norm(
    const at::Tensor& input,const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,int64_t n,int64_t c,
    int64_t hxw,int64_t groups,double eps) {
    // RunOpApiV2闭包持有新输出及工作区；连续暖机时，先提交前次宿主任务，
    // 避免大量尚未执行的闭包同时持有整张量。不调用设备synchronize。
    c10_npu::getCurrentNPUStream(input.device().index()).stream();
    auto out=at::empty(input.sizes(),input.options());
    auto mean=at::empty({n,groups},input.options());
    auto rstd=at::empty({n,groups},input.options());
    auto tensors=std::make_shared<Tensors>();
    uint64_t bytes=0;aclOpExecutor* executor=nullptr;
    auto code=aclnnGroupNormGetWorkspaceSize(tensors->wrap(input),weight?tensors->wrap(*weight):nullptr,
        bias?tensors->wrap(*bias):nullptr,n,c,hxw,groups,eps,tensors->wrap(out),tensors->wrap(mean),tensors->wrap(rstd),&bytes,&executor);
    TORCH_CHECK(code==0,"aclnnGroupNormGetWorkspaceSize failed: ",code);
    dispatch("aclnnGroupNorm",tensors,executor,bytes,input,aclnnGroupNorm);
    return {out,mean,rstd};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,module) {
    module.def("log10_inplace",&log10_inplace);
    module.def("hardswish_out",&hardswish_out);
    module.def("group_norm",&group_norm);
}
