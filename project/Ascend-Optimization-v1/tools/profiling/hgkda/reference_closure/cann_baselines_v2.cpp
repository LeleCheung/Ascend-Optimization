// 独立CANN参考；不调用PyTorch目标算子，不用厂商实现冒充HG-KDA候选。
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_trunc.h>
#include <aclnnop/aclnn_expm1.h>
#include <aclnnop/aclnn_inverse.h>
#include <aclnnop/aclnn_batch_norm.h>
#include <aclnnop/aclnn_fake_quant_per_tensor_affine_cachemask.h>
#include <vector>

struct Descriptors {
    std::vector<aclTensor*> descriptors;
    std::vector<at::Tensor> held;
    ~Descriptors(){for(auto* x:descriptors) aclDestroyTensor(x);}
    aclTensor* wrap(const at::Tensor& x){
        int dt=x.scalar_type()==at::kFloat?0:x.scalar_type()==at::kHalf?1:
            x.scalar_type()==at::kInt?3:x.scalar_type()==at::kLong?9:
            x.scalar_type()==at::kBool?12:x.scalar_type()==at::kBFloat16?27:-1;
        TORCH_CHECK(dt>=0,"unsupported CANN reference dtype");
        std::vector<int64_t> shape(x.sizes().begin(),x.sizes().end()),stride(x.strides().begin(),x.strides().end());
        int64_t span=x.numel()?1:0;
        for(size_t i=0;i<shape.size();++i)span+=(shape[i]-1)*stride[i];
        auto* value=aclCreateTensor(shape.data(),shape.size(),static_cast<aclDataType>(dt),stride.data(),0,ACL_FORMAT_ND,&span,1,x.data_ptr());
        TORCH_CHECK(value,"aclCreateTensor failed");descriptors.push_back(value);held.push_back(x);return value;
    }
};
using Execute=aclnnStatus(*)(void*,uint64_t,aclOpExecutor*,aclrtStream);
void execute(Execute fn,uint64_t bytes,aclOpExecutor* executor,const at::Tensor& like,aclrtStream stream){
    auto workspace=bytes?at::empty({static_cast<int64_t>(bytes)},like.options().dtype(at::kByte)):at::Tensor();
    auto rc=fn(workspace.defined()?workspace.data_ptr():nullptr,bytes,executor,stream);
    TORCH_CHECK(rc==0,"CANN execute rc=",rc);
}
at::Tensor trunc_ref(const at::Tensor& x){
    auto out=at::empty_like(x);auto stream=c10_npu::getCurrentNPUStream(x.device().index()).stream();
    Descriptors d;uint64_t bytes=0;aclOpExecutor* ex=nullptr;
    auto rc=aclnnTruncGetWorkspaceSize(d.wrap(x),d.wrap(out),&bytes,&ex);
    TORCH_CHECK(rc==0,"aclnnTrunc query rc=",rc);execute(aclnnTrunc,bytes,ex,x,stream);return out;
}
at::Tensor expm1_ref(at::Tensor x){
    auto out=x.scalar_type()==at::kBFloat16?at::empty_like(x):x;
    auto stream=c10_npu::getCurrentNPUStream(x.device().index()).stream();
    Descriptors d;uint64_t bytes=0;aclOpExecutor* ex=nullptr;
    auto rc=x.scalar_type()==at::kBFloat16?aclnnExpm1GetWorkspaceSize(d.wrap(x),d.wrap(out),&bytes,&ex):
        aclnnInplaceExpm1GetWorkspaceSize(d.wrap(x),&bytes,&ex);
    TORCH_CHECK(rc==0,"CANN Expm1 query rc=",rc);
    execute(x.scalar_type()==at::kBFloat16?aclnnExpm1:aclnnInplaceExpm1,bytes,ex,x,stream);
    if(x.scalar_type()==at::kBFloat16)x.copy_(out); // BF16原地API未声明支持，明确计入NPU复制。
    return x;
}
at::Tensor inverse_ref(const at::Tensor& x){
    auto out=at::empty_like(x);auto stream=c10_npu::getCurrentNPUStream(x.device().index()).stream();
    Descriptors d;uint64_t bytes=0;aclOpExecutor* ex=nullptr;
    auto rc=aclnnInverseGetWorkspaceSize(d.wrap(x),d.wrap(out),&bytes,&ex);
    TORCH_CHECK(rc==0,"aclnnInverse query rc=",rc);execute(aclnnInverse,bytes,ex,x,stream);return out;
}
std::tuple<at::Tensor,at::Tensor> fake_quant_ref(const at::Tensor& x,double scale,int64_t zp,int64_t qmin,int64_t qmax,at::Tensor out,at::Tensor mask){
    TORCH_CHECK(x.scalar_type()==at::kHalf||x.scalar_type()==at::kFloat||x.scalar_type()==at::kBFloat16,"unsupported FakeQuant dtype");
    auto value=x.scalar_type()!=at::kFloat?x.to(at::kFloat):x;
    auto result=x.scalar_type()!=at::kFloat?at::empty_like(value):out;
    auto scale_tensor=at::full({1},scale,value.options());auto zero=at::full({1},zp,x.options().dtype(at::kInt));
    auto stream=c10_npu::getCurrentNPUStream(x.device().index()).stream();
    Descriptors d;uint64_t bytes=0;aclOpExecutor* ex=nullptr;
    auto rc=aclnnFakeQuantPerTensorAffineCachemaskGetWorkspaceSize(d.wrap(value),d.wrap(scale_tensor),d.wrap(zero),1.f,qmin,qmax,d.wrap(result),d.wrap(mask),&bytes,&ex);
    TORCH_CHECK(rc==0,"CANN FakeQuant query rc=",rc);execute(aclnnFakeQuantPerTensorAffineCachemask,bytes,ex,x,stream);
    if(x.scalar_type()!=at::kFloat)out.copy_(result); // 明确登记BF16转FP32与写回，非厂商原生BF16单算子。
    return {out,mask};
}
std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> batch_norm_ref(
    const at::Tensor& x,const c10::optional<at::Tensor>& weight,const c10::optional<at::Tensor>& bias,
    const at::Tensor& running_mean,const at::Tensor& running_var,double momentum,double eps){
    auto rm=running_mean.to(at::kFloat).clone(),rv=running_var.to(at::kFloat).clone();
    auto value=x.to(at::kFloat);
    std::vector<int64_t> dimensions(value.sizes().begin(),value.sizes().end());
    while(dimensions.size()<4)dimensions.push_back(1);
    auto view=value.reshape(dimensions);
    auto output=at::empty_like(view);
    auto mean=at::empty({x.size(1)},x.options().dtype(at::kFloat));auto invstd=at::empty_like(mean);
    auto w=weight?weight->to(at::kFloat):at::ones({x.size(1)},x.options().dtype(at::kFloat));
    auto bias_value=bias?bias->to(at::kFloat):at::zeros({x.size(1)},x.options().dtype(at::kFloat));
    auto reserve=at::empty({0},x.options().dtype(at::kByte));
    auto stream=c10_npu::getCurrentNPUStream(x.device().index()).stream();
    Descriptors d;uint64_t bytes=0;aclOpExecutor* ex=nullptr;
    auto rc=aclnnBatchNormGetWorkspaceSize(d.wrap(view),d.wrap(w),d.wrap(bias_value),
        d.wrap(rm),d.wrap(rv),true,momentum,eps,d.wrap(output),d.wrap(mean),d.wrap(invstd),&bytes,&ex);
    TORCH_CHECK(rc==0,"CANN BatchNorm query rc=",rc);execute(aclnnBatchNorm,bytes,ex,x,stream);
    auto out=output.reshape(x.sizes()).to(x.scalar_type());
    mean=mean.to(x.scalar_type());invstd=invstd.to(x.scalar_type());
    rm=rm.to(running_mean.scalar_type());rv=rv.to(running_var.scalar_type());
    return {out,mean,invstd,reserve,rm,rv};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
    m.def("trunc",&trunc_ref);m.def("expm1_",&expm1_ref);m.def("inverse_only",&inverse_ref);
    m.def("fake_quant",&fake_quant_ref);m.def("batch_norm",&batch_norm_ref);
}
