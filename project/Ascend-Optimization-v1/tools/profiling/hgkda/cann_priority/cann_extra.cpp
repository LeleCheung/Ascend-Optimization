// CANN补充接口：两种XLogY直接API；归约梯度和累计积分明确标注组合。
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_x_log_y_scalar_self.h>
#include <aclnnop/aclnn_x_log_y_tensor.h>
#include <aclnnop/aclnn_scatter.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_cumsum.h>
#include <vector>

struct Descriptors {
    std::vector<aclTensor*> tensors;
    std::vector<aclScalar*> scalars;
    ~Descriptors(){for(auto p:tensors)aclDestroyTensor(p);for(auto p:scalars)aclDestroyScalar(p);}
    aclTensor* wrap(const at::Tensor& x) {
        int dt=x.scalar_type()==at::kFloat?0:x.scalar_type()==at::kHalf?1:
            x.scalar_type()==at::kInt?3:x.scalar_type()==at::kLong?9:x.scalar_type()==at::kBFloat16?27:-1;
        TORCH_CHECK(dt>=0,"unsupported reference dtype");
        std::vector<int64_t> shape(x.sizes().begin(),x.sizes().end()),strides(x.strides().begin(),x.strides().end());
        int64_t span=x.numel()?1:0;for(size_t i=0;i<shape.size();++i)span+=(shape[i]-1)*strides[i];
        auto p=aclCreateTensor(shape.data(),shape.size(),static_cast<aclDataType>(dt),strides.data(),0,ACL_FORMAT_ND,&span,1,x.data_ptr());
        TORCH_CHECK(p,"aclCreateTensor failed");tensors.push_back(p);return p;
    }
    aclScalar* scalar(double value){auto p=aclCreateScalar(&value,ACL_DOUBLE);TORCH_CHECK(p,"aclCreateScalar failed");scalars.push_back(p);return p;}
};
using Execute=aclnnStatus(*)(void*,uint64_t,aclOpExecutor*,aclrtStream);
void execute(Execute fn,uint64_t bytes,aclOpExecutor* ex,const at::Tensor& x,aclrtStream stream) {
    auto work=bytes?at::empty({static_cast<int64_t>(bytes)},x.options().dtype(at::kByte)):at::Tensor();
    auto rc=fn(work.defined()?work.data_ptr():nullptr,bytes,ex,stream);TORCH_CHECK(rc==0,"CANN execute rc=",rc);
}
at::Tensor scalar_xlogy(double value,const at::Tensor& x) {
    auto out=at::empty_like(x);auto stream=c10_npu::getCurrentNPUStream(x.device().index()).stream();
    Descriptors d;uint64_t bytes=0;aclOpExecutor* ex=nullptr;
    auto rc=aclnnXLogYScalarSelfGetWorkspaceSize(d.scalar(value),d.wrap(x),d.wrap(out),&bytes,&ex);
    TORCH_CHECK(rc==0,"aclnnXLogYScalarSelf query rc=",rc);execute(aclnnXLogYScalarSelf,bytes,ex,x,stream);return out;
}
at::Tensor tensor_xlogy(at::Tensor x,const at::Tensor& other) {
    auto stream=c10_npu::getCurrentNPUStream(x.device().index()).stream();
    Descriptors d;uint64_t bytes=0;aclOpExecutor* ex=nullptr;
    auto rc=aclnnInplaceXLogYTensorGetWorkspaceSize(d.wrap(x),d.wrap(other),&bytes,&ex);
    TORCH_CHECK(rc==0,"aclnnInplaceXLogYTensor query rc=",rc);execute(aclnnInplaceXLogYTensor,bytes,ex,x,stream);return x;
}
at::Tensor value_backward(const at::Tensor& grad,int64_t dim,const at::Tensor& indices,std::vector<int64_t> sizes,bool keepdim) {
    int64_t rank=sizes.size();if(dim<0)dim+=rank;
    auto g=keepdim?grad:grad.unsqueeze(dim), index=keepdim?indices:indices.unsqueeze(dim);
    auto base=at::zeros(sizes,grad.options()),out=at::empty(sizes,grad.options());
    auto stream=c10_npu::getCurrentNPUStream(grad.device().index()).stream();
    Descriptors d;uint64_t bytes=0;aclOpExecutor* ex=nullptr;
    auto rc=aclnnScatterGetWorkspaceSize(d.wrap(base),dim,d.wrap(index),d.wrap(g),0,d.wrap(out),&bytes,&ex);
    TORCH_CHECK(rc==0,"aclnnScatter query rc=",rc);execute(aclnnScatter,bytes,ex,grad,stream);return out;
}
at::Tensor trapezoid(const at::Tensor& y,double dx,int64_t dim) {
    if(dim<0)dim+=y.dim();int64_t n=y.size(dim)-1;
    TORCH_CHECK(n>=0,"scan input must be nonempty");
    auto left=y.narrow(dim,0,n),right=y.narrow(dim,1,n);
    auto sum=at::empty(left.sizes(),y.options()),scaled=at::empty(left.sizes(),y.options()),out=at::empty(left.sizes(),y.options());
    if(n==0)return out;
    auto stream=c10_npu::getCurrentNPUStream(y.device().index()).stream();
    {
        Descriptors d;uint64_t bytes=0;aclOpExecutor* ex=nullptr;
        auto rc=aclnnAddGetWorkspaceSize(d.wrap(left),d.wrap(right),d.scalar(1.0),d.wrap(sum),&bytes,&ex);
        TORCH_CHECK(rc==0,"aclnnAdd query rc=",rc);execute(aclnnAdd,bytes,ex,y,stream);
    }
    {
        Descriptors d;uint64_t bytes=0;aclOpExecutor* ex=nullptr;
        auto rc=aclnnMulsGetWorkspaceSize(d.wrap(sum),d.scalar(dx/2.0),d.wrap(scaled),&bytes,&ex);
        TORCH_CHECK(rc==0,"aclnnMuls query rc=",rc);execute(aclnnMuls,bytes,ex,y,stream);
    }
    {
        Descriptors d;uint64_t bytes=0;aclOpExecutor* ex=nullptr;
        aclDataType dt=y.scalar_type()==at::kFloat?ACL_FLOAT:y.scalar_type()==at::kHalf?ACL_FLOAT16:ACL_BF16;
        auto rc=aclnnCumsumGetWorkspaceSize(d.wrap(scaled),dim,dt,d.wrap(out),&bytes,&ex);
        TORCH_CHECK(rc==0,"aclnnCumsum query rc=",rc);execute(aclnnCumsum,bytes,ex,y,stream);
    }
    return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("scalar_xlogy",&scalar_xlogy);m.def("tensor_xlogy",&tensor_xlogy);
    m.def("value_backward",&value_backward);m.def("trapezoid",&trapezoid);
}
