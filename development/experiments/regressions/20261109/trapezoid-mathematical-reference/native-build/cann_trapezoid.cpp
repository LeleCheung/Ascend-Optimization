// 参考实现：CANN Add、Muls、FP32 Cumsum及必要的Cast。
// 元素加法和缩放保留输入类型舍入；扫描使用FP32累加，避免FP16前缀和误差。
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_cumsum.h>
#include <aclnnop/aclnn_cast.h>
#include <vector>

aclDataType acl_dtype(at::ScalarType dtype) {
    if (dtype==at::kFloat) return ACL_FLOAT;
    if (dtype==at::kHalf) return ACL_FLOAT16;
    if (dtype==at::kBFloat16) return ACL_BF16;
    TORCH_CHECK(false,"unsupported reference dtype");
}

struct Descriptors {
    std::vector<aclTensor*> tensors;
    std::vector<aclScalar*> scalars;
    ~Descriptors() {
        for (auto value:tensors) aclDestroyTensor(value);
        for (auto value:scalars) aclDestroyScalar(value);
    }
    aclTensor* tensor(const at::Tensor& x) {
        std::vector<int64_t> shape(x.sizes().begin(),x.sizes().end());
        std::vector<int64_t> strides(x.strides().begin(),x.strides().end());
        int64_t span=x.numel()?1:0;
        for (size_t dim=0;dim<shape.size();++dim) span+=(shape[dim]-1)*strides[dim];
        auto value=aclCreateTensor(shape.data(),shape.size(),acl_dtype(x.scalar_type()),
                    strides.data(),0,ACL_FORMAT_ND,&span,1,x.data_ptr());
        TORCH_CHECK(value,"aclCreateTensor failed");
        tensors.push_back(value);
        return value;
    }
    aclScalar* scalar(double value) {
        auto result=aclCreateScalar(&value,ACL_DOUBLE);
        TORCH_CHECK(result,"aclCreateScalar failed");
        scalars.push_back(result);
        return result;
    }
};

using Execute=aclnnStatus(*)(void*,uint64_t,aclOpExecutor*,aclrtStream);
void execute(Execute function,uint64_t bytes,aclOpExecutor* executor,
             const at::Tensor& tensor,aclrtStream stream) {
    auto workspace=bytes?at::empty({static_cast<int64_t>(bytes)},tensor.options().dtype(at::kByte)):at::Tensor();
    auto code=function(workspace.defined()?workspace.data_ptr():nullptr,bytes,executor,stream);
    TORCH_CHECK(code==0,"CANN execute rc=",code);
}

at::Tensor trapezoid(const at::Tensor& y,double dx,int64_t dim) {
    TORCH_CHECK(y.dim()>0 && dim>=-y.dim() && dim<y.dim(),"dimension out of range");
    if (dim<0) dim+=y.dim();
    acl_dtype(y.scalar_type());
    auto shape=y.sizes().vec();
    const auto length=y.size(dim)>0?y.size(dim)-1:0;
    shape[dim]=length;
    auto out=at::empty(shape,y.options());
    if (out.numel()==0) return out;
    auto left=y.narrow(dim,0,length),right=y.narrow(dim,1,length);
    auto sum=at::empty(shape,y.options()),scaled=at::empty(shape,y.options());
    auto accum=y.scalar_type()==at::kFloat?out:at::empty(shape,y.options().dtype(at::kFloat));
    auto stream=c10_npu::getCurrentNPUStream(y.device().index()).stream();
    {
        Descriptors d;uint64_t bytes=0;aclOpExecutor* executor=nullptr;
        auto code=aclnnAddGetWorkspaceSize(d.tensor(left),d.tensor(right),d.scalar(1.),
                                           d.tensor(sum),&bytes,&executor);
        TORCH_CHECK(code==0,"aclnnAdd query rc=",code);
        execute(aclnnAdd,bytes,executor,y,stream);
    }
    {
        Descriptors d;uint64_t bytes=0;aclOpExecutor* executor=nullptr;
        auto code=aclnnMulsGetWorkspaceSize(d.tensor(sum),d.scalar(dx/2.),d.tensor(scaled),&bytes,&executor);
        TORCH_CHECK(code==0,"aclnnMuls query rc=",code);
        execute(aclnnMuls,bytes,executor,y,stream);
    }
    {
        Descriptors d;uint64_t bytes=0;aclOpExecutor* executor=nullptr;
        auto code=aclnnCumsumGetWorkspaceSize(d.tensor(scaled),dim,ACL_FLOAT,d.tensor(accum),&bytes,&executor);
        TORCH_CHECK(code==0,"aclnnCumsum query rc=",code);
        execute(aclnnCumsum,bytes,executor,y,stream);
    }
    if (y.scalar_type()!=at::kFloat) {
        Descriptors d;uint64_t bytes=0;aclOpExecutor* executor=nullptr;
        auto code=aclnnCastGetWorkspaceSize(d.tensor(accum),acl_dtype(y.scalar_type()),d.tensor(out),&bytes,&executor);
        TORCH_CHECK(code==0,"aclnnCast query rc=",code);
        execute(aclnnCast,bytes,executor,y,stream);
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,module) {
    module.def("trapezoid",&trapezoid);
}
