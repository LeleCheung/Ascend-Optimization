// 仅用于参考资格探测：直接调用厂商Tensor seed/offset接口，不接入Triton候选。
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_uniform.h>
#include <vector>

struct Descriptor {
    aclTensor* value;
    explicit Descriptor(const at::Tensor& x) {
        auto dtype=x.scalar_type()==at::kFloat?ACL_FLOAT:
            x.scalar_type()==at::kHalf?ACL_FLOAT16:
            x.scalar_type()==at::kBFloat16?ACL_BF16:
            x.scalar_type()==at::kUInt64?ACL_UINT64:ACL_DT_UNDEFINED;
        TORCH_CHECK(dtype!=ACL_DT_UNDEFINED,"unsupported dtype");
        std::vector<int64_t> shape(x.sizes().begin(),x.sizes().end()),strides(x.strides().begin(),x.strides().end());
        int64_t span=x.numel()?1:0;
        for(size_t i=0;i<shape.size();++i)span+=(shape[i]-1)*strides[i];
        value=aclCreateTensor(shape.data(),shape.size(),dtype,strides.data(),0,ACL_FORMAT_ND,&span,1,x.data_ptr());
        TORCH_CHECK(value,"aclCreateTensor failed");
    }
    ~Descriptor(){aclDestroyTensor(value);}
};

at::Tensor uniform_tensor(const at::Tensor& output,double from,double to,
                         const at::Tensor& seed,const at::Tensor& offset,uint64_t scalar_offset) {
    TORCH_CHECK(output.device()==seed.device() && output.device()==offset.device(),"device mismatch");
    Descriptor out(output),s(seed),o(offset);
    uint64_t bytes=0;aclOpExecutor* executor=nullptr;
    auto rc=aclnnInplaceUniformTensorGetWorkspaceSize(out.value,from,to,s.value,o.value,scalar_offset,&bytes,&executor);
    TORCH_CHECK(rc==0,"aclnnInplaceUniformTensor query rc=",rc);
    auto work=bytes?at::empty({static_cast<int64_t>(bytes)},output.options().dtype(at::kByte)):at::Tensor();
    auto stream=c10_npu::getCurrentNPUStream(output.device().index()).stream();
    rc=aclnnInplaceUniformTensor(work.defined()?work.data_ptr():nullptr,bytes,executor,stream);
    TORCH_CHECK(rc==0,"aclnnInplaceUniformTensor execute rc=",rc);
    return output;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("uniform_tensor",&uniform_tensor);}
