// 厂商扫描桥：只接收本次增量张量，不缓存数据、executor或stream。
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_cumsum.h>
#include <vector>

struct Descriptor {
    aclTensor* value=nullptr;
    explicit Descriptor(const at::Tensor& x) {
        TORCH_CHECK(x.is_contiguous() && x.storage_offset()==0,"扫描桥要求独立连续增量张量");
        int dtype=x.scalar_type()==at::kFloat?0:x.scalar_type()==at::kHalf?1:x.scalar_type()==at::kBFloat16?27:-1;
        TORCH_CHECK(dtype>=0,"unsupported dtype");
        std::vector<int64_t> shape(x.sizes().begin(),x.sizes().end()),stride(x.strides().begin(),x.strides().end());
        value=aclCreateTensor(shape.data(),shape.size(),static_cast<aclDataType>(dtype),stride.data(),0,ACL_FORMAT_ND,shape.data(),shape.size(),x.data_ptr());
        TORCH_CHECK(value,"aclCreateTensor failed");
    }
    ~Descriptor(){if(value)aclDestroyTensor(value);}
};

at::Tensor cumsum_ref(const at::Tensor& x,int64_t axis) {
    auto out=at::empty_like(x);Descriptor input(x),output(out);
    int dtype=x.scalar_type()==at::kFloat?0:x.scalar_type()==at::kHalf?1:27;
    uint64_t bytes=0;aclOpExecutor* executor=nullptr;
    auto rc=aclnnCumsumGetWorkspaceSize(input.value,axis,static_cast<aclDataType>(dtype),output.value,&bytes,&executor);
    TORCH_CHECK(rc==0,"aclnnCumsum query rc=",rc);
    auto workspace=bytes?at::empty({static_cast<int64_t>(bytes)},x.options().dtype(at::kByte)):at::Tensor();
    auto stream=c10_npu::getCurrentNPUStream(x.device().index()).stream();
    rc=aclnnCumsum(workspace.defined()?workspace.data_ptr():nullptr,bytes,executor,stream);
    TORCH_CHECK(rc==0,"aclnnCumsum execute rc=",rc);return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("cumsum",&cumsum_ref);}
