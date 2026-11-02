// 图像接口使用NCHW存储描述；只调用CANN插值，非连续输入的复制成本计入参考。
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_upsample_bilinear_2d.h>
struct TensorDesc {
    aclTensor* value;
    TensorDesc(const at::Tensor& x) {
        TORCH_CHECK(x.is_contiguous() && x.dim()==4,"NCHW contiguous input required");
        int dtype=x.scalar_type()==at::kFloat?0:x.scalar_type()==at::kHalf?1:x.scalar_type()==at::kBFloat16?27:-1;
        TORCH_CHECK(dtype>=0,"unsupported dtype");
        value=aclCreateTensor(x.sizes().data(),4,static_cast<aclDataType>(dtype),x.strides().data(),0,
            ACL_FORMAT_NCHW,x.sizes().data(),4,x.data_ptr());
        TORCH_CHECK(value,"aclCreateTensor failed");
    }
    ~TensorDesc(){aclDestroyTensor(value);}
};
at::Tensor run(const at::Tensor& input,std::vector<int64_t> size,bool align,double sh,double sw) {
    TORCH_CHECK(input.device().type()==c10::DeviceType::PrivateUse1 && input.dim()==4 && size.size()==2,"expected NPU NCHW");
    auto x=input.contiguous();auto out=at::empty({x.size(0),x.size(1),size[0],size[1]},x.options());
    auto stream=c10_npu::getCurrentNPUStream(x.device().index()).stream();
    TensorDesc a(x),b(out);auto array=aclCreateIntArray(size.data(),size.size());
    uint64_t bytes=0;aclOpExecutor* executor=nullptr;
    auto rc=aclnnUpsampleBilinear2dGetWorkspaceSize(a.value,array,align,sh,sw,b.value,&bytes,&executor);
    aclDestroyIntArray(array);
    TORCH_CHECK(rc==0,"CANN upsample NCHW query rc=",rc," ",aclGetRecentErrMsg());
    auto work=bytes?at::empty({static_cast<int64_t>(bytes)},x.options().dtype(at::kByte)):at::Tensor();
    rc=aclnnUpsampleBilinear2d(work.defined()?work.data_ptr():nullptr,bytes,executor,stream);
    TORCH_CHECK(rc==0,"CANN upsample NCHW execute rc=",rc," ",aclGetRecentErrMsg());
    return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);}
