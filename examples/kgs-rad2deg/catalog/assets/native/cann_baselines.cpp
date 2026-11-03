// 厂商aclnn参考，PyTorch只提供分配、元数据和当前stream，不调用目标aten算子。
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_selu.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_neg.h>
#include <aclnnop/aclnn_log2.h>
#include <aclnnop/aclnn_log10.h>
#include <aclnnop/aclnn_nan_to_num.h>
#include <aclnnop/aclnn_linalg_cross.h>
#include <aclnnop/aclnn_hardshrink_backward.h>
#include <aclnnop/aclnn_sub.h>
#include <aclnnop/aclnn_round.h>
#include <aclnnop/aclnn_hardswish.h>
#include <aclnnop/aclnn_fmod_tensor.h>
#include <aclnnop/aclnn_transform_bias_rescale_qkv.h>
#include <aclnnop/aclnn_scatter_add.h>
#include <aclnnop/aclnn_upsample_bilinear_2d.h>
#include <aclnnop/aclnn_uniform.h>
#include <vector>

struct Descriptors {
    std::vector<aclTensor*> tensors;
    std::vector<aclScalar*> scalars;
    std::vector<aclIntArray*> arrays;
    ~Descriptors() {
        for (auto p:tensors) aclDestroyTensor(p);
        for (auto p:scalars) aclDestroyScalar(p);
        for (auto p:arrays) aclDestroyIntArray(p);
    }
    aclTensor* wrap(const at::Tensor& x) {
        int dt=x.scalar_type()==at::kFloat?0:x.scalar_type()==at::kHalf?1:
            x.scalar_type()==at::kInt?3:x.scalar_type()==at::kLong?9:
            x.scalar_type()==at::kBool?12:x.scalar_type()==at::kBFloat16?27:-1;
        TORCH_CHECK(dt>=0,"unsupported reference dtype");
        std::vector<int64_t> shape(x.sizes().begin(),x.sizes().end()), strides(x.strides().begin(),x.strides().end());
        int64_t span=x.numel()?1:0;
        for (size_t i=0;i<shape.size();++i) span+=(shape[i]-1)*strides[i];
        auto p=aclCreateTensor(shape.data(),shape.size(),static_cast<aclDataType>(dt),strides.data(),0,
                               ACL_FORMAT_ND,&span,1,x.data_ptr());
        TORCH_CHECK(p,"aclCreateTensor failed");tensors.push_back(p);return p;
    }
    aclScalar* scalar(double value) {
        auto p=aclCreateScalar(&value,ACL_DOUBLE);TORCH_CHECK(p,"aclCreateScalar failed");
        scalars.push_back(p);return p;
    }
    aclIntArray* ints(const std::vector<int64_t>& values) {
        auto p=aclCreateIntArray(values.data(),values.size());TORCH_CHECK(p,"aclCreateIntArray failed");
        arrays.push_back(p);return p;
    }
};
using Execute=aclnnStatus(*)(void*,uint64_t,aclOpExecutor*,aclrtStream);
void execute(Execute fn,uint64_t bytes,aclOpExecutor* ex,const at::Tensor& x,aclrtStream stream) {
    auto work=bytes?at::empty({static_cast<int64_t>(bytes)},x.options().dtype(at::kByte)):at::Tensor();
    auto rc=fn(work.defined()?work.data_ptr():nullptr,bytes,ex,stream);
    TORCH_CHECK(rc==0,"CANN execute rc=",rc);
}
#define BEGIN(x) auto stream=c10_npu::getCurrentNPUStream(x.device().index()).stream(); Descriptors d; uint64_t bytes=0; aclOpExecutor* ex=nullptr
#define END(fn,x) TORCH_CHECK(rc==0,#fn " query rc=",rc);execute(fn,bytes,ex,x,stream)

at::Tensor inplace(at::Tensor x,int op) {
    BEGIN(x);aclnnStatus rc;Execute fn;
    switch(op) {
    case 0:rc=aclnnInplaceSeluGetWorkspaceSize(d.wrap(x),&bytes,&ex);fn=aclnnInplaceSelu;break;
    case 1:rc=aclnnInplaceNegGetWorkspaceSize(d.wrap(x),&bytes,&ex);fn=aclnnInplaceNeg;break;
    case 2:rc=aclnnInplaceLog2GetWorkspaceSize(d.wrap(x),&bytes,&ex);fn=aclnnInplaceLog2;break;
    case 3:rc=aclnnInplaceLog10GetWorkspaceSize(d.wrap(x),&bytes,&ex);fn=aclnnInplaceLog10;break;
    default:TORCH_CHECK(false,"unknown unary reference");
    }
    END(fn,x);return x;
}
at::Tensor rad2deg(at::Tensor x) {
    BEGIN(x);auto rc=aclnnInplaceMulsGetWorkspaceSize(d.wrap(x),d.scalar(57.2957795130823208768),&bytes,&ex);
    END(aclnnInplaceMuls,x);return x;
}
at::Tensor nan_to_num(at::Tensor x,double nan,double posinf,double neginf) {
    BEGIN(x);auto rc=aclnnInplaceNanToNumGetWorkspaceSize(d.wrap(x),float(nan),float(posinf),float(neginf),&bytes,&ex);
    END(aclnnInplaceNanToNum,x);return x;
}
at::Tensor cross(const at::Tensor& x,const at::Tensor& other,int64_t dim,at::Tensor out) {
    BEGIN(x);auto rc=aclnnLinalgCrossGetWorkspaceSize(d.wrap(x),d.wrap(other),dim,d.wrap(out),&bytes,&ex);
    END(aclnnLinalgCross,x);return out;
}
at::Tensor shrink(const at::Tensor& grad,const at::Tensor& x,double lambd) {
    auto out=at::empty_like(x);BEGIN(x);
    auto rc=aclnnHardshrinkBackwardGetWorkspaceSize(d.wrap(grad),d.wrap(x),d.scalar(lambd),d.wrap(out),&bytes,&ex);
    END(aclnnHardshrinkBackward,x);return out;
}
at::Tensor subtract(at::Tensor x,const at::Tensor& other,double alpha) {
    BEGIN(x);auto rc=aclnnInplaceSubGetWorkspaceSize(d.wrap(x),d.wrap(other),d.scalar(alpha),&bytes,&ex);
    END(aclnnInplaceSub,x);return x;
}
at::Tensor round_out(const at::Tensor& x,int64_t decimals,at::Tensor out) {
    BEGIN(x);auto rc=aclnnRoundDecimalsGetWorkspaceSize(d.wrap(x),decimals,d.wrap(out),&bytes,&ex);
    END(aclnnRoundDecimals,x);return out;
}
at::Tensor hardswish_out(const at::Tensor& x,at::Tensor out) {
    BEGIN(x);auto rc=aclnnHardswishGetWorkspaceSize(d.wrap(x),d.wrap(out),&bytes,&ex);
    END(aclnnHardswish,x);return out;
}
at::Tensor hardswish(const at::Tensor& x) {return hardswish_out(x,at::empty_like(x));}
at::Tensor fmod_ref(at::Tensor x,const at::Tensor& other) {
    BEGIN(x);auto rc=aclnnInplaceFmodTensorGetWorkspaceSize(d.wrap(x),d.wrap(other),&bytes,&ex);
    END(aclnnInplaceFmodTensor,x);return x;
}
std::tuple<at::Tensor,at::Tensor,at::Tensor> qkv(const at::Tensor& x,const at::Tensor& bias,int64_t heads) {
    TORCH_CHECK(x.dim()==3 && heads>0 && x.size(2)%(3*heads)==0,"QKV geometry");
    std::vector<int64_t> shape{x.size(0),heads,x.size(1),x.size(2)/(3*heads)};
    auto q=at::empty(shape,x.options()), k=at::empty(shape,x.options()), v=at::empty(shape,x.options());
    BEGIN(x);auto rc=aclnnTransformBiasRescaleQkvGetWorkspaceSize(d.wrap(x),d.wrap(bias),heads,d.wrap(q),d.wrap(k),d.wrap(v),&bytes,&ex);
    END(aclnnTransformBiasRescaleQkv,x);return {q,k,v};
}
at::Tensor scatter(const at::Tensor& x,int64_t dim,const at::Tensor& index,const at::Tensor& src) {
    auto out=at::empty_like(x);BEGIN(x);
    auto rc=aclnnScatterAddGetWorkspaceSize(d.wrap(x),dim,d.wrap(index),d.wrap(src),d.wrap(out),&bytes,&ex);
    END(aclnnScatterAdd,x);return out;
}
at::Tensor upsample(const at::Tensor& x,std::vector<int64_t> size,bool align,double sh,double sw) {
    auto out=at::empty({x.size(0),x.size(1),size.at(0),size.at(1)},x.options());BEGIN(x);
    auto rc=aclnnUpsampleBilinear2dGetWorkspaceSize(d.wrap(x),d.ints(size),align,sh,sw,d.wrap(out),&bytes,&ex);
    END(aclnnUpsampleBilinear2d,x);return out;
}
at::Tensor uniform(const at::Tensor& x,double from,double to,uint64_t seed,uint64_t offset) {
    auto out=at::empty(x.sizes(),x.options());BEGIN(x);
    auto rc=aclnnInplaceUniformGetWorkspaceSize(d.wrap(out),from,to,seed,offset,&bytes,&ex);
    END(aclnnInplaceUniform,x);return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("inplace",&inplace);m.def("rad2deg",&rad2deg);m.def("nan_to_num",&nan_to_num);
    m.def("cross",&cross);m.def("shrink",&shrink);m.def("subtract",&subtract);
    m.def("round_out",&round_out);m.def("hardswish_out",&hardswish_out);m.def("hardswish",&hardswish);
    m.def("fmod",&fmod_ref);m.def("qkv",&qkv);m.def("scatter",&scatter);m.def("upsample",&upsample);
    m.def("uniform",&uniform);
}
