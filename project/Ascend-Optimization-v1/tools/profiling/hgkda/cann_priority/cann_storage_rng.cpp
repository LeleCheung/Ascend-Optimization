// 共轭：CANN Copy/Neg的显式组合；Uniform：厂商随机接口和NPU generator适配。
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/aten/NPUGeneratorImpl.h"
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_copy.h>
#include <aclnnop/aclnn_neg.h>
#include <aclnnop/aclnn_uniform.h>
#include <mutex>
#include <vector>

struct Descriptors {
    std::vector<aclTensor*> held;
    ~Descriptors(){for(auto p:held)aclDestroyTensor(p);}
    aclTensor* wrap(const at::Tensor& x,bool split=false,bool imag=false) {
        int dt=split?0:x.scalar_type()==at::kFloat?0:x.scalar_type()==at::kHalf?1:x.scalar_type()==at::kBFloat16?27:-1;
        TORCH_CHECK(dt>=0,"unsupported reference dtype");
        std::vector<int64_t> shape(x.sizes().begin(),x.sizes().end()),strides(x.strides().begin(),x.strides().end());
        if(split){for(auto& s:strides)s*=2;if(!imag){shape.push_back(2);strides.push_back(1);}}
        int64_t span=x.numel()?1:0;for(size_t i=0;i<shape.size();++i)span+=(shape[i]-1)*strides[i];
        void* pointer=static_cast<char*>(x.data_ptr())+(imag?sizeof(float):0);
        auto p=aclCreateTensor(shape.data(),shape.size(),static_cast<aclDataType>(dt),strides.data(),0,ACL_FORMAT_ND,&span,1,pointer);
        TORCH_CHECK(p,"aclCreateTensor failed");held.push_back(p);return p;
    }
};
using Execute=aclnnStatus(*)(void*,uint64_t,aclOpExecutor*,aclrtStream);
void execute(Execute fn,uint64_t bytes,aclOpExecutor* ex,const at::Tensor& x,aclrtStream stream) {
    auto work=bytes?at::empty({static_cast<int64_t>(bytes)},x.options().dtype(at::kByte)):at::Tensor();
    auto rc=fn(work.defined()?work.data_ptr():nullptr,bytes,ex,stream);TORCH_CHECK(rc==0,"CANN execute rc=",rc);
}
at::Tensor conj_physical(const at::Tensor& input) {
    TORCH_CHECK(!input.is_neg(),"negative-bit view not supported by this reference adapter");
    bool complex=input.scalar_type()==at::kComplexFloat;
    TORCH_CHECK(complex || !input.is_complex(),"only complex64 supported");
    auto out=at::empty(input.sizes(),input.options());
    if(!input.numel())return out;
    auto stream=c10_npu::getCurrentNPUStream(input.device().index()).stream();
    if(input.is_contiguous()) {
        auto rc=aclrtMemcpyAsync(out.data_ptr(),out.nbytes(),input.data_ptr(),input.nbytes(),ACL_MEMCPY_DEVICE_TO_DEVICE,stream);
        TORCH_CHECK(rc==0,"CANN memcpy rc=",rc);
    } else {
        Descriptors d;uint64_t bytes=0;aclOpExecutor* ex=nullptr;
        auto rc=aclnnInplaceCopyGetWorkspaceSize(d.wrap(out,complex),d.wrap(input,complex),&bytes,&ex);
        TORCH_CHECK(rc==0,"CANN copy query rc=",rc);execute(aclnnInplaceCopy,bytes,ex,input,stream);
    }
    if(complex && !input.is_conj()) {
        Descriptors d;uint64_t bytes=0;aclOpExecutor* ex=nullptr;
        auto rc=aclnnInplaceNegGetWorkspaceSize(d.wrap(out,true,true),&bytes,&ex);
        TORCH_CHECK(rc==0,"CANN imag neg query rc=",rc);execute(aclnnInplaceNeg,bytes,ex,input,stream);
    }
    return out;
}
at::Tensor uniform(const at::Tensor& x,double from,double to,c10::optional<at::Generator> generator) {
    TORCH_CHECK(from<=to,"uniform requires from<=to");
    auto out=at::empty(x.sizes(),x.options());
    if(!out.numel())return out;
    auto* gen=at::get_generator_or_default<at_npu::NPUGeneratorImpl>(generator,at_npu::detail::getDefaultNPUGenerator(x.device().index()));
    std::pair<uint64_t,uint64_t> state;
    {
        std::lock_guard<std::mutex> guard(gen->mutex_);
        state=gen->philox_engine_inputs(10); // CANN旧随机模式每调用保留10计数，生成器按4取整。
    }
    auto stream=c10_npu::getCurrentNPUStream(x.device().index()).stream();
    Descriptors d;uint64_t bytes=0;aclOpExecutor* ex=nullptr;
    auto rc=aclnnInplaceUniformGetWorkspaceSize(d.wrap(out),from,to,state.first,state.second,&bytes,&ex);
    TORCH_CHECK(rc==0,"CANN uniform query rc=",rc);execute(aclnnInplaceUniform,bytes,ex,x,stream);return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("conj_physical",&conj_physical);m.def("uniform",&uniform);}
