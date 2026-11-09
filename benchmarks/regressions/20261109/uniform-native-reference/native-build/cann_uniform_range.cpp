// 参考实现：厂商 Uniform 与 Clamp 的同语义组合，修复低精度输出的半开区间边界。
// PyTorch 仅承担分配、生成器状态与当前 stream；不计算目标设备算子。
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/aten/NPUGeneratorImpl.h"
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_uniform.h>
#include <aclnnop/aclnn_clamp.h>
#include <cmath>
#include <limits>
#include <mutex>
#include <vector>

struct Descriptors {
    std::vector<aclTensor*> tensors;
    std::vector<aclScalar*> scalars;
    ~Descriptors() {
        for (auto value:tensors) aclDestroyTensor(value);
        for (auto value:scalars) aclDestroyScalar(value);
    }
    aclTensor* tensor(const at::Tensor& x) {
        auto dtype=x.scalar_type()==at::kFloat?ACL_FLOAT:
            x.scalar_type()==at::kHalf?ACL_FLOAT16:
            x.scalar_type()==at::kBFloat16?ACL_BF16:ACL_DT_UNDEFINED;
        TORCH_CHECK(dtype!=ACL_DT_UNDEFINED,"unsupported reference dtype");
        std::vector<int64_t> shape(x.sizes().begin(),x.sizes().end());
        std::vector<int64_t> strides(x.strides().begin(),x.strides().end());
        int64_t span=x.numel()?1:0;
        for (size_t dim=0;dim<shape.size();++dim) span+=(shape[dim]-1)*strides[dim];
        auto value=aclCreateTensor(shape.data(),shape.size(),dtype,strides.data(),0,
                                  ACL_FORMAT_ND,&span,1,x.data_ptr());
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

// 仅计算宿主标量参数，不读取或处理输入张量的数值。
template<typename T> double adjacent16(double value,bool up) {
    T rounded(static_cast<float>(value));
    float number=static_cast<float>(rounded);
    if (number==0.) rounded.x=up?1:0x8001;
    else if ((number>0)==up) ++rounded.x;
    else --rounded.x;
    return static_cast<double>(static_cast<float>(rounded));
}

template<typename T> std::pair<double,double> bounds16(double low,double high) {
    double lower=static_cast<float>(T(static_cast<float>(low)));
    double upper=static_cast<float>(T(static_cast<float>(high)));
    if (lower<low) lower=adjacent16<T>(lower,true);
    if (upper>=high) upper=adjacent16<T>(upper,false);
    TORCH_CHECK(lower<=upper,"interval has no representable output");
    return {lower,upper};
}

std::pair<double,double> bounds(at::ScalarType dtype,double low,double high) {
    if (dtype==at::kHalf) return bounds16<c10::Half>(low,high);
    if (dtype==at::kBFloat16) return bounds16<c10::BFloat16>(low,high);
    TORCH_CHECK(dtype==at::kFloat,"unsupported reference dtype");
    double lower=static_cast<float>(low),upper=static_cast<float>(high);
    if (lower<low) lower=std::nextafterf(static_cast<float>(lower),std::numeric_limits<float>::infinity());
    if (upper>=high) upper=std::nextafterf(static_cast<float>(upper),-std::numeric_limits<float>::infinity());
    TORCH_CHECK(lower<=upper,"interval has no representable output");
    return {lower,upper};
}

using Execute=aclnnStatus(*)(void*,uint64_t,aclOpExecutor*,aclrtStream);
void execute(Execute fn,uint64_t bytes,aclOpExecutor* executor,
             const at::Tensor& x,aclrtStream stream) {
    auto workspace=bytes?at::empty({static_cast<int64_t>(bytes)},x.options().dtype(at::kByte)):at::Tensor();
    auto code=fn(workspace.defined()?workspace.data_ptr():nullptr,bytes,executor,stream);
    TORCH_CHECK(code==0,"CANN execute rc=",code);
}

at::Tensor uniform(const at::Tensor& input,double low,double high,
                   c10::optional<at::Generator> generator) {
    TORCH_CHECK(std::isfinite(low) && std::isfinite(high) && low<=high,"invalid uniform range");
    TORCH_CHECK(input.scalar_type()==at::kFloat || input.scalar_type()==at::kHalf ||
                input.scalar_type()==at::kBFloat16,"unsupported reference dtype");
    auto interval=low!=high?bounds(input.scalar_type(),low,high):std::make_pair(low,high);
    auto output=at::empty(input.sizes(),input.options());
    if (!output.numel()) return output;
    auto* rng=at::get_generator_or_default<at_npu::NPUGeneratorImpl>(
        generator,at_npu::detail::getDefaultNPUGenerator(input.device().index()));
    std::pair<uint64_t,uint64_t> state;
    {
        std::lock_guard<std::mutex> guard(rng->mutex_);
        state=rng->philox_engine_inputs(10);
    }
    auto stream=c10_npu::getCurrentNPUStream(input.device().index()).stream();
    {
        Descriptors descriptors;
        uint64_t bytes=0;
        aclOpExecutor* executor=nullptr;
        auto code=aclnnInplaceUniformGetWorkspaceSize(descriptors.tensor(output),low,high,
                                                    state.first,state.second,&bytes,&executor);
        TORCH_CHECK(code==0,"aclnnInplaceUniform query rc=",code);
        execute(aclnnInplaceUniform,bytes,executor,output,stream);
    }
    if (low!=high) {
        Descriptors descriptors;
        uint64_t bytes=0;
        aclOpExecutor* executor=nullptr;
        auto out=descriptors.tensor(output);
        auto code=aclnnClampGetWorkspaceSize(out,descriptors.scalar(interval.first),
                                            descriptors.scalar(interval.second),out,&bytes,&executor);
        TORCH_CHECK(code==0,"aclnnClamp query rc=",code);
        execute(aclnnClamp,bytes,executor,output,stream);
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,module) {
    module.def("uniform",&uniform);
    module.def("bounds",[](const std::string& dtype,double low,double high) {
        auto type=dtype=="float16"?at::kHalf:dtype=="bfloat16"?at::kBFloat16:at::kFloat;
        return bounds(type,low,high);
    });
}
