// HG-KDA实验候选：来源可见的Ascend C log10，非厂商算子实现。
// 连续、32字节对齐且长度为16倍数；其他布局由Python候选的Triton路径处理。
#include "kernel_operator.h"

template<typename T>
__aicore__ inline void Log10Impl(GM_ADDR x, uint32_t n) {
    constexpr uint32_t TILE = 16384;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rawBuffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatBuffer;
    pipe.InitBuffer(rawBuffer, TILE * sizeof(T));
    pipe.InitBuffer(floatBuffer, TILE * sizeof(float));
    auto raw = rawBuffer.Get<T>();
    auto values = floatBuffer.Get<float>();
    AscendC::GlobalTensor<T> global;
    global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), n);
    for (uint32_t start = AscendC::GetBlockIdx() * TILE; start < n;
         start += AscendC::GetBlockNum() * TILE) {
        uint32_t count = n - start < TILE ? n - start : TILE;
        AscendC::DataCopy(raw, global[start], count);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Cast(values, raw, AscendC::RoundMode::CAST_NONE, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Ln(values, values, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(values, values, 0.4342944819032518f, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(raw, values, AscendC::RoundMode::CAST_RINT, count);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(global[start], raw, count);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

// FLOAT->FLOAT不使用Cast，避免同类型Cast在部分编译器中的限制。
template<>
__aicore__ inline void Log10Impl<float>(GM_ADDR x, uint32_t n) {
    constexpr uint32_t TILE = 16384;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, TILE * sizeof(float));
    auto values = buffer.Get<float>();
    AscendC::GlobalTensor<float> global;
    global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), n);
    for (uint32_t start = AscendC::GetBlockIdx() * TILE; start < n;
         start += AscendC::GetBlockNum() * TILE) {
        uint32_t count = n - start < TILE ? n - start : TILE;
        AscendC::DataCopy(values, global[start], count);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Ln(values, values, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(values, values, 0.4342944819032518f, count);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(global[start], values, count);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

extern "C" __global__ __aicore__ void hgkda_log10_fp16(GM_ADDR x, uint32_t n) { Log10Impl<half>(x,n); }
extern "C" __global__ __aicore__ void hgkda_log10_bf16(GM_ADDR x, uint32_t n) { Log10Impl<bfloat16_t>(x,n); }
extern "C" __global__ __aicore__ void hgkda_log10_fp32(GM_ADDR x, uint32_t n) { Log10Impl<float>(x,n); }

extern "C" void hgkda_log10_launch(uint32_t blocks, void* stream, uint8_t* x, uint32_t n, int dtype) {
    if (dtype==0) hgkda_log10_fp32<<<blocks,nullptr,stream>>>(x,n);
    else if (dtype==1) hgkda_log10_fp16<<<blocks,nullptr,stream>>>(x,n);
    else if (dtype==27) hgkda_log10_bf16<<<blocks,nullptr,stream>>>(x,n);
}
