// HG-KDA源码可见Ascend C补充参考：仅连续FP32、元素数为8的倍数。
// 使用厂商Ascend C基本API；不是厂商最优negative实现或新的正式速度基线。
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void hgkda_negative_fp32(GM_ADDR x, uint32_t n) {
    constexpr uint32_t TILE = 2048;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, TILE * sizeof(float));
    auto local = buffer.Get<float>();
    AscendC::GlobalTensor<float> global;
    global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), n);
    for (uint32_t start = AscendC::GetBlockIdx() * TILE; start < n;
         start += AscendC::GetBlockNum() * TILE) {
        uint32_t count = n - start < TILE ? n - start : TILE;
        AscendC::DataCopy(local, global[start], count);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Muls(local, local, -1.0f, count);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(global[start], local, count);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

extern "C" void hgkda_negative_launch(uint32_t blocks, void* stream, uint8_t* x, uint32_t n) {
    hgkda_negative_fp32<<<blocks, nullptr, stream>>>(x, n);
}
