// 自写Ascend C：弧度转角度，不调用厂商目标算子。
#include "kernel_operator.h"

template<typename T>
__aicore__ inline void Rad2deg(GM_ADDR x,uint32_t n) {
    using namespace AscendC;
    constexpr uint32_t TILE=8192;
    TPipe pipe;
    TBuf<TPosition::VECCALC> rawBuffer,floatBuffer;
    pipe.InitBuffer(rawBuffer,2*TILE*sizeof(T));
    if constexpr (!__is_same(T,float)) pipe.InitBuffer(floatBuffer,TILE*sizeof(float));
    auto rawBase=rawBuffer.Get<T>();
    GlobalTensor<T> memory;memory.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x),n);
    bool used[2]={false,false};uint32_t iteration=0;
    for(uint32_t start=GetBlockIdx()*TILE;start<n;start+=GetBlockNum()*TILE,++iteration) {
        uint32_t count=n-start<TILE?n-start:TILE;uint32_t slot=iteration&1;
        event_t event=static_cast<event_t>(slot);
        if(used[slot]) WaitFlag<HardEvent::MTE3_MTE2>(event);
        auto raw=rawBase[slot*TILE];DataCopy(raw,memory[start],count);
        SetFlag<HardEvent::MTE2_V>(event);WaitFlag<HardEvent::MTE2_V>(event);
        LocalTensor<float> values;
        if constexpr (__is_same(T,float)) values=raw.template ReinterpretCast<float>();
        else {values=floatBuffer.Get<float>();Cast(values,raw,RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();}
        Muls(values,values,57.2957795130823208768f,count);PipeBarrier<PIPE_V>();
        if constexpr (!__is_same(T,float)) {Cast(raw,values,RoundMode::CAST_RINT,count);PipeBarrier<PIPE_V>();}
        SetFlag<HardEvent::V_MTE3>(event);WaitFlag<HardEvent::V_MTE3>(event);
        DataCopy(memory[start],raw,count);SetFlag<HardEvent::MTE3_MTE2>(event);used[slot]=true;
    }
    for(uint32_t slot=0;slot<2;slot++)if(used[slot])WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(slot));
}
extern "C" __global__ __aicore__ void hgkda_rad2deg_f32(GM_ADDR x,uint32_t n){Rad2deg<float>(x,n);}
extern "C" __global__ __aicore__ void hgkda_rad2deg_f16(GM_ADDR x,uint32_t n){Rad2deg<half>(x,n);}
extern "C" __global__ __aicore__ void hgkda_rad2deg_bf16(GM_ADDR x,uint32_t n){Rad2deg<bfloat16_t>(x,n);}
extern "C" void hgkda_rad2deg_launch(uint32_t blocks,void* stream,uint8_t* x,uint32_t n,int dtype) {
    if(dtype==0)hgkda_rad2deg_f32<<<blocks,nullptr,stream>>>(x,n);
    else if(dtype==1)hgkda_rad2deg_f16<<<blocks,nullptr,stream>>>(x,n);
    else if(dtype==27)hgkda_rad2deg_bf16<<<blocks,nullptr,stream>>>(x,n);
}
