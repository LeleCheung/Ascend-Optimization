// HG-KDA来源可见的Ascend C逐元素实现。不是调用厂商目标算子。
#include "kernel_operator.h"

// OP: 0 negative, 1 log2, 2 trunc, 3 round-to-even, 4 hardswish.
template<int OP>
__aicore__ inline void Compute(AscendC::LocalTensor<float> value,
                              AscendC::LocalTensor<float> tmp, AscendC::LocalTensor<float> aux, AscendC::LocalTensor<uint8_t> mask, uint32_t count) {
    using namespace AscendC;
    if constexpr (OP==0) {
        Muls(value,value,-1.0f,count);
    } else if constexpr (OP==1) {
        Ln(value,value,count);PipeBarrier<PIPE_V>();
        Muls(value,value,1.4426950408889634f,count);
    } else if constexpr (OP==2) {
        Cast(value,value,RoundMode::CAST_TRUNC,count);
    } else if constexpr (OP==3) {
        Cast(value,value,RoundMode::CAST_RINT,count);
    } else if constexpr (OP==5) {
        Exp(tmp,value,count);PipeBarrier<PIPE_V>();
        Adds(tmp,tmp,-1.0f,count);PipeBarrier<PIPE_V>();
        Abs(aux,value,count);PipeBarrier<PIPE_V>();
        CompareScalar(mask,aux,0.01f,CMPMODE::LT,count);PipeBarrier<PIPE_V>();
        Muls(aux,value,0.5f,count);PipeBarrier<PIPE_V>();
        Adds(aux,aux,1.0f,count);PipeBarrier<PIPE_V>();
        Mul(aux,value,aux,count);PipeBarrier<PIPE_V>();
        Select(value,mask,aux,tmp,SELMODE::VSEL_TENSOR_TENSOR_MODE,count);
    } else if constexpr (OP==4) {
        Adds(tmp,value,3.0f,count);PipeBarrier<PIPE_V>();
        Maxs(tmp,tmp,0.0f,count);PipeBarrier<PIPE_V>();
        Mins(tmp,tmp,6.0f,count);PipeBarrier<PIPE_V>();
        Mul(value,value,tmp,count);PipeBarrier<PIPE_V>();
        Muls(value,value,1.0f/6.0f,count);
    }
}



template<typename T,int OP>
__aicore__ inline void Pointwise(GM_ADDR x, GM_ADDR y, uint32_t n) {
    using namespace AscendC;
    constexpr uint32_t TILE=OP==5?8192:16384;
    TPipe pipe;
    TBuf<TPosition::VECCALC> rawBuffer,floatBuffer,tmpBuffer,auxBuffer,maskBuffer;
    pipe.InitBuffer(rawBuffer,2*TILE*sizeof(T));
    if constexpr (!__is_same(T,float)) pipe.InitBuffer(floatBuffer,TILE*4);
    if constexpr (OP==4 || OP==5) pipe.InitBuffer(tmpBuffer,TILE*4);
    if constexpr (OP==5) {pipe.InitBuffer(auxBuffer,TILE*4);pipe.InitBuffer(maskBuffer,TILE/8);}
    auto rawBase=rawBuffer.Get<T>();
    LocalTensor<float> tmp,aux;LocalTensor<uint8_t> mask;
    if constexpr (OP==4 || OP==5) tmp=tmpBuffer.Get<float>();
    if constexpr (OP==5) {aux=auxBuffer.Get<float>();mask=maskBuffer.Get<uint8_t>();}
    GlobalTensor<T> src,dst;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x),n);dst.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y),n);
    uint32_t iteration=0;
    bool used[2]={false,false};
    for(uint32_t start=GetBlockIdx()*TILE;start<n;start+=GetBlockNum()*TILE,++iteration) {
        uint32_t count=n-start<TILE?n-start:TILE;
        uint32_t slot=iteration&1;
        event_t event=static_cast<event_t>(slot);
        if(used[slot]) WaitFlag<HardEvent::MTE3_MTE2>(event);
        auto raw=rawBase[slot*TILE];
        DataCopy(raw,src[start],count);
        SetFlag<HardEvent::MTE2_V>(event);WaitFlag<HardEvent::MTE2_V>(event);
        LocalTensor<float> values;
        if constexpr (__is_same(T,float)) values=raw.template ReinterpretCast<float>();
        else {values=floatBuffer.Get<float>();Cast(values,raw,RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();}
        Compute<OP>(values,tmp,aux,mask,count);PipeBarrier<PIPE_V>();
        if constexpr (!__is_same(T,float)) {Cast(raw,values,RoundMode::CAST_RINT,count);PipeBarrier<PIPE_V>();}
        SetFlag<HardEvent::V_MTE3>(event);WaitFlag<HardEvent::V_MTE3>(event);
        DataCopy(dst[start],raw,count);
        SetFlag<HardEvent::MTE3_MTE2>(event);used[slot]=true;
    }
    for(uint32_t slot=0;slot<2;slot++) if(used[slot]) WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(slot));
}

#define KERNELS(OP,NAME) \
extern "C" __global__ __aicore__ void NAME##_f32(GM_ADDR x,GM_ADDR y,uint32_t n){Pointwise<float,OP>(x,y,n);} \
extern "C" __global__ __aicore__ void NAME##_f16(GM_ADDR x,GM_ADDR y,uint32_t n){Pointwise<half,OP>(x,y,n);} \
extern "C" __global__ __aicore__ void NAME##_bf16(GM_ADDR x,GM_ADDR y,uint32_t n){Pointwise<bfloat16_t,OP>(x,y,n);} \
extern "C" void NAME##_launch(uint32_t blocks,void* stream,uint8_t* x,uint8_t* y,uint32_t n,int dtype){ \
 if(dtype==0) NAME##_f32<<<blocks,nullptr,stream>>>(x,y,n); \
 else if(dtype==1) NAME##_f16<<<blocks,nullptr,stream>>>(x,y,n); \
 else if(dtype==27) NAME##_bf16<<<blocks,nullptr,stream>>>(x,y,n); }
KERNELS(0,hgkda_negative)
KERNELS(1,hgkda_log2)
KERNELS(2,hgkda_fix)
KERNELS(3,hgkda_round)
KERNELS(4,hgkda_hardswish)

KERNELS(5,hgkda_expm1)


template<typename T>
__aicore__ inline void Fmod(GM_ADDR a,GM_ADDR b,uint32_t n) {
    using namespace AscendC;
    constexpr uint32_t TILE=4096;
    TPipe pipe;
    TQue<QuePosition::VECIN,2> inputA,inputB;
    TQue<QuePosition::VECOUT,2> output;
    TBuf<TPosition::VECCALC> fa,fb,tmp;
    pipe.InitBuffer(inputA,2,TILE*sizeof(T));pipe.InitBuffer(inputB,2,TILE*sizeof(T));
    pipe.InitBuffer(output,2,TILE*sizeof(T));pipe.InitBuffer(tmp,TILE*4);
    if constexpr (!__is_same(T,float)) {pipe.InitBuffer(fa,TILE*4);pipe.InitBuffer(fb,TILE*4);}
    auto q=tmp.Get<float>();
    GlobalTensor<T> ga,gb;ga.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(a),n);gb.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(b),n);
    for(uint32_t start=GetBlockIdx()*TILE;start<n;start+=GetBlockNum()*TILE) {
        uint32_t count=n-start<TILE?n-start:TILE;
        auto inA=inputA.AllocTensor<T>();auto inB=inputB.AllocTensor<T>();
        DataCopy(inA,ga[start],count);DataCopy(inB,gb[start],count);inputA.EnQue(inA);inputB.EnQue(inB);
        auto rawA=inputA.DeQue<T>();auto rawB=inputB.DeQue<T>();auto result=output.AllocTensor<T>();
        LocalTensor<float> x,y;
        if constexpr (__is_same(T,float)) {x=rawA.template ReinterpretCast<float>();y=rawB.template ReinterpretCast<float>();}
        else {x=fa.Get<float>();y=fb.Get<float>();Cast(x,rawA,RoundMode::CAST_NONE,count);Cast(y,rawB,RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();}
        Div(q,x,y,count);PipeBarrier<PIPE_V>();Cast(q,q,RoundMode::CAST_TRUNC,count);PipeBarrier<PIPE_V>();
        Mul(q,q,y,count);PipeBarrier<PIPE_V>();Sub(x,x,q,count);PipeBarrier<PIPE_V>();
        if constexpr (__is_same(T,float)) DataCopy(result,x,count);
        else Cast(result,x,RoundMode::CAST_RINT,count);
        output.EnQue(result);inputA.FreeTensor(rawA);inputB.FreeTensor(rawB);
        auto outgoing=output.DeQue<T>();DataCopy(ga[start],outgoing,count);output.FreeTensor(outgoing);
    }
}

extern "C" __global__ __aicore__ void hgkda_fmod_f32(GM_ADDR a,GM_ADDR b,uint32_t n){Fmod<float>(a,b,n);}
extern "C" __global__ __aicore__ void hgkda_fmod_f16(GM_ADDR a,GM_ADDR b,uint32_t n){Fmod<half>(a,b,n);}
extern "C" __global__ __aicore__ void hgkda_fmod_bf16(GM_ADDR a,GM_ADDR b,uint32_t n){Fmod<bfloat16_t>(a,b,n);}
extern "C" void hgkda_fmod_launch(uint32_t blocks,void* stream,uint8_t* a,uint8_t* b,uint32_t n,int dtype) {
    if(dtype==0) hgkda_fmod_f32<<<blocks,nullptr,stream>>>(a,b,n);
    else if(dtype==1) hgkda_fmod_f16<<<blocks,nullptr,stream>>>(a,b,n);
    else if(dtype==27) hgkda_fmod_bf16<<<blocks,nullptr,stream>>>(a,b,n);
}
