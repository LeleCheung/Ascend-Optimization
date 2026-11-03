/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http:
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* !
 * \file expm1_dag.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_EXPM1_DAG_H
#define CANN_CUSTOM_OPS_EXPM1_DAG_H
 
#include "op_kernel/atvoss/atp/dag.h"
#include "op_kernel/atvoss/atp/vec.h"
#include "op_kernel/atvoss/atp/placeholder.h"
#include <limits>

namespace Expm1Op
{
using namespace AscendC;
const int CAST_MODE_NONE = 0;
const int CAST_MODE_RINT = 1;
constexpr uint32_t THREAD_NUM = 1024;
constexpr float INV_LN2_APPROX = 1.4426950216293335f;
constexpr float LN2_HALF_APPROX = 0.4099999964237213f;
constexpr float LN2_APPROX = 0.693145751953125f;
constexpr float ONE_MINUS_LN2_APPROX = 0.000001428606765330187f;
constexpr float FLOAT_128 = 128.000000f;
constexpr float FLOAT_NEG_ONE = -1.00000000f;
constexpr float C5 = 0.008382412604987621f;
constexpr float C4 = 0.0013879507314413786f;
constexpr float C3 = 0.04166783019900322f;
constexpr float C2 = 0.1666639745235443f;
constexpr float C1 = 0.4999999403953552f;
constexpr float FLOAT_INF = std::numeric_limits<float>::infinity();
constexpr float FLOAT_NEG_25 = -25.0000000f;
constexpr float FLOAT_2 = 2.0000000f;

#ifdef __CCE_AICORE__
template<typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void Expm1SimtCompute(__ubuf__ T* x, __ubuf__ T* y, const int64_t totalNum)
{
    for (int64_t i = Simt::GetThreadIdx(); i < totalNum; i += Simt::GetThreadNum()) {
        float f1 = x[i]; 
        float f0 = Simt::Expm1(f1);
        float f2 = f1 * INV_LN2_APPROX; 
        float f3 = Simt::Round(f2); 
        float f4 = Simt::Abs(f1); 
        bool p1 = f4 < LN2_HALF_APPROX; 
        float f5 = p1 ? 0.0f : f3; 
        float f6 = -f5; 
        float f7 = LN2_APPROX; 
        float f8 = Simt::Fma(f6, f7, f1); 
        float f9 = ONE_MINUS_LN2_APPROX; 
        float f10 = Simt::Fma(f6, f9, f8); 
        bool p2 = f5 == FLOAT_128; 
        float f11 = f5 + FLOAT_NEG_ONE; 
        float f12 = p2 ? f11 : f5; 
        float f13 = C5;
        float f14 = C4;
        float f15 = Simt::Fma(f14, f10, f13); 
        float f16 = C3;
        float f17 = Simt::Fma(f15, f10, f16); 
        float f18 = C2;
        float f19 = Simt::Fma(f17, f10, f18); 
        float f20 = C1;
        float f21 = Simt::Fma(f19, f10, f20); 
        float f22 = f10 * f21; 
        float f23 = Simt::Fma(f22, f10, f10); 
        float f24 = Simt::Exp2(f12); 
        float f25 = f24 + FLOAT_NEG_ONE; 
        float f26 = Simt::Fma(f23, f24, f25); 
        float f27 = f26 + f26; 
        float f28 = p2 ? f27 : f26; 
        bool p3 = f12 > FLOAT_128; 
        float f29 = p3 ? FLOAT_INF : f28; 
        bool p4 = f12 < FLOAT_NEG_25; 
        float f30 = p4 ? FLOAT_NEG_ONE : f29; 
        bool p5 = f1 == 0.0f; 
        float f31 = f1 + f1; 
        float f32 = p5 ? f31 : f30; 
        y[i] = Simt::Abs(x[i]) > FLOAT_2 ? f0 : f32;
    }
}
#endif

template<class T>
struct Expm1Custom : public AscendC::Vec::ElemwiseUnaryOP<T, T> {
    __aicore__ inline Expm1Custom(LocalTensor<T> &dst, LocalTensor<T> &src, uint32_t count) {
#ifdef __CCE_AICORE__
        __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();
        Simt::VF_CALL<Expm1SimtCompute<T>>(Simt::Dim3(THREAD_NUM),srcAddr,dstAddr,count);
#endif
    }
};

template <typename U, typename T = float>
struct Expm1DAG {
    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCopyIn0Cast = Bind<Vec::Cast<T, U, CAST_MODE_NONE>, OpCopyIn0>;
    using OpResult1 = Bind<Expm1Custom<T>, OpCopyIn0Cast>;
    using OpResultCast = Bind<Vec::Cast<U, T, CAST_MODE_RINT>, OpResult1>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpResultCast>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
}
#endif  // CANN_CUSTOM_OPS_EXPM1_DAG_H