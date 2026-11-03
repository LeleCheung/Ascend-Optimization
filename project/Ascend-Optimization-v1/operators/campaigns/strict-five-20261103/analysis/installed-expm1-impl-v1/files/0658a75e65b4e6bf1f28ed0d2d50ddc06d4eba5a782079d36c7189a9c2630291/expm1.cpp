/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* !
 * \file expm1.cpp
 * \brief z = expm1(x)
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "v35/expm1_struct.h"
#include "v35/expm1_dag.h"
#include "op_kernel/atvoss/elemwise/elementwise_sch.h"
#include "op_kernel/atvoss/atp/dfx.h"

using namespace AscendC;

template <uint64_t schMode, uint64_t dType>
__global__ __aicore__ void expm1(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if constexpr (dType == TPL_FP32) {
        ElementwiseSch<schMode, Expm1Op::Expm1DAG<float>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (dType == TPL_BF16) {
        ElementwiseSch<schMode, Expm1Op::Expm1DAG<bfloat16_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (dType == TPL_FP16) {
        ElementwiseSch<schMode, Expm1Op::Expm1DAG<half>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(x, y);
        sch.Process();
    }
}
