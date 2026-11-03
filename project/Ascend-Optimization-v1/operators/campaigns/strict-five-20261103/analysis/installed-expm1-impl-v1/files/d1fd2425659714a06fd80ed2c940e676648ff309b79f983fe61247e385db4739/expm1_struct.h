/**
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
 * \file expm1_struct.h
 * \brief
 */

#include "ascendc/host_api/tiling/template_argument.h"

#ifndef CANN_CUSTOM_OPS_EXPM1_STRUCT_H_
#define CANN_CUSTOM_OPS_EXPM1_STRUCT_H_

#define TPL_FP16 1
#define TPL_BF16 2
#define TPL_FP32 3
 
#define TPL_SCH_MODE_0 0
#define TPL_SCH_MODE_1 1
 
ASCENDC_TPL_ARGS_DECL(Expm1,
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, TPL_SCH_MODE_0, TPL_SCH_MODE_1),
    ASCENDC_TPL_DTYPE_DECL(dType, TPL_FP16, TPL_BF16, TPL_FP32)
);
 
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, TPL_SCH_MODE_0, TPL_SCH_MODE_1),
        ASCENDC_TPL_DTYPE_SEL(dType, TPL_FP16, TPL_BF16, TPL_FP32)
    )
);
#endif  // CANN_CUSTOM_OPS_EXPM1_STRUCT_H_