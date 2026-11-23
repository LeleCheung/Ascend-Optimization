/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file batch_copy_cube_in_using_ub.h
 * \brief
 */

#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#pragma message("impl/adv_api/detail/matmul/stage/copy_cube_in/batch/batch_copy_cube_in_using_ub.h is an internal header file and must not be used directly. Functions or variables defined in this file may be removed in the future. Please use \"#include \"adv_api/matmul/matmul.h\"\" and use public functions or variables defined in interface headers files.")
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_DETAIL_MATMUL_STAGE_COPY_CUBE_IN_BATCH_BATCH_COPY_CUBE_IN_USING_UB_H__
#endif

#ifndef IMPL_MATMUL_STAGE_COPY_CUBE_IN_BATCH_BATCH_COPY_CUBE_IN_USING_UB_H
#define IMPL_MATMUL_STAGE_COPY_CUBE_IN_BATCH_BATCH_COPY_CUBE_IN_USING_UB_H

#include "batch_copy_cube_in_intf.h"
#include "batch_copy_cube_in_params.h"
#include "batch_copy_cube_in_base.h"
#include "../base/copy_cube_in_params.h"
#include "../copy_tile_to_cube/copy_tile_to_cube.h"

namespace AscendC {
namespace Impl {
namespace Detail {
// Specialized Template Class of Batch Matmul CopyIn
// Batch Matmul ND Format Data CopyIn From GM, only support NORMAL
template <typename IMPL, const auto& MM_CFG, class INPUT_TYPE>
class BatchCopyCubeIn<IMPL, MM_CFG, INPUT_TYPE, enable_if_t<
    (MatmulFeatureTrait<MM_CFG>::IsNeedUB()) &&
    GetCopyCubeInType<INPUT_TYPE, MM_CFG>() == CopyCubeInType::BMM &&
    (INPUT_TYPE::format == CubeFormat::ND) &&
    (INPUT_TYPE::layout == LayoutMode::NORMAL) &&
    PhyPosIsGM(INPUT_TYPE::pos)>>
    : public BatchCopyCubeInBase<IMPL, MM_CFG, INPUT_TYPE>
{
private:
    MATMUL_USE_MODULE_ON(CubeInBuffer, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(BatchCopyCubeInParams, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(CopyCubeInParams, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(DataCopyWrapper, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(MatmulTensorInfo, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulShapeInfo);

    using TransT = typename INPUT_TYPE::TRANS_T;
    using SrcT = typename INPUT_TYPE::T;

public:
    using BASE_MODULE = AscendC::Impl::Detail::BatchCopyCubeInBase<IMPL, MM_CFG, INPUT_TYPE>;

    inline __aicore__ BatchCopyCubeIn() = default;
    inline __aicore__ ~BatchCopyCubeIn() = default;

    __aicore__ inline void Init()
    {
        MATMUL_MODULE(CubeInBuffer)->Init(
            MATMUL_MODULE(BatchCopyCubeInParams)->GetBatchNum() *
            CeilAlign(MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleWidth<INPUT_TYPE::isTrans>(), c0Size_) *
            CeilAlign(MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleHeight<INPUT_TYPE::isTrans>(), BLOCK_CUBE), 1);
    }

    __aicore__ inline void BatchLoad(LocalTensor<TransT>& dstTensor, const uint32_t matrixStride,
                                     const int32_t outerIdx, const int32_t splitIdx, const int32_t splitSize)
    {
        if (MATMUL_MODULE(BatchCopyCubeInParams)->IsTranspose()) {
            return CopyBatchToCube<true, INPUT_TYPE::TAG == InputTypeTag::A>(
                dstTensor, matrixStride, outerIdx, splitIdx, splitSize);
        } else {
            return CopyBatchToCube<false, INPUT_TYPE::TAG == InputTypeTag::B>(
                dstTensor, matrixStride, outerIdx, splitIdx, splitSize);
        }
    }

private:
    template <bool isTrans = false, bool isKRow = false>
    __aicore__ inline void CopyBatchToCube(LocalTensor<TransT>& dstTensor, const uint32_t matrixStride,
                                           const int32_t outerIdx, const int32_t splitIdx, const int32_t splitSize)
    {
        // 1. calculate src stride and dst stride by db split loop index
        int32_t batchNum = MATMUL_MODULE(BatchCopyCubeInParams)->GetBatchNum();
        int32_t batchNumIdx = batchNum / splitSize;
        auto alignWidth = CeilAlign(MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleWidth<isTrans>(), c0Size_);
        auto alignHeight = CeilAlign(MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleHeight<isTrans>(), BLOCK_CUBE);
        auto srcStride = GetSingleSize();
        auto dstStride = alignWidth * alignHeight;
        uint64_t srcOffset = batchNumIdx * splitIdx * srcStride;
        uint64_t dstOffset = batchNumIdx * splitIdx * dstStride;
        // if odd ground, the first block is unequal with the second block
        auto batchBlock = splitIdx == 0 ? batchNumIdx : batchNum - batchNumIdx;

        // 2. copy batch matrix in
        int64_t batchOffset = outerIdx * MATMUL_MODULE(BatchCopyCubeInParams)->GetBatchNum() * srcStride;
        GlobalTensor<SrcT> srcGlobal;
        srcGlobal.SetGlobalBuffer(MATMUL_MODULE(MatmulTensorInfo)->GetGlobalTensor().address_);
        srcGlobal.SetAddr(batchOffset + srcOffset);
        if constexpr (ToMatmulConfig(MM_CFG).enVecND2NZ) {
            CopyND2NZThroughVec<isTrans>(
                dstTensor[dstOffset], srcGlobal, batchBlock, outerIdx, splitIdx, alignHeight, alignWidth);
        } else {
            if constexpr (isKRow) {
                MATMUL_MODULE(DataCopyWrapper)->CopyND2NZOnTheFly(
                    dstTensor[dstOffset], srcGlobal, 0, 0,
                    MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleHeight<isTrans>(),
                    batchBlock * alignWidth, batchBlock * alignWidth);
            } else {
                MATMUL_MODULE(DataCopyWrapper)->CopyND2NZOnTheFly(
                    dstTensor[dstOffset], srcGlobal, 0, 0, batchBlock * alignHeight,
                    MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleWidth<isTrans>(),
                    MATMUL_MODULE(CopyCubeInParams)->template GetOrgWidth<isTrans>());
            }
        }
    }

    template <bool isTrans = false, InputTypeTag tag = INPUT_TYPE::TAG>
    __aicore__ inline enable_if_t<tag == InputTypeTag::A, void>
    CopyND2NZThroughVec(const LocalTensor<TransT>& dstTensor, const GlobalTensor<SrcT>& srcTensor, int32_t batchNum,
                        int32_t batchOuterIdx, int32_t splitOuterIdx, int32_t alignHeight, int32_t alignWidth)
    {
        auto srcStride = GetSingleSize();
        int64_t srcOffset = 0;
        int64_t dstOffset = 0;

        bool ubEnough = MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleHeight<isTrans>() * c0Size_ <=
                        MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetTransLength() ? true : false;
        if (ubEnough) {
            event_t eventIDMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            auto gCol = isTrans ? MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleWidth<isTrans>() :
                        MATMUL_MODULE(CopyCubeInParams)->template GetOrgWidth<isTrans>();
            int colNum = Ceil(MATMUL_MODULE(CopyCubeInParams)->template GetTotalCol<isTrans>() *
                                  MATMUL_MODULE(CopyCubeInParams)->template GetBaseWidth<isTrans>(), c0Size_);
            for (auto iterBatch = 0; iterBatch < batchNum; ++iterBatch) {
                for (auto i = 0; i < colNum; ++i) {
                    MATMUL_MODULE(DataCopyWrapper)->CopyND2NZWithVecOp(
                        dstTensor[dstOffset], srcTensor[srcOffset], 0, i * c0Size_,
                        MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleHeight<isTrans>(), c0Size_, gCol);
                    dstOffset += alignHeight * c0Size_;
                    SetFlag<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
                    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
                }
                srcOffset += srcStride;
            }
        } else {
            if constexpr (isTrans) {
                MATMUL_MODULE(DataCopyWrapper)->CopyND2NZOnTheFly(
                    dstTensor, srcTensor, 0, 0,
                    MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleHeight<isTrans>(),
                    batchNum * alignWidth, batchNum * alignWidth);
            } else {
                int tail = MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleWidth<isTrans>() % c0Size_;
                if (tail == 0) {
                    for (int iterBatch = 0; iterBatch < batchNum; ++iterBatch) {
                        MATMUL_MODULE(DataCopyWrapper)->CopyND2NZOnTheFly(
                            dstTensor[dstOffset], srcTensor[srcOffset], 0, 0,
                            MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleHeight<isTrans>(),
                            MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleWidth<isTrans>(),
                            MATMUL_MODULE(CopyCubeInParams)->template GetOrgWidth<isTrans>());
                        dstOffset += MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleHeight<isTrans>() * alignWidth;
                        srcOffset += srcStride;
                    }
                } else {
                    for (int iterBatch = 0; iterBatch < batchNum; ++iterBatch) {
                        int64_t innerSrcOffset = srcOffset;
                        for (auto i = 0; i < MATMUL_MODULE(CopyCubeInParams)->template GetTotalRow<isTrans>(); ++i) {
                            MATMUL_MODULE(DataCopyWrapper)->CopyND2NZOnTheFly(
                                dstTensor[dstOffset], srcTensor[innerSrcOffset], 0, 0,
                                MATMUL_MODULE(MatmulTensorInfo)->template GetBaseUseHeight<isTrans>(),
                                MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleWidth<isTrans>(),
                                MATMUL_MODULE(CopyCubeInParams)->template GetOrgWidth<isTrans>());
                            dstOffset += MATMUL_MODULE(MatmulTensorInfo)->template GetBaseUseHeight<isTrans>() * alignWidth;
                            innerSrcOffset += MATMUL_MODULE(MatmulTensorInfo)->template GetBaseUseHeight<isTrans>() *
                                              MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleWidth<isTrans>();
                        }
                        srcOffset += srcStride;
                    }
                }
            }
        }
    }

    template <bool isTrans = false, InputTypeTag tag = INPUT_TYPE::TAG>
    __aicore__ inline enable_if_t<tag == InputTypeTag::B, void>
    CopyND2NZThroughVec(const LocalTensor<TransT>& dstTensor, const GlobalTensor<SrcT>& srcTensor, int32_t batchNum,
                        int32_t batchOuterIdx, int32_t splitOuterIdx, int32_t alignHeight, int32_t alignWidth)
    {
        auto srcStride = GetSingleSize();
        int64_t srcOffset = 0;
        int64_t dstOffset = 0;

        bool ubEnough = MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleHeight<isTrans>() * c0Size_ <=
                        MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetTransLength() ? true : false;
        if (ubEnough) {

            auto colNum = Ceil(MATMUL_MODULE(CopyCubeInParams)->template GetTotalCol<isTrans>() *
                               MATMUL_MODULE(CopyCubeInParams)->template GetBaseWidth<isTrans>(), c0Size_);
            event_t eventIDMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            for (auto iterBatch = 0; iterBatch < batchNum; ++iterBatch) {
                for (auto i = 0; i < colNum; ++i) {
                    MATMUL_MODULE(DataCopyWrapper)->CopyND2NZWithVecOp(
                        dstTensor[dstOffset], srcTensor[srcOffset], 0, i * c0Size_,
                        MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleHeight<isTrans>(), c0Size_,
                        MATMUL_MODULE(CopyCubeInParams)->template GetOrgWidth<isTrans>());
                    dstOffset += alignHeight * c0Size_;
                    SetFlag<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
                    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
                }
                if constexpr (isTrans) {
                    auto tail = MATMUL_MODULE(CopyCubeInParams)->template GetOrgWidth<isTrans>() % c0Size_;
                    if (tail != 0) {
                        MATMUL_MODULE(DataCopyWrapper)->CopyND2NZWithVecOp(
                            dstTensor[dstOffset], srcTensor[srcOffset], 0, colNum * c0Size_,
                            MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleHeight<isTrans>(), tail,
                            MATMUL_MODULE(CopyCubeInParams)->template GetOrgWidth<isTrans>());
                        dstOffset += alignHeight * c0Size_;
                        SetFlag<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
                        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
                    }
                }
                srcOffset += srcStride;
            }
        } else {
            if constexpr (isTrans) {
                MATMUL_MODULE(DataCopyWrapper)->CopyND2NZOnTheFly(
                    dstTensor, srcTensor, 0, 0, batchNum * alignHeight,
                    MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleWidth<isTrans>(),
                    MATMUL_MODULE(CopyCubeInParams)->template GetOrgWidth<isTrans>());
            } else {
                MATMUL_MODULE(DataCopyWrapper)->CopyND2NZOnTheFly(
                    dstTensor, srcTensor, 0, 0,
                    MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleHeight<isTrans>(),
                    batchNum * alignWidth, batchNum * alignWidth);
            }
        }
    }

    __aicore__ inline int32_t GetSingleSize() const
    {
        // not support constantization
        return MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleWidth() *
               MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleHeight();
    }

private:
    constexpr static int32_t c0Size_ = AuxGetC0Size<TransT>();
};

// Specialized Template Class of Batch Matmul CopyIn
// Batch Matmul NZ Format Data CopyIn From GM/UB, only support NORMAL
template <typename IMPL, const auto& MM_CFG, class INPUT_TYPE>
class BatchCopyCubeIn<IMPL, MM_CFG, INPUT_TYPE, enable_if_t<(
    MatmulFeatureTrait<MM_CFG>::IsNeedUB()) &&
    GetCopyCubeInType<INPUT_TYPE, MM_CFG>() == CopyCubeInType::BMM &&
    (INPUT_TYPE::format == CubeFormat::NZ) &&
    (INPUT_TYPE::layout == LayoutMode::NORMAL) &&
    (PhyPosIsUB(INPUT_TYPE::pos) || PhyPosIsGM(INPUT_TYPE::pos))>>
    : public BatchCopyCubeInBase<IMPL, MM_CFG, INPUT_TYPE>
{
    MATMUL_USE_MODULE_ON(CubeInBuffer, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(BatchCopyCubeInParams, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(CopyCubeInParams, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(DataCopyWrapper, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(MatmulTensorInfo, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE(MatmulShapeInfo);

    using TransT = typename INPUT_TYPE::TRANS_T;
    using SrcT = typename INPUT_TYPE::T;

public:
    using BASE_MODULE = AscendC::Impl::Detail::BatchCopyCubeInBase<IMPL, MM_CFG, INPUT_TYPE>;

    inline __aicore__ BatchCopyCubeIn() = default;
    inline __aicore__ ~BatchCopyCubeIn() = default;

    __aicore__ inline void Init()
    {
        if constexpr (INPUT_TYPE::isTrans) {
            MATMUL_MODULE(CubeInBuffer)->Init(
                MATMUL_MODULE(BatchCopyCubeInParams)->GetBatchNum() *
                MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleSizeAlign<true, INPUT_TYPE::TAG == InputTypeTag::A>(), 1);
        } else {
            MATMUL_MODULE(CubeInBuffer)->Init(
                MATMUL_MODULE(BatchCopyCubeInParams)->GetBatchNum() *
                MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleSizeAlign<false, INPUT_TYPE::TAG == InputTypeTag::B>(), 1);
        }
    }

    __aicore__ inline void BatchLoad(LocalTensor<TransT>& dstTensor, const uint32_t matrixStride,
                                     const int32_t outerIdx, const int32_t splitIdx, const int32_t splitSize)
    {
        if (MATMUL_MODULE(BatchCopyCubeInParams)->IsTranspose()) {
            CopyBatchToCube<true, INPUT_TYPE::TAG == InputTypeTag::A>(
                dstTensor, outerIdx, splitIdx, splitSize);
        } else {
            CopyBatchToCube<false, INPUT_TYPE::TAG == InputTypeTag::B>(
                dstTensor, outerIdx, splitIdx, splitSize);
        }
    }

private:
    constexpr static int32_t c0Size_ = AuxGetC0Size<TransT>();
    template <bool isTrans = false, bool isKRow = false>
    __aicore__ inline void CopyBatchToCube(LocalTensor<TransT>& dstTensor,
                                           const int32_t outerIdx, const int32_t splitIdx, const int32_t splitSize)
    {
        // 1. Calculate batch outer loop offset
        auto alignHeight = CeilAlign(MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleHeight<isTrans>(), BLOCK_CUBE);
        auto alignWidth = CeilAlign(MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleWidth<isTrans>(), c0Size_);
        int32_t batchNum = MATMUL_MODULE(BatchCopyCubeInParams)->GetBatchNum();
        int32_t batchNumIdx = batchNum / splitSize;
        bool iskRowDirec = isKRow && IsSameTypeV<TransT, int8_t>;

        // 2. Calculate src and dst stride of one step
        auto srcStride = alignWidth * alignHeight;
        auto dstStride = MATMUL_MODULE(BatchCopyCubeInParams)->template GetSingleSizeAlign<isTrans, isKRow>();
        int64_t srcOffset = batchNumIdx * splitIdx * srcStride;
        int64_t dstOffset = batchNumIdx * splitIdx * dstStride;
        // if odd ground, the first block is unequal with the second block
        auto batchBlock = splitIdx == 0 ? batchNumIdx : batchNum - batchNumIdx;

        // 3. set input srctensor addr
        auto batchOffset = outerIdx * MATMUL_MODULE(BatchCopyCubeInParams)->GetBatchNum() * srcStride;
        using TensorType =
            typename AscendC::Conditional<PhyPosIsGM(INPUT_TYPE::pos), GlobalTensor<SrcT>, LocalTensor<SrcT>>::type;
        TensorType srcTensor;
        if constexpr (PhyPosIsGM(INPUT_TYPE::pos)) {
            srcTensor.SetGlobalBuffer(MATMUL_MODULE(MatmulTensorInfo)->GetGlobalTensor().address_);
            srcTensor.SetAddr(batchOffset);
        } else {
            srcTensor.SetAddr(MATMUL_MODULE(MatmulTensorInfo)->GetLocalTensor().address_);
            srcTensor = srcTensor[batchOffset];
        }

        // 4. loop copy NZ data by batch
        for (auto i = 0; i < batchBlock; ++i) {
            MATMUL_MODULE(DataCopyWrapper)->CopyNZ2NZ(dstTensor[dstOffset], srcTensor[srcOffset], 0, 0,
                alignHeight, alignWidth, alignHeight, iskRowDirec);
            dstOffset += dstStride;
            srcOffset += srcStride;
        }
    }
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif
#if defined(__UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_DETAIL_MATMUL_STAGE_COPY_CUBE_IN_BATCH_BATCH_COPY_CUBE_IN_USING_UB_H__)
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_DETAIL_MATMUL_STAGE_COPY_CUBE_IN_BATCH_BATCH_COPY_CUBE_IN_USING_UB_H__
#endif // IMPL_MATMUL_STAGE_COPY_CUBE_IN_BATCH_BATCH_COPY_CUBE_IN_USING_UB_H