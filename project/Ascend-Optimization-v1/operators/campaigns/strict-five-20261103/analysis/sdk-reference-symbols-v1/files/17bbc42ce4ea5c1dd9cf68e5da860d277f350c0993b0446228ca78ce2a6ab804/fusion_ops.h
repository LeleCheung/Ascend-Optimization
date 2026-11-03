/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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

/*!
 * \file fusion_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_

#ifndef OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_WARN_SHOWN
#define OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_WARN_SHOWN
#warning "fusion_ops.h is scheduled to be deprecated in December 2026, and will be replaced by the \
ops_proto_math.h, ops_proto_cv.h, ops_proto_nn.h, ops_proto_transformer.h, ops_proto_legacy.h. \
We apologize for any inconvenience caused and appreciate your timely migration to the new interface."

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Fast and Memory-Efficient Exact Attention with IO-Awareness.

* @par Inputs:
* twelve inputs, including:
* @li query: A matrix Tensor. The type support float16, bf16, float32.
* @li key: A matrix Tensor. The type support float16, bf16, float32.
* @li value: A matrix Tensor. The type support float16, bf16, float32.
* @li real_shift: A matrix Tensor. An optional input parameter. The type support float16, bf16, float32.
* @li drop_mask: A matrix Tensor. An optional input parameter. The type support uint8.
* @li padding_mask: A matrix Tensor. An optional input parameter. The type support float16, bf16, float32.
* @li atten_mask: A matrix Tensor. An optional input parameter. The type support bool, uint8.
* @li prefix: A matrix Tensor. An optional input parameter. The type support int64.
* @li actual_seq_qlen: A matrix Tensor. An optional input parameter. The type support int64. If used,
* layout need to be setted TND. ex. If the q seqlen is [2,2,2,2,2], this parameter need be setted [2,4,6,8,10].
* @li actual_seq_kvlen: A matrix Tensor. An optional input parameter. The type support int64. If used,
* layout need to be setted TND. ex. If the kv seqlen is [2,2,2,2,2], this parameter need be setted [2,4,6,8,10].
* @li q_start_idx: A matrix Tensor. An optional input parameter. The type support int64.
* @li kv_start_idx: A matrix Tensor. An optional input parameter. The type support int64.
* @li d_scale_q: A matrix Tensor. An optional input parameter. The type support float32.
* @li d_scale_k: A matrix Tensor. An optional input parameter. The type support float32.
* @li d_scale_v: A matrix Tensor. An optional input parameter. The type support float32.
* @li queryRope: A matrix Tensor. An optional input parameter. The type support float16, bf16, float32.
* @li keyRope: A matrix Tensor. An optional input parameter. The type support float16, bf16, float32.
* @li sink: A matrix Tensor. An optional input parameter. The type support float32.

* @par Attributes:
* @li scale_value: A float. The scale value. Default: 1.0.
* @li keep_prob: A float. The keep probability of dropout. Default: 1.0.
* @li pre_tockens: An int. Previous tokens.
* @li next_tockens: An int. Next tokens.
* @li head_num: An int. A required attribute. The number of the heads.
* @li input_layout: A string. A required attribute. Specifies the layout of `query`, the value must be one of ["BSH",
* "SBH", "BNSD", "BSND", "TND"].
* @li inner_precise: An int. 0, 1, reserved value. 2, support invalid lines.
* @li sparse_mode: An int. 0, defaultMsk. 1, allMask. 2, leftUpCausal. 3, rightDownCausal. 4, band. 5, prefix.
* 6, prefixCompress. 7, rightDownCausalBand. 8, bandLeftUpCausal.
* @li pse_type: An int. Optional attribute. Users can pass in 1 if they do not specify it.
* The supported configuration values ​​are 0, 1, 2, and 3.
* @li seed: An int. Optional attribute. Default: 0.
* @li offset: An int. Optional attribute.  Default: 0.
* @li out_dtype: An int. Optional attribute.  Default: 0.
* @li softmax_out_layout: A string. Optional attribute.  Default: "", the value must be one of ["", "same_as_input"].

* @par Outputs:
* @li softmax_max: A matrix Tensor. The type support float32.
* @li softmax_sum: A matrix Tensor. The type support float32.
* @li softmax_out: A matrix Tensor. The type support float16, bf16, float32.
* @li attention_out: A matrix Tensor. The type support float16, bf16, float32.
*/
REG_OP(FlashAttentionScore)
    .INPUT(query, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .INPUT(key, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .INPUT(value, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(real_shift, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(drop_mask, TensorType({DT_UINT8}))
    .OPTIONAL_INPUT(padding_mask, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_BOOL, DT_UINT8}))
    .OPTIONAL_INPUT(prefix, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_qlen, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_kvlen, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(q_start_idx, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(kv_start_idx, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(d_scale_q, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(d_scale_k, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(d_scale_v, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(query_rope, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(key_rope, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(sink, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(p_scale, TensorType({DT_FLOAT32}))
    .OUTPUT(softmax_max, TensorType({DT_FLOAT32}))
    .OUTPUT(softmax_sum, TensorType({DT_FLOAT32}))
    .OUTPUT(softmax_out, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .ATTR(scale_value, Float, 1.0)
    .ATTR(keep_prob, Float, 1.0)
    .ATTR(pre_tockens, Int, 2147483647)
    .ATTR(next_tockens, Int, 2147483647)
    .REQUIRED_ATTR(head_num, Int)
    .REQUIRED_ATTR(input_layout, String)
    .ATTR(inner_precise, Int, 0)
    .ATTR(sparse_mode, Int, 0)
    .ATTR(pse_type, Int, 1)
    .ATTR(seed, Int, 0)
    .ATTR(offset, Int, 0)
    .ATTR(out_dtype, Int, 0)
    .ATTR(softmax_out_layout, String, "")
    .OP_END_FACTORY_REG(FlashAttentionScore)

/**
*@brief Data Scanning Operator for the Attention Side in Decoupled Attention-FFN Deployment Scenarios. This operator
polls data from FFNToAttention to confirm its readiness state.
*@par Inputs:
*@li  schedule_context: A tensor. The type support int8. Its shape must be 1D (1024). Format: ND

*@par Outputs:
*schedule_context: A ND Tensor that holds the new value of schedule_context after the value has been assigned.
*Has the same shape and dtype and format as the input "schedule_context". \n
*/
REG_OP(AttentionWorkerScheduler)
    .INPUT(schedule_context, TensorType({DT_INT8}))
    .OUTPUT(schedule_context, TensorType({DT_INT8}))
    .OP_END_FACTORY_REG(AttentionWorkerScheduler)

/**
* @brief Implement incremental inference based on full inference.

* @par Inputs:
* @li query: A matrix Tensor. The type support float16, bf16, int8.
* @li key: It's a dynamic input. A matrix Tensor. The type support float16, bf16, int8.
* @li value: It's a dynamic input. A matrix Tensor. The type support float16, bf16, int8.
* @li pse_shift: A matrix Tensor. Position coding inside the attention structure. The type support float16, bf16.
* @li atten_mask: A matrix Tensor. Mask the result of multiplying query by key to indicate whether to calculate the correlation between tokens.
* The type support bool, int8, uint8.
* @li actual_seq_lengths: A matrix Tensor. Indicates the valid sequence length of the key/value in different batches.
* The type support int64.
* @li dequant_scale1: A matrix Tensor. Dequantization factor after multiplying query by key.
* The type support uint64, float32.
* @li quant_scale1: A matrix Tensor. Indicates the quantization factor before multiplying query by key.
* The type support float32.
* @li dequant_scale2: A matrix Tensor. Dequantization factor after multiplying the result of softmax by value.
* The type support uint64, float32.
* @li quant_scale2: A matrix Tensor. Quantization factor of the output. The type support float32, bf16.
* @li quant_offset2: A matrix Tensor. Indicates the quantization offset of the output. The type support float32, bf16.
* @li antiquant_scale: A matrix Tensor. Indicates the antiquant factor. The type support float16, bf16.
* @li antiquant_offset: A matrix Tensor. Indicates the antiquant offset. The type support float16, bf16.
* @li block_table: A matrix Tensor. Indicates the block mapping table used by KV storage in PageAttention.
* The type support int32.
* @li kv_padding_size: A matrix Tensor. Indicates whether the data of each batch in the key/value is
* right-aligned and the number of right-aligned data.The type support int64.

* @par Attributes:
* @li num_heads: A required int. The number of the heads.
* @li scale_value: An optional float. The scale value. Default: 1.0.
* @li input_layout: An optional string. Specifies the layout of query, the value must be one of ["BSH", "BNSD", "BSND"]. Default: "BSH".
* @li num_key_value_heads: An optional int. Key value num heads. Default: 1.
* @li block_size: An optional int. Max length in pageattention's kv block. Default: 0.
* @li inner_precise: An optional int. When innerPrecise is 0, the high-precision mode is used.
* When innerPrecise is 1, the high-performance mode is used. Default: 1.

* @par Outputs:
* attention_out: A matrix Tensor. The type support float16, bf16, int8. \n

* @attention Constraints:
* - Constraints for empty Input:
* @li Direct return if query is empty.
* @li If query exists, key and value are empty: output a zero-filled tensor of corresponding shape.
* @li AscendCLNN framework handles if attention_out is an empty tensor.
* @li No processing for parameters marked as can pass nullptr if they are null pointers.
* @li Key and value tensor shapes must match; batch in non-continuous scenarios can only be 1.
*
* - Constraints for int8 quantization:
* @li Specific parameter existence and data format requirements based on input/output data formats.
* @li Both input and output int8: need deqScale1, quantScale1, deqScale2, quantScale2.
* @li Input int8, output float16: need deqScale1, quantScale1, deqScale2; error if quantOffset2 or quantScale2 not nullptr.
* @li Input float16/bf16, output int8: only quantScale2 needs to exist.
*
* - Constraints for antiquant:
* @li Support per-tensor/per-channel formats and float32/bf16 data types.
* @li Types and shape of quantScale2 and quantOffset2 need to be consistent.
* @li Specific recommendations for quantScale2 shape based on input data type and output layout.
* @li Support per-channel, per-tensor, and per-token modes, and symmetric/asymmetric quantization.
* @li Per-channel mode: shape supports (2, N, 1, D), (2, N, D), (2, H); data type matches query; antiquantMode set to 0.
* @li Per-tensor mode: shape (2), data type matches query; antiquantMode set to 0.
* @li Per-token mode: shape (2, B, S), data type float32; antiquantMode set to 1.
* @li Symmetric quantization: antiquantOffset can be empty; if empty, symmetric quantization is performed.
* @li Asymmetric quantization: both antiquantScale and antiquantOffset need to exist.
*/
REG_OP(IncreFlashAttention)
    .INPUT(query, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .DYNAMIC_INPUT(key, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .DYNAMIC_INPUT(value, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OPTIONAL_INPUT(pse_shift, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_BOOL, DT_INT8, DT_UINT8}))
    .OPTIONAL_INPUT(actual_seq_lengths, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(dequant_scale1, TensorType({DT_UINT64, DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale1, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(dequant_scale2, TensorType({DT_UINT64, DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale2, TensorType({DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(quant_offset2, TensorType({DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(block_table, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(kv_padding_size, TensorType({DT_INT64}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .REQUIRED_ATTR(num_heads, Int)
    .ATTR(scale_value, Float, 1.0)
    .ATTR(input_layout, String, "BSH")
    .ATTR(num_key_value_heads, Int, 1)
    .ATTR(block_size, Int, 0)
    .ATTR(inner_precise, Int, 1)
    .OP_END_FACTORY_REG(IncreFlashAttention)


/**
* @brief Implement MlaProlog.

* @par Inputs:
* @li token_x: A matrix Tensor. The type support int8 and bf16.
* @li weight_dq: A matrix Tensor. The downsampling weight matrix of query. The type support int8 and bf16.
* @li weight_uq_qr: A matrix Tensor. The upsampling and positional encoding weight matrix of query.
* The type support int8 and bf16.
* @li weight_uk: A matrix Tensor. The second upsampling weight matrix of query. The type support int8 and bf16.
* @li weight_dkv_kr: A matrix Tensor. The upsampling and positional encoding weight matrix of key.
* The type support int8 and bf16.
* @li rmsnorm_gamma_cq: A matrix Tensor. The gamma factor for the rmsnorm of query. The type support float16 and bf16.
* @li rmsnorm_gamma_ckv: A matrix Tensor. The gamma factor for the rmsnorm of key. The type support float16 and bf16.
* @li rope_sin: A matrix Tensor. The position encoding sin information of each token. The type support float16 and bf16.
* @li rope_cos: A matrix Tensor. The position encoding cos information of each token. The type support float16 and bf16.
* @li cache_index: A matrix Tensor. The index of the cache in each batch. The type support int64.
* @li kv_cache: A matrix Tensor. The type support float16 and bf16.
* @li kr_cache: A matrix Tensor. The type support float16 and bf16.
* @li dequant_scale_x: A matrix Tensor. This parameter is used for dequantization after downsampling when tokenX is of the int8 type. The quantization mode of tokenX is per-token.
* The type support float.
* @li dequant_scale_w_dq: A matrix Tensor. This parameter is used for dequantization after downsampling when tokenX is of the int8 type. The quantization mode is per-channel.
* The type support float.
* @li dequantScaleWUqQr: A matrix Tensor. Parameter used for dequantization after matrix multiplication during dynamic quantization of MatmulQcQr.
* The type support float.
* @li dequant_scale_w_dkv_kr: A matrix Tensor. This parameter is used for quantization after MatmulCkvKr when tokenX is of the int8 type.
* The type support float.
* @li quantScaleCkv: A matrix Tensor. Parameter used for quantizing the RmsNormCkv output. The parameter is aclTensor on the device side.
* The type support float.
* @li quantScaleCkr: A matrix Tensor. This parameter is used for quantizing the RoPEKr output. It is aclTensor on the device side.
* The type support float.
* @li smoothScalesCq: A matrix Tensor. Smoothquant parameter required for dynamic quantization of RmsNormDq output.

* @par Attributes:
* @li rmsnorm_epsilon_cq: An optional float. The epsilon factor for the rmsnorm of query. Default: 1e-5.
* @li rmsnorm_epsilon_ckv: An optional float. The epsilon factor for the rmsnorm of key. Default: 1e-5.
* @li cache_mode: An optional int. The mode of kvcache. Default: PA_BSND.

* @par Outputs:
* query: A matrix Tensor. The type support float16 and bf16.
* query_rope: A matrix Tensor. The type support float16 and bf16.
* kv_cache_out: A matrix Tensor. The type support float16 and bf16.
* kr_cache_out: A matrix Tensor. The type support float16 and bf16.\n

* @attention Constraints:
*
*/
REG_OP(MlaProlog)
    .INPUT(token_x, TensorType({DT_INT8, DT_BF16}))
    .INPUT(weight_dq, TensorType({DT_INT8, DT_BF16}))
    .INPUT(weight_uq_qr, TensorType({DT_INT8, DT_BF16}))
    .INPUT(weight_uk, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(weight_dkv_kr, TensorType({DT_INT8, DT_BF16}))
    .INPUT(rmsnorm_gamma_cq, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(rmsnorm_gamma_ckv, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(rope_sin, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(rope_cos, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(cache_index, TensorType({DT_INT64}))
    .INPUT(kv_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .INPUT(kr_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OPTIONAL_INPUT(dequant_scale_x, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(dequant_scale_w_dq, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(dequant_scale_w_uq_qr, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(dequant_scale_w_dkv_kr, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale_ckv, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale_ckr, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(smooth_scales_cq, TensorType({DT_FLOAT}))
    .OUTPUT(query, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OUTPUT(query_rope, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OUTPUT(kv_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OUTPUT(kr_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .ATTR(rmsnorm_epsilon_cq, Float, 1e-05)
    .ATTR(rmsnorm_epsilon_ckv, Float, 1e-05)
    .ATTR(cache_mode, String, "PA_BSND")
    .OP_END_FACTORY_REG(MlaProlog)

/**
* @brief
* Implement the pre-calculation of Multi-Head Latent Attention.

* @par Inputs:
* @li token_x: A matrix tensor. The type support int8 and bfloat16.
* @li weight_dq: A matrix tensor. The downsampling weight matrix of query. The type support int8 and bfloat16.
* @li weight_uq_qr: A matrix tensor. The upsampling and positional encoding weight matrix of query.
* The type support int8 and bfloat16.
* @li weight_uk: A matrix tensor. The second upsampling weight matrix of query. The type support float16 and bfloat16.
* @li weight_dkv_kr: A matrix tensor. The upsampling and positional encoding weight matrix of key.
* The type support int8 and bfloat16.
* @li rmsnorm_gamma_cq: A matrix tensor. The gamma factor for the rmsnorm of query. The type support float16 and bfloat16.
* @li rmsnorm_gamma_ckv: A matrix tensor. The gamma factor for the rmsnorm of key. The type support float16 and bfloat16.
* @li rope_sin: A matrix tensor. The position encoding sin information of each token. The type support float16 and bfloat16.
* @li rope_cos: A matrix tensor. The position encoding cos information of each token. The type support float16 and bfloat16.
* @li cache_index: A matrix tensor. The index of the cache in each batch. The type support int64.
* @li kv_cache: A matrix tensor, representing the cache of kv matrix. The type support float16 and bfloat16 and int8.
* @li kr_cache: A matrix tensor, representing the cache of kv postion embedding matrix. The type support float16 and bfloat16 and int8.
* @li dequant_scale_x: A matrix tensor. This parameter is used for dequantization after downsampling when tokenX is of the int8 type. The quantization mode of tokenX is per-token.
* The type support float32.
* @li dequant_scale_w_dq: A matrix tensor. This parameter is used for dequantization after downsampling when tokenX is of the int8 type. The quantization mode is per-channel.
* The type support float32.
* @li dequant_scale_w_uq_qr: A matrix tensor. Parameter used for dequantization after matrix multiplication during dynamic quantization of MatmulQcQr.
* The type support float32.
* @li dequant_scale_w_dkv_kr: A matrix tensor. This parameter is used for quantization after MatmulCkvKr when tokenX is of the int8 type.
* The type support float32.
* @li quant_scale_ckv: A matrix tensor. Parameter used for quantizing the RmsNormCkv output. The parameter is aclTensor on the device side.
* The type support float32.
* @li quant_scale_ckr: A matrix tensor. This parameter is used for quantizing the RoPEKr output. It is aclTensor on the device side.
* The type support float32.
* @li smooth_scales_cq: A matrix tensor. Smoothquant parameter required for dynamic quantization of RmsNormDq output.

* @par Attributes:
* @li rmsnorm_epsilon_cq: An optional float32. The epsilon factor for the rmsnorm of query. Default: 1e-5.
* @li rmsnorm_epsilon_ckv: An optional float32. The epsilon factor for the rmsnorm of key. Default: 1e-5.
* @li cache_mode: An optional int. The mode of kvcache. The type support PA_NZ and PA_BSND. Default: PA_BSND.
* PA stands for page attention. This means kv_cache and kr_cache are stored in the page attention format, and updated by the order of BSND or NZ.

* @par Outputs:
* @li query: A matrix tensor, representing the query for Multi-Head Latent Attention. The type support float16 and bfloat16 and int8.
* @li query_rope: A matrix tensor, representing the position embedding of query. The type support float16 and bfloat16 and int8.
* @li kv_cache: A matrix tensor, representing the updated kv cache. This parameter uses the same memory of kv_cache. The type support float16 and bfloat16 and int8.
* @li kr_cache: A matrix tensor, representing the updated kr cache. This parameter uses the same memory of kr_cache. The type support float16 and bfloat16 and int8.
* @li dequant_scale_q_nope: A matrix tensor, representing the dequant weights for query if query is quantilized. The type support float32.

*
*/
REG_OP(MlaPrologV2)
.INPUT(token_x, TensorType({DT_INT8, DT_BF16}))
.INPUT(weight_dq, TensorType({DT_INT8, DT_BF16}))
.INPUT(weight_uq_qr, TensorType({DT_INT8, DT_BF16}))
.INPUT(weight_uk, TensorType({DT_FLOAT16, DT_BF16}))
.INPUT(weight_dkv_kr, TensorType({DT_INT8, DT_BF16}))
.INPUT(rmsnorm_gamma_cq, TensorType({DT_FLOAT16, DT_BF16}))
.INPUT(rmsnorm_gamma_ckv, TensorType({DT_FLOAT16, DT_BF16}))
.INPUT(rope_sin, TensorType({DT_FLOAT16, DT_BF16}))
.INPUT(rope_cos, TensorType({DT_FLOAT16, DT_BF16}))
.INPUT(cache_index, TensorType({DT_INT64}))
.INPUT(kv_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
.INPUT(kr_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
.OPTIONAL_INPUT(dequant_scale_x, TensorType({DT_FLOAT}))
.OPTIONAL_INPUT(dequant_scale_w_dq, TensorType({DT_FLOAT}))
.OPTIONAL_INPUT(dequant_scale_w_uq_qr, TensorType({DT_FLOAT}))
.OPTIONAL_INPUT(dequant_scale_w_dkv_kr, TensorType({DT_FLOAT}))
.OPTIONAL_INPUT(quant_scale_ckv, TensorType({DT_FLOAT}))
.OPTIONAL_INPUT(quant_scale_ckr, TensorType({DT_FLOAT}))
.OPTIONAL_INPUT(smooth_scales_cq, TensorType({DT_FLOAT}))
.OUTPUT(query, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
.OUTPUT(query_rope, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
.OUTPUT(kv_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
.OUTPUT(kr_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
.OUTPUT(dequant_scale_q_nope, TensorType({DT_FLOAT}))
.ATTR(rmsnorm_epsilon_cq, Float, 1e-05)
.ATTR(rmsnorm_epsilon_ckv, Float, 1e-05)
.ATTR(cache_mode, String, "PA_BSND")
.OP_END_FACTORY_REG(MlaPrologV2)

/**
* @brief
* Implement the pre-calculation of Multi-Head Latent Attention.

* @par Inputs:
* @li token_x: A matrix tensor. The type support int8 and bfloat16 and float8_e4m3.
* @li weight_dq: A matrix tensor. The downsampling weight matrix of query. The type support int8 and bfloat16 and float8_e4m3.
* @li weight_uq_qr: A matrix tensor. The upsampling and positional encoding weight matrix of query.
* The type support int8 and bfloat16 and float8_e4m3.
* @li weight_uk: A matrix tensor. The second upsampling weight matrix of query. The type support float16 and bfloat16.
* @li weight_dkv_kr: A matrix tensor. The upsampling and positional encoding weight matrix of key.
* The type support int8 and bfloat16 and float8_e4m3.
* @li rmsnorm_gamma_cq: A matrix tensor. The gamma factor for the rmsnorm of query. The type support float16 and bfloat16.
* @li rmsnorm_gamma_ckv: A matrix tensor. The gamma factor for the rmsnorm of key. The type support float16 and bfloat16.
* @li rope_sin: A matrix tensor. The position encoding sin information of each token. The type support float16 and bfloat16.
* @li rope_cos: A matrix tensor. The position encoding cos information of each token. The type support float16 and bfloat16.
* @li kv_cache: A matrix tensor, representing the cache of kv matrix. The type support float16 and bfloat16 and int8 and float8_e4m3.
* @li kr_cache: A matrix tensor, representing the cache of kv postion embedding matrix. The type support float16 and bfloat16 and int8.
* @li cache_index: A matrix tensor. The index of the cache in each batch. The type support int64.
* @li dequant_scale_x: A matrix tensor. This parameter is used for dequantization after downsampling when tokenX is of the int8 type. The quantization mode of tokenX is per-token.
* The type support float32 and float8_e8m0.
* @li dequant_scale_w_dq: A matrix tensor. This parameter is used for dequantization after downsampling when tokenX is of the int8 type. The quantization mode is per-channel.
* The type support float32 and float8_e8m0.
* @li dequant_scale_w_uq_qr: A matrix tensor. Parameter used for dequantization after matrix multiplication during dynamic quantization of MatmulQcQr.
* The type support float32 and float8_e8m0.
* @li dequant_scale_w_dkv_kr: A matrix tensor. This parameter is used for quantization after MatmulCkvKr when tokenX is of the int8 type.
* The type support float32 and float8_e8m0.
* @li quant_scale_ckv: A matrix tensor. Parameter used for quantizing the RmsNormCkv output. The parameter is aclTensor on the device side.
* The type support float32.
* @li quant_scale_ckr: A matrix tensor. This parameter is used for quantizing the RoPEKr output. It is aclTensor on the device side.
* The type support float32.
* @li smooth_scales_cq: A matrix tensor. Smoothquant parameter required for dynamic quantization of RmsNormDq output.
* The type support float32.
* @li actual_seq_len: A matrix tensor. Currently reserved. The type support int32.

* @par Attributes:
* @li rmsnorm_epsilon_cq: An optional float32. The epsilon factor for the rmsnorm of query. Default: 1e-5.
* @li rmsnorm_epsilon_ckv: An optional float32. The epsilon factor for the rmsnorm of key. Default: 1e-5.
* @li cache_mode: An optional int. The mode of kvcache. The type support PA_NZ and PA_BSND. Default: PA_BSND.
* PA stands for page attention. This means kv_cache and kr_cache are stored in the page attention format, and updated by the order of BSND or NZ.
* @li query_norm_flag: An optional bool. Currently reserved. Default: false.
* @li weight_quant_mode: An optional int. Currently reserved. Default: 0.
* @li kv_cache_quant_mode: An optional int. Currently reserved. Default: 0.
* @li query_quant_mode: An optional int. Currently reserved. Default: 0.
* @li ckvkr_repo_mode: An optional int. Currently reserved. Default: 0.
* @li quant_scale_repo_mode: An optional int. Currently reserved. Default: 0.
* @li tile_size: An optional int. Currently reserved. Default: 128.
* @li k_nope_clip_alpha: An optional float32. Currently reserved. Default: 1.0.
* @li qc_qr_scale: An optional float32. The correction scale of query. Default: 1.0.
* @li kc_scale: An optional float32. The correction scale of key. Default: 1.0.

* @par Outputs:
* @li query: A matrix tensor, representing the query for Multi-Head Latent Attention. The type support float16 and bfloat16 and int8 and float8_e4m3.
* @li query_rope: A matrix tensor, representing the position embedding of query. The type support float16 and bfloat16 and int8.
* @li kv_cache: A matrix tensor, representing the updated kv cache. This parameter uses the same memory of kv_cache. The type support float16 and bfloat16 and int8 and float8_e4m3.
* @li kr_cache: A matrix tensor, representing the updated kr cache. This parameter uses the same memory of kr_cache. The type support float16 and bfloat16 and int8.
* @li dequant_scale_q_nope: A matrix tensor, representing the dequant weights for query if query is quantilized. The type support float32.
* @li query_norm: A matrix tensor, currently reserved. The type support bfloat16 and int8 and float8_e4m3.
* @li dequant_scale_q_norm: A matrix tensor, currently reserved. The type support float32 and float8_e8m0.

*
*/
REG_OP(MlaPrologV3)
    .INPUT(token_x, TensorType({DT_INT8, DT_BF16, DT_FLOAT8_E4M3FN}))
    .INPUT(weight_dq, TensorType({DT_INT8, DT_BF16, DT_FLOAT8_E4M3FN}))
    .INPUT(weight_uq_qr, TensorType({DT_INT8, DT_BF16, DT_FLOAT8_E4M3FN}))
    .INPUT(weight_uk, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(weight_dkv_kr, TensorType({DT_INT8, DT_BF16, DT_FLOAT8_E4M3FN}))
    .INPUT(rmsnorm_gamma_cq, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(rmsnorm_gamma_ckv, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(rope_sin, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(rope_cos, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(kv_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT8_E4M3FN}))
    .INPUT(kr_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OPTIONAL_INPUT(cache_index, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(dequant_scale_x, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(dequant_scale_w_dq, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(dequant_scale_w_uq_qr, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(dequant_scale_w_dkv_kr, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(quant_scale_ckv, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale_ckr, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(smooth_scales_cq, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(actual_seq_len, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(k_nope_clip_alpha, TensorType({DT_FLOAT}))
    .OUTPUT(query, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT8_E4M3FN}))
    .OUTPUT(query_rope, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OUTPUT(kv_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT8_E4M3FN}))
    .OUTPUT(kr_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OUTPUT(dequant_scale_q_nope, TensorType({DT_FLOAT}))
    .OUTPUT(query_norm, TensorType(({DT_INT8, DT_BF16, DT_FLOAT8_E4M3FN})))
    .OUTPUT(dequant_scale_q_norm, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .ATTR(rmsnorm_epsilon_cq, Float, 1e-05)
    .ATTR(rmsnorm_epsilon_ckv, Float, 1e-05)
    .ATTR(cache_mode, String, "PA_BSND")
    .ATTR(query_norm_flag, Bool, false)
    .ATTR(weight_quant_mode, Int, 0)
    .ATTR(kv_cache_quant_mode, Int, 0)
    .ATTR(query_quant_mode, Int, 0)
    .ATTR(ckvkr_repo_mode, Int, 0)
    .ATTR(quant_scale_repo_mode, Int, 0)
    .ATTR(tile_size, Int, 128)
    .ATTR(qc_qr_scale, Float, 1.0)
    .ATTR(kc_scale, Float, 1.0)
    .OP_END_FACTORY_REG(MlaPrologV3)

/**
* @brief Compute the GeGluV2,
* where the activations function in GLU is Gelu.

* @par Inputs:
* x: A Tensor. Must be one of the following types: bfloat16, float16, float32.
* Shape supports at least 1 dimensions, and at most 8 dimensions.
* The length of the split dimension in x must be an even number.

* @par Outputs:
* Two outputs, including:
* @li y: A Tensor. Must be one of the following types: bfloat16, float16, float32.
* The dtype of y must exactly same with input x.
* The shape of y matches the shape of x in all dimensions except for the split dimension,
* where its length is half of length of x's split dimension.
* @li gelu: A Tensor. Must be one of the following types: bfloat16, float16, float32.
* The dtype of gelu must exactly same with input x.
* The shape of gelu matches the shape of x in all dimensions except for the split dimension,
* where its length is half of length of x's split dimension.

* @par Attributes:
* Three attributes, including:
* @li dim: An optional int. The dimension to be split, default is -1.
* @li approximate: An optional int. Which formula used for the activation computation.
* The gelu approximation algorithm to use: 'none'(0) or 'tanh'(1), default is 'tanh'(1).
* Atlas Inference Series Product only support 'tanh'(1).
* @li activate_left: An optional bool.
* The gelu activate_left algorithm to use:
*     'false'(activate right) or 'true'(activate left), defalut is 'false'(activate right).
*/
REG_OP(GeGluV2)
    .INPUT(x, "T")
    .OUTPUT(y, "T")
    .OUTPUT(gelu, "T")
    .DATATYPE(T, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .ATTR(dim, Int, -1)
    .ATTR(approximate, Int, 1)
    .ATTR(activate_left, Bool, false)
    .OP_END_FACTORY_REG(GeGluV2)

/**
* @brief Computes the gradient of x through GeGluV2.
*
* @par Inputs:
* Three inputs, including:
* @li dy: A Tensor of the same type as "x".
      The shape of dy matches the shape of x in all dimensions except for the split dimension,
      where its length is half of the length of x's split dimension.
* @li x: A Tensor. Must be one of the following types: float16, bfloat16, float32.
      Shape supports at least 1 dimension, and at most 8 dimensions.
      The length of the split dimension in x must be an even number.
* @li gelu: A Tensor of the same type as "x".
      The shape of gelu matches the shape of x in all dimensions except for the split dimension,
      where its length is half of the length of x's split dimension.
*
* @par Outputs:
* dx: A Tensor. Has the same type as "x". Shape should be same as x.
*
* @par Attributes:
* @li dim: An optional Int. The dimension to be split, default is -1.
* @li approximate: An optional Int. Determines which formula to use for the activation computation.
* The gelu grad approximation algorithm to use: 0('none') or 1('tanh'), default is 1('tanh').
* Atlas Inference Series Product only supports 'tanh'(1).
* @li activate_left: An optional Bool.
* Whether the left side of x is used as an input parameter to the activation function,
* default is false, use the right side.
*
* @par Third-party framework compatibility
* Compatible with the Pytorch operator GeGluGradV2.
*
*/
REG_OP(GeGluGradV2)
    .INPUT(dy, "T")
    .INPUT(x, "T")
    .INPUT(gelu, "T")
    .OUTPUT(dx, "T")
    .DATATYPE(T, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .ATTR(dim, Int, -1)
    .ATTR(approximate, Int, 1)
    .ATTR(activate_left, Bool, false)
    .OP_END_FACTORY_REG(GeGluGradV2)

/**
* @brief Function PromptFlashAttention.

* @par Inputs:
* @li query: A matrix Tensor. The type support float16, bf16, int8.
* @li key: A matrix Tensor. The type support float16, bf16, int8.
* @li value: A matrix Tensor. The type support float16, bf16, int8.
* @li pse_shift: A matrix Tensor. The type support float16, bf16.
* @li atten_mask: A matrix Tensor. The type support float16, bool, int8, uint8.
* @li actual_seq_lengths: A Tensor. The type support int64.
* @li actual_seq_lengths_kv: A Tensor. The type support int64.
* @li deq_scale1: A Tensor. The type support uint64, float32.
* @li quant_scale1: A Tensor. The type support float32.
* @li deq_scale2: A Tensor. The type support uint64, float32.
* @li quant_scale2: A Tensor. The type support float32, bf16.
* @li quant_offset2: A Tensor. The type support float32, bf16.

* @par Attributes:
* @li num_heads: An int. The number of the heads.
* @li scale_value: A float. The scale value. Default: 1.0.
* @li pre_tokens: An int. Previous tokens. Default: 214748647.
* @li next_tokens: An int. Next tokens. Default: 0.
* @li input_layout: A string. Specifies the layout of `query`, the value must be one of ["BSH", "BNSD", "BSND", "BNSD_BSND"]. Default: "BSH".
* @li num_key_value_heads: Key value num heads. Default: 0.
* @li sparse_mode: Sparse mode. Default: 0.
* @li inner_precise: An int. 0, float16 high precision. 1, high performance. Default: 1.

* @par Outputs:
* @li attention_out: A matrix Tensor. The type support float16, bf16, int8.

* @attention Constraints:
* @li Ensure CANN and PyTorch package version compatibility when using this interface with PyTorch.
* @li Handle empty input: If 'query' is empty, return directly. If 'query' is non-empty and 'key', 'value' are empty tensors (S2=0), fill 'attention_out' with zeros of the corresponding shape.
* If 'attention_out' is an empty tensor, AscendCLNN will process it.
* @li The 'sparseMode' parameter currently only supports values 0, 1, 2, 3, and 4; other values will cause an error.
* @li Output is INT8. 'quantOffset2' must be a non-empty pointer and tensor. 'sparseMode', 'preTokens', and 'nextTokens' must meet certain conditions.
* If some rows of the matrix do not participate in calculations, resulting in computational errors, this scenario will be blocked
* (solution: if you want this scenario not to be blocked, post-quantization operations should be performed outside the PFA interface, not enabled within).
* For `sparseMode = 0`, if `attenMask` is a non-empty pointer, the condition for interception is `actualSeqLengths - actualSeqLengthsKV - preTokens > 0` or `nextTokens < 0` per batch.
* For `sparseMode = 1` or `2`, no interception conditions are met.
* For `sparseMode = 3`, the condition for interception is `actualSeqLengthsKV - actualSeqLengths < 0` per batch.
* For `sparseMode = 4`, the condition for interception is `preTokens < 0` or `nextTokens + actualSeqLengthsKV - actualSeqLengths < 0` per batch.
*/
REG_OP(PromptFlashAttention)
    .INPUT(query, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .INPUT(key, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .INPUT(value, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OPTIONAL_INPUT(pse_shift, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_FLOAT16, DT_BOOL, DT_INT8, DT_UINT8}))
    .OPTIONAL_INPUT(actual_seq_lengths, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_lengths_kv, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(deq_scale1, TensorType({DT_UINT64, DT_FLOAT32}))
    .OPTIONAL_INPUT(quant_scale1, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(deq_scale2, TensorType({DT_UINT64, DT_FLOAT32}))
    .OPTIONAL_INPUT(quant_scale2, TensorType({DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(quant_offset2, TensorType({DT_FLOAT32, DT_BF16}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .REQUIRED_ATTR(num_heads, Int)
    .ATTR(scale_value, Float, 1.0)
    .ATTR(pre_tokens, Int, 214748647)
    .ATTR(next_tokens, Int, 0)
    .ATTR(input_layout, String, "BSH")
    .ATTR(num_key_value_heads, Int, 0)
    .ATTR(sparse_mode, Int, 0)
    .ATTR(inner_precise, Int, 1)
    .OP_END_FACTORY_REG(PromptFlashAttention)


/**
* @brief Function FusedInferAttentionScore.

* @par Inputs:
* @li query: A matrix tensor. The type support int8, float16, bfloat16, hifloat8, float8_e5m2, float8_e4m3fn.
* Query for attention structure.
* @li key: It's a dynamic input. A matrix tensor. The type support int8, float16, bfloat16, hifloat8, float8_e5m2, float8_e4m3fn, int4, float4_e1m2, float4_e2m1.
* Key for attention structure.
* @li value: It's a dynamic input. A matrix tensor. The type support int8, float16, bfloat16, hifloat8, float8_e5m2, float8_e4m3fn, int4, float4_e1m2, float4_e2m1.
* Value for attention structure.
* @li pse_shift: A matrix tensor. The type support float16, bfloat16.
* Postional encoding parameters inside the attention structure.
* @li atten_mask: A matrix tensor. The type support float16, bool, uint8, int8.
* Mask of the result QK, indicating whether to calculate the correlation between tokens.
* @li actual_seq_lengths: A matrix tensor. The type support int64.
* Efective sequence length of query in different batches.
* @li actual_seq_lengths_kv: A matrix tensor. The type support int64.
* Effective sequence length of key/value in different batches.
* @li dequant_scale1: A matrix tensor. The type support uint64, float32.
* Dequantization factor behind BMM1.
* @li quant_scale1: A matrix tensor. The type support float32.
* Quantification factor in front of BMM2.
* @li dequant_scale2: A matrix tensor. The type support uint64, float32.
* Dequantization factor behind BMM2
* @li quant_scale2: A matrix tensor. The type support float32, bfloat16.
* Quantization factor of the output.
* @li quant_offset2: A matrix tensor. The type support float32, bfloat16.
* Qantization offset of the output.
* @li antiquant_scale: A matrix tensor. The type support float16, bfloat16.
* Antiquantization factor.
* @li antiquant_offset: A matrix tensor. The type support float16, bfloat16.
* Antiquantization offset.
* @li block_table: A matrix tensor. The type support int32.
* The block mapping table used in KV storage of PageAttention.
* @li query_padding_size: A matrix tensor. The type support int64.
* Whether the data in each batch of query is right-aligned, and the number of right-aligned entries.
* @li kv_padding_size: A matrix tensor. The type support int64.
* Whether the data in each batch of key/value is right-aligned, and the number of right-aligned entries.
* @li key_antiquant_scale: A matrix tensor. The type support float16, bfloat16, float32.
* The dequantization factor of key when separating the KV antiquantization parameters.
* @li key_antiquant_offset: A matrix tensor. The type support float16, bfloat16, float32.
* The dequantization offset of key when separating the KV antiquantization parameters.
* @li value_antiquant_scale: A matrix tensor. The type support float16, bfloat16, float32.
* The dequantization factor of value when separating the KV antiquantization parameters.
* @li value_antiquant_offset: A matrix tensor. The type support float16, bfloat16, float32.
* The dequantization offset of value when separating the KV antiquantization parameters.
* @li key_shared_prefix: A matrix tensor. The type support int8, float16, bfloat16.
* The input of the system prefix part of key in the attention structure.
* @li value_shared_prefix: A matrix tensor. The type support int8, float16, bfloat16.
* The input of the system prefix part of value in the attention structure.
* @li actual_shared_prefix_len: A matrix tensor. The type support int64.
* Effective Sequence Length of key_shared_prefix/value_shared_prefix.
* @li query_rope: A matrix tensor. The type support int8, float16, bfloat16.
* The rope information of query in the MLA structure.
* @li key_rope: A matrix tensor. The type support int8, float16, bfloat16.
* The rope information of key in the MLA structure.
* @li key_rope_antiquant_scale: A matrix tensor. The type support float16, bfloat16.
* The dequantization factor of the rope information of key in the MLA structure.
* @li dequant_scale_query: A matrix tensor. The type support float32.
* The dequantization factor of query.
* @li learnable_sink: A matrix tensor. The type support bfloat16.
* The sink token factor of attention score.
* @li q_start_idx: A matrix tensor. The type support int64.
* start idx of q for alibi pse.
* @li kv_start_idx: A matrix tensor. The type support int64.
* start idx of kv for alibi pse.

* @par Attributes:
* @li num_heads: An int. The number of the heads.
* @li scale: A float. The scale value. Default: 1.0.
* @li pre_tokens: An int. Previous tokens. Default: 2147483647.
* @li next_tokens: An int. Next tokens. Default: 2147483647.
* @li input_layout: A string. Specifies the layout of `query`, the value must be one of ["BSH", "BNSD", "BSND", "BNSD_BSND"]. Default: "BSH".
* @li num_key_value_heads: key value num heads. Default: 0.
* @li sparse_mode: sparse mode. Default: 0.
* - 0: default mask
* - 1: all mask
* - 2: leftUpCausal mask
* - 3: rightDownCausal make
* - 4: band mask
* @li inner_precise: An int. 0, float16 high precision. 1, high performance. 2, high precision with row invalid fix. 3, high performance with row invalid fix. Default: 1.
* @li block_size: An int. Default: 0.
* In PageAttention, KV stores the maximum number of tokens in each block.
* @li antiquant_mode: An int. Antiquantization mode, 0: per-channel (per-channel includes per-tensor); 1: per-token. Default: 0.
* @li softmax_lse_flag: A bool. Whether to output softmax_lse. Default: false.
* @li key_antiquant_mode: An int. Antiquantization mode of key. Default: 0.
* - 0: per-channel (per-channel includes per-tensor)
* - 1: per-token
* - 2: per-tensor+per-head
* - 3: per-token+per-head
* - 4: per-token+using page attention mode to manage scale/offset
* - 5: per-token+per-head+using page attention mode to manage scale/offset
* - 6: per-token-group
* @li value_antiquant_mode: An int. Antiquantization mode of key. The mode number is the same as key_antiquant_mode. Default: 0.
* @li query_quant_mode: An int. Quantization mode of key. Only support mode 3: per-token+per-head now. Default: 0.
* @li pse_type: An int. Type of pse, 0: generate pse outside, 2: generate pse inside, 3: generate pse inside and do sqrt. Default: 0.
* @li out_dtype: An int. Dtype of attention_out tensor. Only supprt in PTA graph mode.
* - 5: fp16, 15: bf16, 23: fp8 e5m2, 24: fp8 e4m3, 290: hifp8. Default: 0.

* @par Outputs:
* @li attention_out: A matrix tensor. The type support float16, int8, bfloat16.
* The output of attention structure.
* @li softmax_lse: A matrix tensor. The type support float32.
* Ring attention takes the result of multiplying query by key, first finds the max to get softmax_max.
* The result of query multiplied by key minus softmax_max is then exponentiated, followed by summing to obtain softmax_sum.
* Finally, take the log of softmax_sum and add the result obtained from softmax_max.

* @attention Constraints:
* @li Ensure CANN and PyTorch package version compatibility when using this interface with PyTorch.
* @li Handle empty input: Check if 'query' is empty; return if so.
* If 'query' is non-empty and 'key', 'value' are empty tensors (i.e., S2=0), output a zero-filled tensor of the corresponding shape (fill 'attention_out').
* If 'attention_out' is an empty tensor, AscendCLNN framework will handle it.
* @li The shapes of tensors corresponding to 'key' and 'value' must be identical;
* in non-continuous scenarios, the batch size in the tensor list of 'key' and 'value' must be 1, equal to the number of 'query', with B, N, and D being equal.
*/
REG_OP(FusedInferAttentionScore)
    .INPUT(query, TensorType({DT_INT8, DT_FLOAT16, DT_BF16, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .DYNAMIC_INPUT(key, TensorType({DT_INT8, DT_FLOAT16, DT_BF16, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, DT_INT4}))
    .DYNAMIC_INPUT(value, TensorType({DT_INT8, DT_FLOAT16, DT_BF16, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, DT_INT4}))
    .OPTIONAL_INPUT(pse_shift, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_FLOAT16, DT_BOOL, DT_UINT8, DT_INT8}))
    .OPTIONAL_INPUT(actual_seq_lengths, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_lengths_kv, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(dequant_scale1, TensorType({DT_UINT64, DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale1, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(dequant_scale2, TensorType({DT_UINT64, DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale2, TensorType({DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(quant_offset2, TensorType({DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(block_table, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(query_padding_size, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(kv_padding_size, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(key_antiquant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(key_antiquant_offset, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(value_antiquant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(value_antiquant_offset, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(key_shared_prefix, TensorType({DT_INT8, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(value_shared_prefix, TensorType({DT_INT8, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(actual_shared_prefix_len, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(query_rope, TensorType({DT_INT8, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(key_rope, TensorType({DT_INT8, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(key_rope_antiquant_scale, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(dequant_scale_query, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(learnable_sink, TensorType({DT_BF16}))
    .OPTIONAL_INPUT(q_start_idx, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(kv_start_idx, TensorType({DT_INT64}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_INT8, DT_BF16}))
    .OUTPUT(softmax_lse, TensorType({DT_FLOAT32}))
    .REQUIRED_ATTR(num_heads, Int)
    .ATTR(scale, Float, 1.0)
    .ATTR(pre_tokens, Int, 2147483647)
    .ATTR(next_tokens, Int, 2147483647)
    .ATTR(input_layout, String, "BSH")
    .ATTR(num_key_value_heads, Int, 0)
    .ATTR(sparse_mode, Int, 0)
    .ATTR(inner_precise, Int, 1)
    .ATTR(block_size, Int, 0)
    .ATTR(antiquant_mode, Int, 0)
    .ATTR(softmax_lse_flag, Bool, false)
    .ATTR(key_antiquant_mode, Int, 0)
    .ATTR(value_antiquant_mode, Int, 0)
    .ATTR(query_quant_mode, Int, 0)
    .ATTR(pse_type, Int, 0)
    .ATTR(out_dtype, Int, 0)
    .OP_END_FACTORY_REG(FusedInferAttentionScore)


/**
* @brief Backwards calculation of FlashAttentionScore.

* @par Inputs:
* Seventeen inputs, including:
* @li query: A matrix Tensor. The type support float16, bf16, float32.
* @li key: A matrix Tensor. The type support float16, bf16, float32.
* @li value: A matrix Tensor. The type support float16, bf16, float32.
* @li dy: A matrix Tensor. The type support float16, bf16, float32.
* @li pse_shift: A scalar. An optional input parameter. The type support float16, bf16, float32.
* @li drop_mask: A matrix Tensor. An optional input parameter. The type support uint8.
* @li padding_mask: A matrix Tensor. An optional input parameter. The type support float16, bf16, float32.
* @li atten_mask: A matrix Tensor. An optional input parameter. The type support uint8, bool.
* @li softmax_max: A matrix Tensor. An optional input parameter. The type support float32.
* @li softmax_sum: A matrix Tensor. An optional input parameter. The type support float32.
* @li softmax_in: A matrix Tensor. An optional input parameter. The type support float16, bf16, float32.
* @li attention_in: A matrix Tensor. An optional input parameter. The type support float16, bf16, float32.
* @li prefix: A matrix Tensor. An optional input parameter. The type support int64.
* @li actual_seq_qlen: A matrix Tensor. An optional input parameter. The type support int64.
* If used, layout need to be setted TND. ex. If the q seqlen is [2,2,2,2,2], this parameter need be setted [2,4,6,8,10].
* @li actual_seq_kvlen: A matrix Tensor. An optional input parameter. The type support int64. If used,
* layout need to be setted TND. ex. If the kv seqlen is [2,2,2,2,2], this parameter need be setted [2,4,6,8,10].
* @li q_start_idx: A matrix Tensor. An optional input parameter. The type support int64.
* @li kv_start_idx: A matrix Tensor. An optional input parameter. The type support int64.
* @li d_scale_q: A matrix Tensor. An optional input parameter. The type support float32.
* @li d_scale_k: A matrix Tensor. An optional input parameter. The type support float32.
* @li d_scale_v: A matrix Tensor. An optional input parameter. The type support float32.
* @li d_scale_dy: A matrix Tensor. An optional input parameter. The type support float32.
* @li d_scale_o: A matrix Tensor. An optional input parameter. The type support float32.
* @li query_rope: A matrix Tensor. An optional input parameter. The type support float16, bf16, float32.
* @li key_rope: A matrix Tensor. An optional input parameter. The type support float16, bf16, float32.
* @li sink: A matrix Tensor. An optional input parameter. The type support float32.

* @par Attributes:
* @li scale_value: A float. The scale value. Default: 1.0.
* @li keep_prob: A float. The keep probability of dropout. Default: 1.0.
* @li pre_tockens: An int. Previous tokens.
* @li next_tockens: An int. Next tokens.
* @li head_num: An int. A required attribute. The number of the heads.
* @li input_layout: A string. A required attribute. Specifies the layout of `query`, the value must be one of ["BSH",
* "SBH", "BNSD", "BSND", "TND"].
* @li inner_precise: An int. 0, 1, reserved value. 2, support invalid lines.
* @li sparse_mode: An int. 0, defaultMsk. 1, allMask. 2, leftUpCausal. 3, rightDownCausal. 4, band. 5, prefix.
* 6, prefixCompress. 7, rightDownCausalBand. 8, bandLeftUpCausal.
* @li pse_type: An int. Optional attribute. Users can pass in 1 if they do not specify it.
* The supported configuration values ​​are 0, 1, 2, and 3.
* @li seed: An int. Optional attribute. Default: 0.
* @li offset: An int. Optional attribute.  Default: 0.
* @li out_dtype: An int. Optional attribute.  Default: 0.
* @li softmax_in_layout: A string. Optional attribute.  Default: "", the value must be one of ["", "same_as_input"].

* @par Outputs:
* @li dq: A matrix Tensor. The type support float16, bf16, float32.
* @li dk: A matrix Tensor. The type support float16, bf16, float32.
* @li dv: A matrix Tensor. The type support float16, bf16, float32.
* @li dpse: A matrix Tensor. The type support float16, bf16, float32.
* @li dqRope: A matrix Tensor. The type support float16, bf16, float32.
* @li dkRope: A matrix Tensor. The type support float16, bf16, float32.
* @li dsink: A matrix Tensor. The type support float32.
*/
REG_OP(FlashAttentionScoreGrad)
    .INPUT(query, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .INPUT(key, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .INPUT(value, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .INPUT(dy, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(pse_shift, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(drop_mask, TensorType({DT_UINT8}))
    .OPTIONAL_INPUT(padding_mask, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_BOOL, DT_UINT8}))
    .OPTIONAL_INPUT(softmax_max, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(softmax_sum, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(softmax_in, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(attention_in, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(prefix, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_qlen, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_kvlen, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(q_start_idx, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(kv_start_idx, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(d_scale_q, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(d_scale_k, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(d_scale_v, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(d_scale_dy, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(d_scale_o, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(query_rope, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(key_rope, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(sink, TensorType({DT_FLOAT32}))
    .OUTPUT(dq, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(dk, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(dv, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(dpse, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(dq_rope, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(dk_rope, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(dsink, TensorType({DT_FLOAT32}))
    .ATTR(scale_value, Float, 1.0)
    .ATTR(keep_prob, Float, 1.0)
    .ATTR(pre_tockens, Int, 2147483647)
    .ATTR(next_tockens, Int, 2147483647)
    .REQUIRED_ATTR(head_num, Int)
    .REQUIRED_ATTR(input_layout, String)
    .ATTR(inner_precise, Int, 0)
    .ATTR(sparse_mode, Int, 0)
    .ATTR(pse_type, Int, 1)
    .ATTR(seed, Int, 0)
    .ATTR(offset, Int, 0)
    .ATTR(out_dtype, Int, 0)
    .ATTR(softmax_in_layout, String, "")
    .OP_END_FACTORY_REG(FlashAttentionScoreGrad)


/**
* @brief Fusion op for FFN. This op supports to compute MoeFFN(Mixture-of-Experts) or FFN.
* @par Inputs:
* fourteen inputs, including:
* @li x: A matrix Tensor. The type support int8, float16, bfloat16.
* Format support ND, FRACTAL_NZ. Shape supports at least 2 dimensions (M,K1), and at most 8 dimensions.
* @li weight1: A matrix Tensor for weight of the first matmul. The type support int4, int8, float16, bfloat16.
* Format support ND, FRACTAL_NZ. When having experts/having no expert, shape should be (E, K1, N1)/(K1, N1).
* @li weight2: A matrix Tensor for weight of the second matmul. The type support int4, int8, float16, bfloat16.
* Format support ND, FRACTAL_NZ. When having experts/having no expert, shape should be (E, K2, N2)/(K2, N2).
* @li expert_tokens: A matrix Tensor. Indicating num of tokens in each of experts. If having experts, expert_tokens should be passed; if having no experts, expert_tokens should not be passed.
* The type support int64. Format support ND. If not null, shape should be (E) and should satisfy E <= 256.
* @li bias1: A matrix Tensor for bias of the first matmul. The type support int32, float16, float32. Format support ND.
* When having experts/having no expert, shape should be (E, N1)/(N1).
* @li bias2: A matrix Tensor for bias of the secend matmul. The type support int32, float16, float32. Format support ND.
* When having experts/having no expert, shape should be (E, N2)/(N2).
* @li scale: A matrix Tensor. Indicating scaling factor of quantization parameter.
* The type support float32. Format support ND. In per-tensor quantization cases, when having experts/having no expert, shape should be (E)/(1).
* In per-channel quantization cases, when having experts/having no expert, shape should be (E, N1)/(N1).
* @li offset: A matrix Tensor. Indicating the offset of the quantization parameter.
* The type support float32. Format support ND. When having experts/having no expert, shape should be (E)/(1).
* @li deq_scale1: A matrix Tensor. Indicating scaling factor of dequantization parameter for the first matmul.
* The type support uint64, int64, float32, bfloat16. Format support ND. When having experts/having no expert, shape should be (E, N1)/(N1).
* @li deq_scale2: A matrix Tensor. Indicating scaling factor of dequantization parameter for the second matmul.
* The type support uint64, int64, float32, bfloat16. Format support ND. When having experts/having no expert, shape should be (E, N2)/(N2).
* @li antiquant_scale1: A matrix Tensor. Indicating the scaling factor of the fake-quantization parameter for the first matmul.
* The type support float16, bfloat16. Format support ND. In per-channel fake-quantization cases, when having experts/having no expert, shape should be (E, N1)/(N1).
* In per-in-group fake-quantization cases, when having experts/having no expert, shape should be (E, G1, N1)/(G1, N1).
* @li antiquant_scale2: A matrix Tensor. Indicating the scaling factor of the fake-quantization parameter for the second matmul.
* The type support float16, bfloat16. Format support ND. In per-channel fake-quantization cases, when having experts/having no expert, shape should be (E, N1)/(N1).
* In per-in-group fake-quantization cases, when having experts/having no expert, shape should be (E, G2, N2)/(G2, N2).
* @li antiquant_offset1: A matrix Tensor. Indicating the offset of the fake-quantization parameter for the first matmul.
* The type support float16, bfloat16. Format support ND. In per-channel fake-quantization cases, when having experts/having no expert, shape should be (E, N2)/(N2).
* In per-in-group fake-quantization cases, when having experts/having no expert, shape should be (E, G1, N1)/(G1, N1).
* @li antiquant_offset2: A matrix Tensor. Indicating the offset of the fake-quantization parameter for the second matmul.
* The type support float16, bfloat16. Format support ND. In per-channel fake-quantization cases, when having experts/having no expert, shape should be (E, N2)/(N2).
* In per-in-group fake-quantization cases, when having experts/having no expert, shape should be (E, G2, N2)/(G2, N2).

* @par Attributes:
* @li activation: A string. The type of activation. Support fastgelu, gelu, relu, silu, geglu, swiglu and reglu.
* @li inner_precise: An int. 0, fp16 high precision. 1, high performance. Default value: 0
* @li output_dtype: An int. -1, output data type is float16. 0, output data type is float16. 1, output data type is bfloat16. Default -1.
* @li tokens_index_flag: A bool. false, values in expert_tokens are values. true, values in expert_tokens are indices. Default value: false
*
* @par Outputs:
* y: A matrix Tensor. The type support float16, bfloat16.
* Format support ND, FRACTAL_NZ. Num of dimension should be same as x.
*\n
*\n
* The following are the supported data formats and data types (for Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component):
*\n
| Tensor    | x       | weight1/weight2 | bias1/bias2 | scale/offset | deq_scale1/deq_scale2 | antiquant_scale1/antiquant_scale2  | antiquant_offset1/antiquant_offset2 | y       |
| :-------: | :-----: | :-------------: | :---------: | :----------: | :-------------------: | :--------------------------------: | :---------------------------------: | :-----: |
| Format1   | ND      | ND              | ND          | ND           | ND                    | ND                                 | ND                                  | ND      |
| Data Type | float16 | float16         | float16     | -            | -                     | -                                  | -                                   | float16 |
|           | bfloat16| bfloat16        | float32     | -            | -                     | -                                  | -                                   | bfloat16|
|           | int8    | int8            | int32       | float32      | uint64                | -                                  | -                                   | float16 |
|           | int8    | int8            | int32       | float32      | bfloat16              | -                                  | -                                   | bfloat16|
|           | int8    | int8            | int32       | float32      | int64                 | -                                  | -                                   | float16 |
|           | int8    | int8            | int32       | float32      | float32               | -                                  | -                                   | float16 |
|           | float16 | int8            | float16     | -            | -                     | float16                            | float16                             | float16 |
|           | bfloat16| int8            | float32     | -            | -                     | bfloat16                           | bfloat16                            | bfloat16|
|           | float16 | int4            | float16     | -            | -                     | float16                            | float16                             | float16 |
|           | bfloat16| int4            | float32     | -            | -                     | bfloat16                           | bfloat16                            | bfloat16|
*\n
* The following are the supported data formats and data types (for Atlas Inference Series Product):
*\n
| Tensor    | x          | weight1/weight2 | bias1/bias2 | scale/offset | deq_scale1/deq_scale2 | antiquant_scale1/antiquant_scale2  | antiquant_offset1/antiquant_offset2 | y          |
| :-------: | :--------: | :-------------: | :---------: | :----------: | :-------------------: | :--------------------------------: | :---------------------------------: | :--------: |
| Format1   | FRACTAL_NZ | FRACTAL_NZ      | ND          | ND           | ND                    | ND                                 | ND                                  | FRACTAL_NZ |
| Data Type | float16    | float16         | float16     | -            | -                     | -                                  | -                                   | float16    |
*\n
* @attention Constraints:
* @li Atlas Inference Series Product only support non-quantization high performance no-expert cases; x and y must have two dimensions; activation only supports gelu/fastgelu/relu/silu.
* @li If expert_tokens is passed, when tokens_index_flag is true, it must be a non-negative monotone non-decreasing array; when tokens_index_flag is false, it must be a non-negative array.
* @li If expert_tokens is passed, when tokens_index_flag is false, sum of expert_tokens should be equal to the first dim M of x; when tokens_index_flag is true, the last value in expert_tokens should be equal to the first dim M of x.
* @li If activation is geglu/swiglu/reglu, only supporting float16 (data type of required inputs are all float16) high performance no-expert cases, and should satisfy N1=2*K2.
* @li If activation is gelu/fastgelu/relu/silu, should satisfy N1=K2.
* @li All cases should satisfy K1=N2, K1<65536, K2<65536, and M should be less than 2147483547 after aligning to 32 byte.
* @li Non-quantization cases should not pass quantization or fake-quantization related optional inputs, quantization cases should not pass fake-quantization related optional inputs, fake-quantization cases should not pass quantization related optional inputs.
* @li Per-tensor quantization cases support data type template with deq_scale1/deq_scale2 float32, while per-channel quantization cases do not support this data type template.
* @li If data type of weight1 and weight2 is int4, the last dimension of weight1 and weight2 must be even.
* @li In per-in-group fake-quantization cases, group num G1 of antiquant_scale1 and antiquant_offset1 must be divisible by K1, group num G2 of antiquant_scale2 and antiquant_offset2 must be divisible by K2.
* @li Attr inner_precise is only valid in non-quantization cases. In non-quantization cases, if data type of required inputs are all bfloat16, inner_precise only supports 0; if data type of required inputs are all float16, inner_precise can pass 0 or 1.
* @li Attr output_dtype is only valid in quantization cases.
*/
REG_OP(FFN)
    .INPUT(x, TensorType({DT_INT8, DT_FLOAT16, DT_BF16}))
    .INPUT(weight1, TensorType({DT_INT8, DT_FLOAT16, DT_BF16, DT_INT4}))
    .INPUT(weight2, TensorType({DT_INT8, DT_FLOAT16, DT_BF16, DT_INT4}))
    .OPTIONAL_INPUT(expert_tokens, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(bias1, TensorType({DT_INT32, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(bias2, TensorType({DT_INT32, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(deq_scale1, TensorType({DT_UINT64, DT_BF16, DT_INT64, DT_FLOAT}))
    .OPTIONAL_INPUT(deq_scale2, TensorType({DT_UINT64, DT_BF16, DT_INT64, DT_FLOAT}))
    .OPTIONAL_INPUT(antiquant_scale1, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_scale2, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_offset1, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_offset2, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(activation, String)
    .ATTR(inner_precise, Int, 0)
    .ATTR(output_dtype, Int, -1)
    .ATTR(tokens_index_flag, Bool, false)
    .OP_END_FACTORY_REG(FFN)


/**
* @brief Fusion op of allgather and matmul.
* @par Inputs:
* three inputs, including:
* @li x1: A matrix Tensor. The type support float16, bfloat16. The format supports ND. The x1 only supports 2 dimensions in current version, for example (M, K). The x1 doesn't support transposed.
* @li x2: A matrix Tensor. The type support float16, bfloat16. The format supports ND. The x2 only supports 2 dimensions in current version, for example (K, N). The x2 supports transposed and non-transposed.
  The K value in x2 should be same as the K value in x1 when x2 is non-transposed, and the K value should be in range [256, 65535).
* @li bias: A matrix Tensor. The type support float16, bfloat16. The format supports ND. The current version does not support the scenario where bias is not 0.\n
*
* @par Attributes:
* @li group: A string. A required string identifying the group of ranks participating in the op.
* @li is_trans_a: A bool. If true, changes the shape of "x1" from [K, M] to [M, K] before multiplication. Default: false.
* @li is_trans_b: A bool. If true, changes the shape of "x2" from [N, K] to [K, N] before multiplication. Default: false.
* @li gather_index: An int. Represents the input index for doing gather, 0: left matrix, 1: right matrix. Default: 0. The gather_index only supports 0 in current version.
* @li comm_turn: An int. Number of communications with AICPU. Default: 0. The comm_turn only supports 0 in current version.
* @li rank_size: An int. Number of ranks in the group. Default: 0. \n
  The Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component support 2, 4, 8. \n
  The Atlas A3 Training Series Product/Atlas A3 Inference Series Product support 2, 4, 8, 16. \n
* @li is_gather_out: A bool. If true, output gather_out matrix. Default: true. \n
*
* @par Outputs:
* @li y: A matrix Tensor. The type support float16, bfloat16. The format supports ND. The y is 2 dimensions, for example (M*rank_size, N).
* @li gather_out: A matrix Tensor. The type support float16, bfloat16. The format supports ND. \n
*/
REG_OP(AllGatherMatmul)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(gather_out, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(group, String)
    .ATTR(is_trans_a, Bool, false)
    .ATTR(is_trans_b, Bool, false)
    .ATTR(gather_index, Int, 0)
    .ATTR(comm_turn, Int, 0)
    .ATTR(rank_size, Int, 0)
    .ATTR(is_gather_out, Bool, true)
    .OP_END_FACTORY_REG(AllGatherMatmul)

/**
* @brief Fusion op of allgather and matmul v2.
* @par Inputs:
* six inputs, including:
* @li x1: A matrix tensor. The type support float16, bfloat16, float8_e4m3fn, float8_e5m2, hifloat8. The format support ND. The x1 only support 2 dimensions in current version, for example (M, K). The x1 doesn't support transposed.
* @li x2: A matrix tensor. The type support float16, bfloat16, float8_e4m3fn, float8_e5m2, hifloat8. The format support ND. The x2 only support 2 dimensions in current version, for example (K, N). The x2 support transposed and non-transposed.
  The K value in x2 should be same as the K value in x1 when x2 is non-transposed, and the K value should be in range [256, 65535).
* @li bias: A matrix tensor. If x1 type is float8_e4m3fn, float8_e5m2, hifloat8, then the bias type support float32.
  If x1 type is float16, bfloat16, then the bias type support float16, bfloat16. The format support ND. The current version not support the scenario where bias is not 0.
* @li x1_scale: A matrix tensor. The type support float32, float_e8m0. The format support ND. The x1_scale only support 1 dimension and only one element in current version, for expample (1,).
* @li x2_scale: A matrix tensor. The type support float32, float_e8m0. The format support ND. The x2_scale only support 1 dimension and only one element in current version, for expample (1,).
* @li quant_scale: A matrix tensor. The type support float32. The format support ND. The quant_scale only support 1 dimension and only one element, for example (1,). The quant_scale only support nullptr in current version. \n
*
* @par Attributes:
* @li group: A required String identifying the group of ranks
  participating in the op.
* @li is_trans_a: A bool. If True, changes the shape of "x1" from [K, M] to
  [M, K] before multiplication. Default: "false".
* @li is_trans_b: A bool. If True, changes the shape of "x2" from [N, K] to
  [K, N] before multiplication. Default: "false".
* @li gather_index: An int. Represents the input index for doing gather.
  Default: "0". The gather_index only support 0 in current version.
* @li comm_turn: An int. Number of communications with CCU. Default: "0". The comm_turn only support 0 in current version.
* @li rank_size: An int. Number of rank num. Default: "0".
* @li block_size: An int. Number of values in the M, N directions in y correspond to one dequantization coefficient. Default: "0". The block_size only supports 0 in current version.
* @li group_size: An int. Number of values in the M, N, K directions in x1/x2 correspond to one dequantization coefficient. \n
  Default: "0". Reserved attribute. The block_size only supports 549764202624 in the per_block scenario. The block_size only supports 0 in the other scenario. \n
* @li is_gather_out: A bool. If True, output gather_out matrix. Default: "true".
* @li is_amax_out: A bool. If True, output amax matrix. Default: "false".
* @li y_dtype: An int. Type of the matmul output. Default: "0". \n
*
* @par Outputs:
* @li y: A matrix tensor. The type support float16, bfloat16, float32. The format support ND. The y is 2 dimension, for example (M*rank_size, N).
* @li gather_out: A matrix tensor. All data returned from communication. The type of gather_out is consistent with that of x1. The type support float16, bfloat16, float8_e4m3fn, float8_e5m2, hifloat8. The format support ND.
* @li amax_out: A matrix tensor. Maximum value of output matrix. The type support float32. The format support ND. The amax_out is 1 dimension and only one element, for example (1,). The amax_out only support nullptr in current version. \n
* @li comm_mode: A string. Communication mode. Default: "ccu". The comm_mode only supports "ccu", "aiv" or "aicpu" in current version. \n
*/
REG_OP(AllGatherMatmulV2)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_HIFLOAT8}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_HIFLOAT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(x1_scale, TensorType({DT_FLOAT, DT_FLOAT_E8M0}))
    .OPTIONAL_INPUT(x2_scale, TensorType({DT_FLOAT, DT_FLOAT_E8M0}))
    .OPTIONAL_INPUT(quant_scale, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(gather_out, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_HIFLOAT8}))
    .OUTPUT(amax_out, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(group, String)
    .ATTR(is_trans_a, Bool, false)
    .ATTR(is_trans_b, Bool, false)
    .ATTR(gather_index, Int, 0)
    .ATTR(comm_turn, Int, 0)
    .ATTR(rank_size, Int, 0)
    .ATTR(block_size, Int, 0)
    .ATTR(group_size, Int, 0)
    .ATTR(is_gather_out, Bool, true)
    .ATTR(is_amax_out, Bool, false)
    .ATTR(y_dtype, Int, 0)
    .ATTR(comm_mode, String, "ccu")
    .OP_END_FACTORY_REG(AllGatherMatmulV2)

/**
* @brief Fusion op of alltoall, allgather, and batch matmul.
* @par Inputs:
* Three inputs, including (Tp is short for tp_world_size, and ep is short for ep_world_size.):
* @li x: A matrix Tensor. The type support float16, bfloat16. The format supports ND. The x only supports 3 dimensions.
* x_shard_type 0: x (E, C, H/tp), x_shard_type 1: x (E, C/tp, H).
* @li weight: A matrix Tensor. The type support float16, bfloat16 and its type must be the same as the tpye of x. The format supports ND. The weight only supports 3 dimensions.
* weight (E/ep, H, M/tp).
* @li bias: A matrix Tensor. The type support float16, float32. When x is float16, bias must be float16. When x is bfloat16, bias must be float32. 2D and 3D are supported. The data format can be ND.
* bias (E/ep, 1, M/tp), 3 dims; bias (E/ep, M/tp), 2 dims. \n
*
* @par Attributes:
* @li group_ep: A required String identifying the expert group of ranks
  participating in the op. The string length must be greater than 0 and less than 128.
* @li group_tp: A required String identifying the tensor group of ranks
  participating in the op. The string length must be greater than 0 and less than 128.
* @li ep_world_size: A required int identifying the number of expert parallel group rank num. The value can be 2, 4, 8, 16 or 32.
* @li tp_world_size: A required int identifying the number of tensor parallel group rank num. The value can be 2, 4, 8, 16 or 32.
* @li x_shard_type: An int. Represents the input x shards on dim 2 or 3 in tensor parallel group. Default: "0". The value 0 indicates that AllGather is performed in the H dimension (2nd dimension of x, which has three dimensions: 0th, 1st, and 2nd) based on the tensor parallel group.
  The value 1 indicates that AllGather is performed in the C dimension (1st dimension of x) based on the tensor parallel group.
* @li act_type: An int. Represents the activation function type. Default: "0". 0: None, 1: GELU, 2: Silu, 3: Relu, 4: FastGELU. None indicates no activation function.
* @li transpose_weight: A bool. If True, changes the shape of "weight" from [E, N, K] to
* [E, K, N] before multiplication. Default: "false".
* @li output_y2_flag: A bool. If True, y2 tensor will output a allgather matrix Tensor
* as a result. Default: "false".
* @li output_y3_flag: A bool. If True, y3 tensor will output a batch matmul matrix Tensor
* as a result. Default: "false". \n
*
* @par Outputs:
* @li y1: A batch matmul or activation matrix Tensor. The type support float16, bfloat16. 3D is supported. The data type is the same as that of the input x. The data format can be ND. If there is an activation function, the result is the output of the activation function. Otherwise, the result is the output of BatchMatMul.
* x_shard_type 0: y1 (E/ep, ep*C, M/tp), x_shard_type 1: y1 (E/ep, ep*tp*C/tp, M/tp).
* @li y2: A allgather matrix Tensor. The type support float16, bfloat16. 3D is supported. The data type is the same as that of the input x. The data format can be ND. This parameter indicates AllGather output, which may be required for backpropagation. A null pointer indicates that the output is not required.
* x_shard_type 0: y2 (E/ep, ep*C, H), x_shard_type 1: y2 (E/ep, ep*tp*C/tp, H).
* @li y3: A batch matmul matrix Tensor. The type support float16, bfloat16. 3D is supported. The data type is the same as that of the input x. The data format can be ND. This parameter indicates BatchMatMul output when there is an activation function. A null pointer indicates that the output is not required.
* x_shard_type 0: y3 (E/ep, ep*C, M/tp), x_shard_type 1: y3 (E/ep, ep*tp*C/tp, M/tp). \n
*
* @attention Constraints:
* @li The x[0] means E value, w[0] means E/ep, so the w[0] multi ep should equal the x[0]. Other divisible relationships are similar to this one.
* @li The value range of E is [2, 512], and E is an integer multiple of ep.
* @li The value range of H is [1, 65535], and H is an integer multiple of tp when x_shard_type is 0.
* @li The value range of M/tp is [1, 65535].
* @li The value range of E/ep is [1, 32].
* @li C must be greater than 0 and cannot exceed the upper limit of the operator's device memory. C is an integer multiple of tp when x_shard_type is 1.
* @li Group_ep and group_tp can't be consistent. Ep and tp can only be 2, 4, 8, 16 or 32. Supernodes cannot be crossed.
*/
REG_OP(AlltoAllAllGatherBatchMatMul)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(weight, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y1, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y2, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y3, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(group_ep, String)
    .REQUIRED_ATTR(group_tp, String)
    .REQUIRED_ATTR(ep_world_size, Int)
    .REQUIRED_ATTR(tp_world_size, Int)
    .ATTR(x_shard_type, Int, 0)
    .ATTR(act_type, Int, 0)
    .ATTR(transpose_weight, Bool, false)
    .ATTR(output_y2_flag, Bool, false)
    .ATTR(output_y3_flag, Bool, false)
    .OP_END_FACTORY_REG(AlltoAllAllGatherBatchMatMul)

/**
* @brief Fusion op of batch matmul, reduce scatter, and alltoall.
* @par Inputs:
* Three inputs, including (Tp is short for tp_world_size, and ep is short for ep_world_size.):
* @li x: A matrix Tensor. The type support float16, bfloat16. The x only supports 3 dimensions. The data format can be ND.
* x (E/ep, ep*C, M/tp).
* @li weight: A matrix Tensor. The type support float16, bfloat16 and its type must be the same as the tpye of x. The weight only supports 3 dimensions. The data format can be ND.
* weigh (E/ep, M/tp, H).
* @li bias: A matrix Tensor. The type support float16, float32. If x is float16, bias must be float16. If x is bfloat16, bias must be float32. 2D and 3D are supported. The data format can be ND. \n
* y_shard_type 0: bias (E/ep, 1, H/tp), y_shard_type 1: bias (E/ep, 1, H), 3 dims; \n
* y_shard_type 0: bias (E/ep, H/tp), y_shard_type 1: bias (E/ep, H), 2 dims.
* @par Attributes:
* @li group_ep: A required String identifying the expert group of ranks
  participating in the op. The string length must be greater than 0 and less than 128.
* @li group_tp: A required String identifying the tensor group of ranks
  participating in the op. The string length must be greater than 0 and less than 128.
* @li ep_world_size: A required int identifying the number of expert parallel group rank num. The value can be 2, 4, 8, 16 or 32.
* @li tp_world_size: A required int identifying the number of tensor parallel group rank num. The value can be 2, 4, 8, 16 or 32.
* @li y_shard_type: An int. Represents the output y shards on dim 2 or 3 in the tensor parallel group.
* Default: "0". The value 0 indicates that ReduceScatter is performed by tensor parallel group in the H dimension (2nd dimension of the BatchMatMul computation result, which has three dimensions: 0th, 1st, and 2nd).
* The value 1 indicates that ReduceScatter is performed by tensor parallel group in the C dimension (1st dimension of the BatchMatMul computation result).
* @li transpose_weight: A bool. If True, changes the shape of "weight" from [E, N, K] to
* [E, K, N] before multiplication. Default: "false". \n
*
* @par Outputs:
* y: A matrix Tensor. The type support float16, bfloat16. 3D is supported. The type is the same as that of input x. The data format can be ND.
* y_shard_type 0: y (E, C, H/tp), y_shard_type 1: y (E, C/tp, H).
*
* @attention Constraints:
* @li The x[0] means E/tp value, y[0] means E, so the x[0] multi tp should equal the y[0].  Other divisible relationships are similar to this one.
* @li The value range of E is [2, 512], and E is an integer multiple of ep.
* @li he value range of H is [1, 65535], and H is an integer multiple of tp when y_shard_type is 0.
* @li The value range of M/tp is [1, 65535].
* @li The value range of E/ep is [1, 32].
* @li C is greater than 0 and cannot exceed the upper limit of the operator's device memory. C is an integer multiple of tp when y_shard_type is 1.
* @li Group_ep and group_tp can't be consistent. Ep and tp can only be 2, 4, 8, 16 or 32. Supernodes cannot be crossed. \n
*/
REG_OP(BatchMatMulReduceScatterAlltoAll)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(weight, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(group_ep, String)
    .REQUIRED_ATTR(group_tp, String)
    .REQUIRED_ATTR(ep_world_size, Int)
    .REQUIRED_ATTR(tp_world_size, Int)
    .ATTR(y_shard_type, Int, 0)
    .ATTR(transpose_weight, Bool, false)
    .OP_END_FACTORY_REG(BatchMatMulReduceScatterAlltoAll)

/**
* @brief Combine similar tokens using the matching algorithm.
* @par Inputs:
* @li token_a: A Tensor. Type is:DT_FLOAT16. Shape is (B, S1, H).
* @li token_b: A Tensor. Type is:DT_FLOAT16. Shape is (B, S2, H).
* @li topk_indice: A Tensor. Type is:DT_INT64. Shape is (B, S1, H), S1 must equal with token_a. Value range is [0, S1), no dup.
* @li arg_max: A Tensor. Type is:DT_INT64. Shape is (B, S1, H), S1 must equal with token_a. Value range is [0, S2), can dup.
* @par Outputs:
* @li unmerge_token_a: A Tensor. Type is:DT_FLOAT16.
* @li unmerge_token_b: A Tensor. Type is:DT_FLOAT16.
* @li unreduce_count: A Tensor. Type is:DT_FLOAT.
* @par Attributes:
* @li top_rate: Type is:Float. rate to calculate how many rows of token_a merge to token_b. default is "0.5".
* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(TomeMerge)
    .INPUT(token_a, TensorType({DT_FLOAT16}))
    .INPUT(token_b, TensorType({DT_FLOAT16}))
    .INPUT(topk_indice, TensorType({DT_INT64}))
    .INPUT(arg_max, TensorType({DT_INT64}))
    .OUTPUT(unmerge_token_a, TensorType({DT_FLOAT16}))
    .OUTPUT(unreduce_token_b, TensorType({DT_FLOAT16}))
    .OUTPUT(unreduce_count, TensorType({DT_FLOAT}))
    .ATTR(top_rate, Float, 0.5)
    .OP_END_FACTORY_REG(TomeMerge)

/**
* @brief Fusion op of matmul and reduce scatter.
* @par Inputs:
* three inputs, including:
* @li x1: A matrix Tensor. The type support float16, bfloat16. The format supports ND. The x1 only supports 2 dimensions in current version, for example (M, K). The x1 doesn't support transposed.
* @li x2: A matrix Tensor. The type support float16, bfloat16. The x2 only supports 2 dimensions in current version, for example (K, N). The x2 supports transposed and non-transposed.
  The K value in x2 should be same as the K value in x1 when x2 is non-transposed, and the K value should be in range [256, 65535).
* @li bias: A matrix Tensor. The type support float16, bfloat16. The format supports ND. The current version does not support the scenario where bias is not 0.\n
*
* @par Attributes:
* @li group: A string. A required string identifying the group of ranks participating in the op.
* @li reduce_op: A string. A required string identifying the reduction operation to perform. Default: "sum".
* @li is_trans_a: A bool. If True, changes the shape of "x1" from [K, M] to [M, K] before multiplication. Default: false.
* @li is_trans_b: A bool. If True, changes the shape of "x2" from [N, K] to [K, N] before multiplication. Default: false.
* @li comm_turn: An int. Number of communications with AICPU. Default: 0. The comm_turn only supports 0 in current version.
* @li rank_size: An int. Number of ranks in the group. Default: 0. \n
  The Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component support 2, 4, 8. \n
  The Atlas A3 Training Series Product/Atlas A3 Inference Series Product support 2, 4, 8, 16. \n
*
* @par Outputs:
* y: A matrix Tensor. The type support float16, bfloat16. The format supports ND. The y is 2 dimensions, for example (M/rank_size, N).\n
*/
REG_OP(MatmulReduceScatter)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(group, String)
    .ATTR(reduce_op, String, "sum")
    .ATTR(is_trans_a, Bool, false)
    .ATTR(is_trans_b, Bool, false)
    .ATTR(comm_turn, Int, 0)
    .ATTR(rank_size, Int, 0)
    .OP_END_FACTORY_REG(MatmulReduceScatter)

/**
* @brief Fusion op of matmul and reduce scatter v2.
* @par Inputs:
* six inputs, including:
* @li x1: A matrix tensor. The type supports float16, bfloat16, hifloat8, float8_e5m2, float8_e4m3fn.The format supports ND. The x1 only supports two dimensions in current version, for example (M, K). The x1 doesn't support transposed.
* @li x2: A matrix tensor. The type supports float16, bfloat16, hifloat8, float8_e5m2, float8_e4m3fn.The x2 only supports two dimensions in current version, for example (K, N). The x2 supports transposed and non-transposed.
  The K value in x2 should be same as the K value in x1 when x2 is non-transposed, and the K value should be in range [256, 65535).
* @li bias: A matrix tensor. The format supports ND. If x1 type is float8_e4m3fn, float8_e5m2, hifloat8, then the bias type supports float32. \n
  If x1 type is float16, bfloat16, then the bias type supports float16, bfloat16 and the current version doesn't supports the scenario where bias is not 0. \n
  The bias only supports one dimension in current version, for expample (N,). \n
  The bias only supports nullptr in the per_block scenario. \n
* @li x1_scale: A matrix tensor. The type supports float32, float_e8m0. The format supports ND. \n The x1_scale only supports nullptr when x1's type is float16 or bfloat16. \n
  The x1_scale only supports one dimension and only one element in the per_tensor scenario, for expample (1,). \n
  The x1_scale only supports two dimensions in the per_block scenario, for expample (ceildiv(M, 128), ceildiv(K, 128)). \n
* @li x2_scale: A matrix tensor. The type supports float32, float_e8m0. The format supports ND. \n The x2_scale only supports nullptr when x1's type is float16 or bfloat16. \n
  The x2_scale only supports one dimension and only one element in the per_tensor scenario, for expample (1,). \n
  The x2_scale only supports two dimensions in the per_block scenario, for expample (ceildiv(K, 128), ceildiv(N, 128)). \n
* @li quant_scale: A matrix tensor. The type supports float32. The format supports ND. The quant_scale only supports one dimension and only one element, for example (1,). The quant_scale only supports nullptr in current version. \n
*
* @par Attributes:
* @li group: A required String identifying the group of ranks
  participating in the op.
* @li reduce_op: A required string identifying the reduction operation to perform. Default: "sum". The reduce_op only supports sum in current version.
* @li is_trans_a: A bool. If True, changes the shape of "x1" from [K, M] to [M, K] before multiplication.
* If False, don't change the shape of "x1" before multiplication. Default: "false". The is_trans_a only supports false in current version.
* @li is_trans_b: A bool. If True, changes the shape of "x2" from [N, K] to [K, N] before multiplication.
* If False, don't change the shape of "x2" before multiplication. Default: "false".
* @li comm_turn: An int. Number of communications with CCU. Default: "0". The comm_turn only supports 0 in current version.
* @li rank_size: An int. Number of rank num. Default: "0".The rank_size only supports 2/4/8/16/32/64 in current version.
* @li block_size: An int. Number of values in the M, N directions in y correspond to one dequantization coefficient. Default: "0". The block_size only supports 0 in current version.
* @li group_size: An int. Number of values in the M, N, K directions in x1/x2 correspond to one dequantization coefficient. \n
  Default: "0". Reserved attribute. The block_size only supports 549764202624 in the per_block scenario. The block_size only supports 0 in the other scenario. \n
* @li is_amax_out: A bool. whether the output of amax_out is supported . Default: "false". If True, supports output amax_out.
* @li y_dtype: An int. Default: "0". The y_dtype only supports 0(float32)/1(float16)/27(bfloat16) in current version. \n
* @li comm_mode: A string. communication Mode. Default: "". The comm_mode only supports "aicpu" or "aiv" in current version. \n
*
* @par Outputs:
* @li y: A matrix tensor. If x1 type is float16, bfloat16, The type is same as x1.
  If x1 type is float8_e4m3fn, float8_e5m2, hifloat8, the output type supports float16, bfloat16, float32. The format supports ND. The y is two dimensions, for example (M/rank_size, N).
* @li amax_out: A matrix tensor. The type supports float32. The format supports ND. The amax_out is 1 dimension and only one element, for example (1,). The amax_out only supports nullptr in current version. \n
*/
REG_OP(MatmulReduceScatterV2)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_HIFLOAT8}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_HIFLOAT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(x1_scale, TensorType({DT_FLOAT, DT_FLOAT_E8M0}))
    .OPTIONAL_INPUT(x2_scale, TensorType({DT_FLOAT, DT_FLOAT_E8M0}))
    .OPTIONAL_INPUT(quant_scale, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(amax_out, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(group, String)
    .ATTR(reduce_op, String, "sum")
    .ATTR(is_trans_a, Bool, false)
    .ATTR(is_trans_b, Bool, false)
    .ATTR(comm_turn, Int, 0)
    .ATTR(rank_size, Int, 0)
    .ATTR(block_size, Int, 0)
    .ATTR(group_size, Int, 0)
    .ATTR(is_amax_out, Bool, false)
    .ATTR(y_dtype, Int, 0)
    .ATTR(comm_mode, String, "aicpu")
    .OP_END_FACTORY_REG(MatmulReduceScatterV2)

/**
* @brief Function MatmulAllReduce.
* @par Inputs:
* @li x1: A matrix tensor, left matrix of mm. The type support float16, bf16, int8, hifloat8,
      float8_e5m2, float8_e4m3, float4_e2m1, float4_e1m2.
* @li x2: A matrix tensor, right matrix of mm. The type support float16, bf16, int8, int4, hifloat8, float8_e5m2,
      float8_e4m3, float4_e2m1, float4_e1m2.
* @li bias: A matrix tensor, input biasOptional in the formula. The type support float16, bf16, int32, float32.
* @li x3: A matrix tensor, input x3Optional in the formula. The type support float16, bf16, float32.
* @li antiquant_scale: A matrix tensor. The type support float16, bf16, float32.
* @li antiquant_offset: A matrix tensor. The type support float16, bf16.
* @li dequant_scale: A matrix tensor. This parameter indicates the dequantization coefficient after the mm operation.
      The shape is (1) in the per-tensor scenario and (n)/(1, n) in the per-channel scenario.
      The type support float16, bf16, uint64, int64, float32, float8_e8m0.
* @li pertoken_scale: A matrix tensor, the per-token dequantization coefficient after the MM computation.
      The type support float32, float8_e8m0, float16, bf16.
* @li comm_quant_scale_1: A matrix tensor. The per-channel quantization coefficient after the matmulAdd computation.
      The type support float16, bf16, float32.
* @li comm_quant_scale_2: A matrix tensor. The per-channel quantization coefficient after the allGather computation.
      The type support float16, bf16, float32. \n

* @par Attributes:
* @li group: A required String identifying the group of ranks
*  participating in the op.
* @li reduce_op: A required string identifying the reduction operation to
*  perform. support "sum", "min", "max", "prod", currently only support "sum".
* @li is_trans_a: A bool. If True, changes the shape of "x1" from [K, M] to
*  [M, K] before multiplication. Default: false.
* @li is_trans_b: A bool. If True, changes the shape of "x2" from [N, K] to
*  [K, N] before multiplication. Default: false.
* @li comm_turn: An int. Number of communications with AICPU. Default: 0.
* @li antiquant_group_size: An int. Number of per-group for quant. Default: 0.
* @li group_size: An int. group_size = group size K | group size N << 16 | group size M << 32. Default: 0.
* @li y_dtype: An int. The y_dtype only support 0(float32)/1(float16)/27(bfloat16) in current version.
      Default: 28(ge::DataType::DT_UNDEFINED).
* @li comm_quant_mode: An int. Number of low-bits communicate mode. Static quant: 0. Dynamic quant: 1. Default: 0. \n

* @par Outputs:
* y: A matrix tensor. The type support float16, bf16. \n

* @attention Constraints:
* - Constraints for MatmulAllreudce:
* @li MatmulAllReduce has poor performance when the product of the 1th dimension(b) and 2st dimension(s) of input x1 is small.
* @li x1 can be 2-dimensional or 3-dimensional, and the dimension is (b, s, k) or (m, k). x2 must be
*  2-dimensional and its dimension is (k, n). The axis meets the input parameter requirements of the mm operator,
*  and their k axes are equal. If bias is not empty, it is 1-dimensional.
* @li Dimensions except the last one of output are the same as those of x1. The last dimension is the same as
*  that of x2. If bias is not empty, the shape size is the same as the last dimension of output.
* @li The input data type of x1, x2 and bias (if supported) computation must be the same as the output data
*  type of output computation.
* @li The x2 matrix can be transposed or not transposed. The x1 matrix cannot be transposed.
* @li The Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component support 1, 2, 4, and 8 cards.
*
* - Constraints for WeightQuantMatmulAllreudce:
* @li WeightQuantMatmulAllreudce has poor performance when the product of the 1th dimension(b) and 2st dimension(s) of input x1 is small.
* @li x1 can be 2-dimensional or 3-dimensional, and the dimension is (b, s, k) or (m, k). x2 must be
*  2-dimensional. Its dimension is (k, n). The k axis meets the input parameter requirements of the matmul operator.
*  Their k axes are equal. The range of k and n is [1, 65535].
* @li The passed x1, x2, antiquant_scale, or output cannot be a null pointer.
* @li Dimensions except the last one of x3 (non-empty) and output are the same as those of x1. The last
*  dimension of x3 (non-empty) and output are the same as that of x2. If bias is not empty, the shape
*  size is the same as the last dimension of output. The shape of antiquant_scale is [1] in the per-tensor
*  scenario, [1,n]\[n] in the per-channel scenario, and [ceil(k,antiquant_group_size),n] in the per-group scenario. If
*  `n` is 1, there is only one element in both per-tensor and per-channel scenarios, and the per-channel scenario
*  equals the per-tensor scenario. If antiquantOffset is not empty, the shape is the same as that of antiquant_scale.
* @li The data types and data formats of x1, x2, x3 (non-empty), antiquant_scale,
*  antiquantOffset (non-empty), output, and bias (non-empty) must be supported.
* @li The output data types of x1, antiquant_scale, antiquantOffset (non-empty), x3 (non-empty), and
*  bias (non-empty) must be the same.
* @li The value of antiquant_group_size must be within the value range and be a multiple of 32.
* @li The x2 matrix can be transposed or not transposed. The x1 matrix cannot be transposed.
* @li In the long sequence scenario, as b/s or m increases, OOM or computation timeout may occur.
* @li When the format of x2 is FRACTAL_NZ, only two dimensions are supported. CalculateMatmulWeightSizeV2
*  TransMatmulWeightGetWorkspaceSize/TransMatmulWeight needs to be used to convert the format ND into NZ.
* @li The Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component support 1, 2, 4, and 8 cards.
*
* - Constraints for QuantMatmulAllreudce:
* @li QuantMatmulAllreudce has poor performance when the product of the 1th dimension(b) and 2st dimension(s) of input x1 is small.
* @li x1 can be a 2-dimensional or 3-dimensional tensor and cannot be empty. The dimension of x1 is (b, s, k)
*  or (m, k). x2 must be 2-dimensional. Its dimension is (k, n). The k axis meets the input parameter
*  requirements of the mm operator, and their k axes are equal.
* @li Dimensions except the last one of output are the same as those of x1. The last dimension is the same as
*  that of x2. If bias is not empty, the shape size is the same as the last dimension of output. If x3
*  is not empty, the shape size is the same as that of output.
* @li The passed x1, x2, dequantScale, or output cannot be a null pointer.
* @li The data types and data formats of x1, x2, dequantScale, output, bias (non-empty),
*  and x3 (non-empty) must be within the supported ranges.
* @li If output is of FLOAT16 type, the type of dequantScale is INT64 or UINT64 (x3 is not supported in
*  this case). If  output is of BFLOAT16 type, the types of dequantScaleand x3 both are BFLOAT16.
* @li The value of reduce_op must be within the available range. Currently, only sum is supported.
* @li The x2 matrix can be transposed or not transposed. The x1 matrix cannot be transposed.
* @li The Ascend 950 AI processor newly suported hifloat8, float8_e5m2, float8_e4m3, float4_e2m1, float4_e1m2,
*  output suport float32 when input datatype is hifloat8, float8_e5m2, float8_e4m3, float4_e2m1, float4_e1m2.
* @li The Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component support 1, 2, 4, and 8 cards.
*/
REG_OP(MatmulAllReduce)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_INT4, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_BF16, DT_INT32, DT_FLOAT}))
    .OPTIONAL_INPUT(x3, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(dequant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_UINT64, DT_INT64, DT_FLOAT, DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(pertoken_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(comm_quant_scale_1, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(comm_quant_scale_2, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .REQUIRED_ATTR(group, String)
    .ATTR(reduce_op, String, "sum")
    .ATTR(is_trans_a, Bool, false)
    .ATTR(is_trans_b, Bool, false)
    .ATTR(comm_turn, Int, 0)
    .ATTR(antiquant_group_size, Int, 0)
    .ATTR(group_size, Int, 0)
    .ATTR(y_dtype, Int, ge::DataType::DT_UNDEFINED)
    .ATTR(comm_quant_mode, Int, 0)
    .OP_END_FACTORY_REG(MatmulAllReduce)

/**
* @brief Function MatmulAllReduceAddRmsNorm.
* @par Inputs:
* @li x1: A matrix Tensor. The type support float16, bf16, int8.
* @li x2: A matrix Tensor. The type support float16, bf16, int8, int4.
* @li bias: A matrix Tensor. The type support float16, bf16, int32.
* @li residual: A matrix Tensor. The type support float16, bf16.
* @li gamma: A matrix Tensor. The type support float16, bf16.
* @li antiquant_scale: A matrix Tensor. The type support float16, bf16.
* @li antiquant_offset: A matrix Tensor. The type support float16, bf16.
* @li dequant_scale: A matrix Tensor. The type support float16, bf16, uint64. \n

* @par Attributes:
* @li group: A required String identifying the group of ranks participating in the op.
* @li reduce_op: A required string identifying the reduction operation to perform. support "sum", "min", "max" ,"prod", *  currently only support "sum".
* @li is_trans_a: A bool. If True, changes the shape of "x1" from [K, M] to
*  [M, K] before multiplication. Default: false.
* @li is_trans_b: A bool. If True, changes the shape of "x2" from [N, K] to
*  [K, N] before multiplication. Default: false.
* @li comm_turn: An int. Number of communications with AICPU. Default: 0.
* @li antiquant_group_size: An int. Number of per-group for quant. Default: 0.
* @li epsilon: A float32. Default: 1e-06. \n

* @par Outputs:
* @li y: A matrix Tensor. The type support float16, bf16.
* @li norm_out: A matrix Tensor. The type support float16, bf16. \n

* @attention Constraints:
* - Constraints for MatmulAllReduceAddRmsNorm
* @li The application scenario is the same as that [MatmulAllReduce]. MatmulAllReduceAddRmsNorm is disabled in
*  incremental scenarios but enabled in full scenarios.
* @li x1 can be 2-dimensional or 3-dimensional, and the dimension is (b, s, k) or (s, k). x2 must be
*  2-dimensional and its dimension is (k, n). The axis meets the input parameter requirements of the mm operator, and
*  their k axes are equal. If bias is not empty, it is one-dimensional and its dimension is (n).
* @li The input residual must be 3-dimensional and its dimension are (b, s, n). When x1 is two-dimensional,
*  (b*s) of residual is equal to s of x1. The input gamma must be one-dimensional
*  and its dimension is (n).
* @li The dimensions and data types of the outputs y and normOut are the same as those of residual.
*  If bias is not empty, the shape size is the same as the last dimension.
* @li The data types of x1, x2, bias (if supported), residual, gamma, y,
*  and normOut computation input must be the same.
* @li The x2 matrix can be transposed or not transposed. The x1 matrix cannot be transposed.
* @li The value of epsilon must be within the value range (0,1).
* @li The Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component support 1, 2, 4, and 8 cards.
*
* - Constraints for WeightQuantMatmulAllReduceAddRmsNorm
* @li The application scenario is the same as that of [MatmulAllReduce].
*  WeightQuantMatmulAllReduceAddRmsNorm is disabled in incremental scenarios but enabled in full scenarios.
* @li x1 can be 2-dimensional or 3-dimensional, and the dimension is (b, s, k) or (s, k). x2 must be
*  2-dimensional and its dimension is (k, n). The axis meets the input parameter requirements of the mm operator.
*  Their k axes are equal. The range of k and n is [1, 65535]. If bias is not empty,
*  bias has one dimension and its dimension is (n).
* @li The input residual must have three dimensions: (b, s, n). When x1 is two-dimensional,
*  (b*s) of residual is equal to s of x1. gamma must be one-dimensional and its dimension is (n).
* @li The shape of antiquantScale is [1] in the per-tensor scenario; [1,n]\[n] in the per-channel scenario;
*  and [ceil(k,antiquantGroupSize),n] in the per-group scenario. If antiquant_offset is not empty,
*  the shape is the same as that of antiquantScale.
* @li The dimensions and data types of the outputs y and normOut are the same as those of residual.
*  If bias is not empty, the shape size is the same as the last dimension.
* @li The data type of x2 must be int8 or int4. The data types of x1, bias (if supported), residual,
*  gamma, y, or normOut must be the same.
* @li The x2 matrix can be transposed or not transposed. The x1 matrix cannot be transposed.
* @li The value of antiquantGroupSize falls in the range of [32, min(k-1, INT_MAX)] and must be a multiple of 32.
* @li The value of epsilon must be within the value range (0,1).
* @li The Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component support 1, 2, 4, and 8 cards.
*
* - Constraints for QuantMatmulAllReduceAddRmsNorm
* @li The application scenario is the same as that of [MatmulAllReduce].
*  QuantMatmulAllReduceAddRmsNorm is disabled in incremental scenarios but enabled in full scenarios.
* @li The input x1 can be 2-dimensional or 3-dimensional, and the dimension is (b, s, k) or (s, k). x2 must
*  be 2-dimensional and its dimension is (k, n). The axis meets the input parameter requirements of the mm operator,
*  and their k axes are equal. If bias is not empty, bias has one dimension and its dimension is (n).
* @li The input residual must have three dimensions: (b, s, n). When x1 is two-dimensional, (b*s) of
*  residual is equal to s of x1. The input gamma must be one-dimensional and its dimension is (n).
* @li The dimensions and data types of the outputs y and normOut are the same as those of residual. If
*  bias is not empty, the shape size is the same as the last dimension.
* @li If the output residual is of FLOAT16 type, the type of dequantScale is UINT64. If the output residual
*  is of BFLOAT16 type, the type of dequantScale is BFLOAT16.
* @li When the data types of x1 and x2 are INT8, and the data type of bias (if supported) is INT32, the
*  input data types of residual, gamma, y, and normOut must be the same.
* @li The x2 matrix can be transposed or not transposed. The x1 matrix cannot be transposed.
* @li The value of epsilon must be within the value range (0, 1).
* @li The Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component support 1, 2, 4, and 8 cards.
*/
REG_OP(MatmulAllReduceAddRmsNorm)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_INT4}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_BF16, DT_INT32}))
    .INPUT(residual, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(dequant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_UINT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(norm_out, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(group, String)
    .ATTR(reduce_op, String, "sum")
    .ATTR(is_trans_a, Bool, false)
    .ATTR(is_trans_b, Bool, false)
    .ATTR(comm_turn, Int, 0)
    .ATTR(antiquant_group_size, Int, 0)
    .ATTR(epsilon, Float, 1e-06f)
    .OP_END_FACTORY_REG(MatmulAllReduceAddRmsNorm)

/**
* @brief A fusion operator for fused Ema-Adam.

* @par Inputs:
* Six inputs, including:
* @li grad: A Tensor with ND format specifying gradient. Support float16, float32, bfloat16.
* @li var: A Tensor with ND format specifying parameters to be updated. Support float16, float32, bfloat16.
* @li m: A Tensor with ND format specifying first moment. Support float16, float32, bfloat16.
* @li v: A Tensor with ND format specifying second moment. Support float16, float32, bfloat16.
* @li s: A Tensor with ND format specifying weight of EMA. Support float16, float32, bfloat16.
* @li step: A Tensor with ND format specifying time step. Support int64. \n

* @par Attributes:
* @li lr: A Float specifying the learning rate. Optional and defaults to "1e-3".
* @li ema_decay: A Float specifying ema decay. Must be between 0 and 1. Optional and defaults to "0.9999".
* @li beta1: A Float used for computing running averages of gradient. Optional and defaults to "0.9".
* @li beta2: A Float used for computing running averages of gradient's square. Optional and defaults to "0.999".
* @li eps: A Float ued for improving numerical stability. Optional and defaults to "1e-8".
* @li mode: An Integer must be 1 or 0. Set to "1" for AdamW and "0" for L2 regularization. Optional and defaults to "1".
* @li bias_correction: A bool. Set to "true" for bias correction and "false" for no correction. Optional and defaults to "true".
* @li weight_decay: A Float specifying weight decay. Optional and defaults to "0".

* @par Outputs:
* Four outputs, including:
* @li var: A Tensor specifying updated parameters. Must be one of the following types: float16, float32, bfloat16.
* @li m: A Tensor specifying updated first moment. Must be one of the following types: float16, float32, bfloat16.
* @li v: A Tensor specifying updated second moment. Must be one of the following types: float16, float32, bfloat16.
* @li s: A Tensor specifying updated weight of EMA. Must be one of the following types: float16, float32, bfloat16. \n

* @attention Constraints:
* @li grad, var, m, s and v are required to be of the same datatype and shape.
*/
REG_OP(ApplyFusedEmaAdam)
    .INPUT(grad, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(var, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(m, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(v, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(s, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(step, TensorType({DT_INT64}))
    .OUTPUT(var, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(m, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(v, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(s, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(lr, Float, 1e-3f)
    .ATTR(ema_decay, Float, 0.9999)
    .ATTR(beta1, Float, 0.9)
    .ATTR(beta2, Float, 0.999)
    .ATTR(eps, Float, 1e-8f)
    .ATTR(mode, Int, 1)
    .ATTR(bias_correction, Bool, true)
    .ATTR(weight_decay, Float, 0.0)
    .OP_END_FACTORY_REG(ApplyFusedEmaAdam)

/**
* @brief Function InplaceMatmulAllReduceAddRmsNorm.
* @par Inputs:
* @li x1: A matrix Tensor. The type support float16, bf16, int8.
* @li x2: A matrix Tensor. The type support float16, bf16, int8, int4.
* @li bias: A matrix Tensor. The type support float16, bf16, int32.
* @li residual: A matrix Tensor. The type support float16, bf16.
* @li gamma: A matrix Tensor. The type support float16, bf16.
* @li antiquant_scale: A matrix Tensor. The type support float16, bf16.
* @li antiquant_offset: A matrix Tensor. The type support float16, bf16.
* @li dequant_scale: A matrix Tensor. The type support float16, bf16, uint64. \n

* @par Attributes:
* @li group: A required String identifying the group of ranks
*  participating in the op.
* @li reduce_op: A required string identifying the reduction operation to
*  perform. support "sum", "min", "max" ,"prod", currently only support "sum".
* @li is_trans_a: A bool. If True, changes the shape of "x1" from [K, M] to
*  [M, K] before multiplication. Default: false.
* @li is_trans_b: A bool. If True, changes the shape of "x2" from [N, K] to
*  [K, N] before multiplication. Default: false.
* @li comm_turn: An int. Number of communications with AICPU. Default: 0.
* @li antiquant_group_size: An int. Number of per-group for quant. Default: 0.
* @li epsilon: A float32. Default: 1e-06. \n

* @par Outputs:
* @li residual: A matrix Tensor. The type support float16, bf16.
* @li norm_out: A matrix Tensor. The type support float16, bf16. \n

* @attention Constraints:
* - Constraints for InplaceMatmulAllReduceAddRmsNorm
* @li The application scenario is the same as that [MatmulAllReduce].
*  InplaceMatmulAllReduceAddRmsNorm is disabled in incremental scenarios but enabled in full scenarios.
* @li x1 can be 2-dimensional or 3-dimensional, and the dimension is (b, s, k) or (s, k). x2 must be
*  2-dimensional and its dimension is (k, n). The axis meets the input parameter requirements of the mm operator, and
*  their k axes are equal. If bias is not empty, it is one-dimensional and its dimension is (n).
* @li The input residual must be 3-dimensional and its dimension are (b, s, n). When x1 is two-dimensional,
*  (b*s) of residual is equal to s of x1. The input gamma must be one-dimensional and its dimension is (n).
* @li The dimensions and data types of the outputs y and normOut are the same as those of residual.
*  If bias is not empty, the shape size is the same as the last dimension.
* @li The data types of x1, x2, bias (if supported), residual, gamma, y,
*  and normOut computation input must be the same.
* @li The x2 matrix can be transposed or not transposed. The x1 matrix cannot be transposed.
* @li The value of epsilon must be within the value range (0,1).
* @li The Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component support 1, 2, 4, and 8 cards.
*
* - Constraints for InplaceWeightQuantMatmulAllReduceAddRmsNorm
* @li The application scenario is the same as that of [MatmulAllReduce].
*  InplaceWeightQuantMatmulAllReduceAddRmsNorm is disabled in incremental scenarios but enabled in full scenarios.
* @li x1 can be 2-dimensional or 3-dimensional, and the dimension is (b, s, k) or (s, k). x2 must be
*  2-dimensional and its dimension is (k, n). The axis meets the input parameter requirements of the mm operator.
*  Their k axes are equal. The range of k and n is [1, 65535]. If bias is not empty,
*  bias has one dimension and its dimension is (n).
* @li The input residual must have three dimensions: (b, s, n). When x1 is two-dimensional,
* (b*s) of residual is equal to s of x1. gamma must be one-dimensional and its dimension is (n).
* @li The shape of antiquantScale is [1] in the per-tensor scenario; [1,n]\[n] in the per-channel scenario;
*  and [ceil(k,antiquantGroupSize),n] in the per-group scenario. If antiquantOffset is not empty,
*  the shape is the same as that of antiquantScale.
* @li The dimensions and data types of the outputs y and normOut are the same as those of residual.
*  If bias is not empty, the shape size is the same as the last dimension.
* @li The data type of x2 must be int8 or int4. The data types of x1, bias (if supported), residual,
*  gamma, y, or normOut must be the same.
* @li The x2 matrix can be transposed or not transposed. The x1 matrix cannot be transposed.
* @li The value of antiquantGroupSize falls in the range of [32, min(k-1, INT_MAX)] and must be a multiple of 32.
* @li The value of epsilon must be within the value range (0,1).
* @li The Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component support 1, 2, 4, and 8 cards.
*
* - Constraints for InplaceQuantMatmulAllReduceAddRmsNorm
* @li The application scenario is the same as that of [MatmulAllReduce].
*  InplaceQuantMatmulAllReduceAddRmsNorm is disabled in incremental scenarios but enabled in full scenarios.
* @li The input x1 can be 2-dimensional or 3-dimensional, and the dimension is (b, s, k) or (s, k). x2 must
*  be 2-dimensional and its dimension is (k, n). The axis meets the input parameter requirements of the mm operator,
*  and their k axes are equal. If bias is not empty, bias has one dimension and its dimension is (n).
* @li The input residual must have three dimensions: (b, s, n). When x1 is two-dimensional, (b*s) of
*  residual is equal to s of x1. The input gamma must be one-dimensional and its dimension is (n).
* @li The dimensions and data types of the outputs y and normOut are the same as those of residual. If
*  bias is not empty, the shape size is the same as the last dimension.
* @li If the output residual is of FLOAT16 type, the type of dequantScale is UINT64. If the output residual
*  is of BFLOAT16 type, the type of dequantScale is BFLOAT16.
* @li When the data types of x1 and x2 are INT8, and the data type of bias (if supported) is INT32, the
*  input data types of residual, gamma, y, and normOut must be the same.
* @li The x2 matrix can be transposed or not transposed. The x1 matrix cannot be transposed.
* @li The value of epsilon must be within the value range (0,1).
* @li The Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component support 1, 2, 4, and 8 cards.
*/
REG_OP(InplaceMatmulAllReduceAddRmsNorm)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_INT4}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_BF16, DT_INT32}))
    .INPUT(residual, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(dequant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_UINT64}))
    .OUTPUT(residual, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(norm_out, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(group, String)
    .ATTR(reduce_op, String, "sum")
    .ATTR(is_trans_a, Bool, false)
    .ATTR(is_trans_b, Bool, false)
    .ATTR(comm_turn, Int, 0)
    .ATTR(antiquant_group_size, Int, 0)
    .ATTR(epsilon, Float, 1e-06f)
    .OP_END_FACTORY_REG(InplaceMatmulAllReduceAddRmsNorm)

/**
* @brief matmul layer norm reduce.
*
* @par Inputs:
* @li x1: A Tensor. Must be one of the following types: float16.
* @li x2: A Tensor. Must be one of the following types: float16.
* @li bias: A Tensor. Must be one of the following types: float16.
*
* @par Outputs:
* y: A Tensor. Must be one of the following types: float16.
* sum: A Tensor. Must be one of the following types: float16.
* square_sum: A Tensor. Must be one of the following types: float16.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/

REG_OP(MatmulLayerNormReduce)
    .INPUT(x1, TensorType({DT_FLOAT16}))
    .INPUT(weight, TensorType({DT_FLOAT16}))
    .INPUT(bias, TensorType({DT_FLOAT16}))
    .INPUT(x2, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(x3, TensorType({DT_FLOAT16}))
    .OUTPUT(x2, TensorType({DT_FLOAT16}))
    .OUTPUT(sum, TensorType({DT_FLOAT}))
    .OUTPUT(square_sum, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(MatmulLayerNormReduce)


/**
* @brief Function AGLU. \n

* @par Inputs:
* four inputs, including:
* @li x: A required matrix Tensor. The type support float16.
* @li weight1: A required matrix Tensor. The type support float16.
* @li bias1: A optional matrix Tensor. The type support float16.
* @li weight2: A optional matrix Tensor. The type support float16.
* @li bias2: A optional matrix Tensor. The type support float16.

* @par Attributes:
* @li activate_func: A required string. The type of activation.
* @li activate_left: A optional bool. Default: false.

* @par Outputs:
* y: A matrix Tensor. The type support float16. \n

* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
*/
REG_OP(AGLU)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(weight1, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(bias1, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(weight2, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(bias2, TensorType({DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(activate_func, String)
    .ATTR(activate_left, Bool, false)
    .OP_END_FACTORY_REG(AGLU)


/**
* @brief The fusion operator of antiquant function and matmul.

* @par Inputs:
* @li x: A matrix tensor. Shape supports (m,k)/(k,m), Format supports ND.
* The type support float16, bfloat16. The m value must be in [1, 65535] when
* transpose_x is true or [1, 2147483647] when transpose_x is false. The k value
* must be in [1, 65535].
* @li weight: A matrix tensor of quantized weight. Shape supports (n,k)/(k,n),
* Format supports ND/NZ. The type support int8, int4, int32, float8_e5m2, float8_e4m3fn, hifloat8, float4_e2m1, float4_e1m2. \n
* Format must be ND when the type is float8_e5m2, float8_e4m3fn or hifloat8.
* For Ascend 950 AI Processor, transpose_weight must be false when format is NZ. \n
* The k, n value must be in [1, 65535].
* The k value must be even when type is int4/float4_e2m1/float4_e1m2 and transpose_weight
* is true, and the n value must be even when type is int4/float4_e2m1/float4_e1m2 and transpose_weight
* is false. When type is int32, the input is int4-packed data,
* and shape supports (n, k/8)/(k, n/8).
* @li antiquant_scale: A tensor for antiquant scale.
* For different anti quantization modes, the per_tensor mode's shape supports
* (1)/(1,1), the per_channel mode's shape supports (n,)/(1,n)/(n,1), the per_group
* modes' shape supports (ceil(k/antiquant_group_size),n)/
* (n,ceil(k/antiquant_group_size)). Format supports ND. The type support
* float16, bfloat16, uint64 and int64. \n
* When the type is float16 or bfloat16, it should be the same with x. When the type is uint64 or
* int64, there are the following constraints: \n
* 1. antiquant_scale only support the per_channel mode. \n
* 2. x's type should be float16. \n
* 3. transpose_x should be false and transpose_weight should be true. \n
* 4. the m value must be in [1, 96] and the k, n value must be 64 aligned. \n
* 5. weight's type should be int8 and format should be ND. \n
* Additionally, the type supports float8_e8m0 when the weight dtype is float4_e2m1 or float4_e1m2.
* @li antiquant_offset: An Optional tensor for antiquant offset. Shape, format
* and type should be the same with antiquant_scale if antiquant_scale's type is
* not uint64/int64. If antiquant_scale's type is uint64/int64, antiquant_offset should be int32.
* When weight dtype is float8_e5m2, float8_e4m3fn, hifloat8, float4_e2m1 or float4_e1m2, antiquant_offset is ​excluded from selection.
* @li quant_scale: An Optional tensor for quantization parameters.
* Shape supports (1)/(1,n), format supports ND.
* The type support float32, uint64.
* This parameter must exist when type of output is int8.
* This parameter must not exist when type of antiquant_scale is uint64/int64.
* @li quant_offset: An Optional tensor for quantization parameters.
* Shape and format is the same with quant_scale.
* The type support float32.
* This parameter must not exist when type of antiquant_scale is uint64/int64.
* @li bias: An Optional tensor. Shape supports (n)/(1,n), Format supports ND.
* When type of x is float16, the type of bias should be float16. When type of x
* is bfloat16, the type of bias should be float32 or bfloat16. \n
* Specifically, these optional inputs support the shape (0,). At this point,
* it means that the optional input doesn't exist.

* @par Attributes:
* @li transpose_x: A bool. x is transposed if true. Default: false.
* When transpose_x is true, x's shape is (k, m).
* @li transpose_weight: A bool. weight is transposed if true. Default: false.
* When transpose_weight is true, weight's shape is (n, k),
* antiquant_scale's shape should be (1)/(1,1)/(n,)/(n,1)/(n,ceil(k/antiquant_group_size)).
* When transpose_weight is false, weight's shape is (k, n),
* antiquant_scale's shape should be (1)/(1,1)/(n,)/(1,n)/(ceil(k/antiquant_group_size),n).
* @li antiquant_group_size: int, antiquant_group_size must in [0, k-1]
* and antiquant_group_size % 32 == 0. \n
* When antiquant_scale dtype is float8_e8m0, antiquant_group_size must be 32. \n
* When the antiquant_group_size is 0, it means that the per-group mode is not used. Default: 0. \n
* @li dtype: An int. Declare the output dtype, supports 1(float16), 2(int8), 27(bfloat16).
* Default: -1, if the input has quant_scale, the output dtype is int8,
* otherwise the output dtype is the same as the input x dtype.
* @li inner_precise: An int. Calculation mode, only supports 0(high precision), 1(high Performance). Default: 0

* @par Outputs:
* y: A matrix tensor. The type supports float16, bfloat16, int8. The format supports ND.
* The type should be int8 when quant_scale exist.
* The type should be the same with x when quant_scale not exits.

* @attention Constraints:
* @li It is not recommended to use weight NZ format on Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component,
* because its performance may not be better than ND format.
* @li All of these conditions must be met on Atlas Inference Series Product: weight type is int8,
* weight format is NZ, transpose_weight is true, x type is float16, antiquant_scale only support the per_channel mode,
* quant_scale not exists, quant_offset not exists, antiquant_group_size is 0.
* @li Per_channel mode: To improve performance, it is recommended to use the
* weight input after transposition. When the range of m is [65,96], it is
* recommended to use the antiquant_scale with data type UINT64/INT64.
*/
REG_OP(WeightQuantBatchMatmulV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(weight, TensorType({DT_INT8, DT_INT4, DT_INT32, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2}))
    .INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_UINT64, DT_INT64, DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16, DT_INT32}))
    .OPTIONAL_INPUT(quant_scale, TensorType({DT_FLOAT, DT_UINT64}))
    .OPTIONAL_INPUT(quant_offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .ATTR(transpose_x, Bool, false)
    .ATTR(transpose_weight, Bool, false)
    .ATTR(antiquant_group_size, Int, 0)
    .ATTR(dtype, Int, -1)
    .ATTR(inner_precise, Int, 0)
    .OP_END_FACTORY_REG(WeightQuantBatchMatmulV2)


/**
* @brief Function GroupedMatmul. This op computes groups of matmuls.

* @par Inputs:
* @li x: A Tensor List. Format supports ND. The type support float16, bfloat16, int8, float32, hifloat8, float8_e5m2, float8_e4m3fn, float4_e2m1.
* Maximum length of x is 128.
* @li weight: A Tensor List of weight. Format supports ND/NZ. The type support float16, bfloat16, int8, float32, int4, int32, hifloat8, float8_e5m2, float8_e4m3fn, float4_e2m1.
* Maximum length of weight is 128. When type is float32/int32 in fake-quantization modes which only supported in Ascend 950 AI Processor, the input is float4_e2m1/int4-packed data.
* @li bias: A Tensor List of bias. Format supports ND. The type support float16, float32, int32, bfloat16.
* Length of bias must be the same as weight.
* @li scale: A Tensor List of scale. Indicating scaling factor of quantization parameter.
* Format supports ND. The type support uint64, int64, bfloat16, float32, float8_e8m0. Length of scale must be the same as weight.
* @li offset: A Tensor List of offset. Indicating the offset of the quantization parameter.
* Format supports ND. The type support float32. Length of offset must be the same as weight.
* @li antiquant_scale: A Tensor List of antiquant_scale. Indicating the scaling factor of the fake-quantization parameter.
* Format supports ND. The type support float16, bfloat16, float8_e8m0. Length of antiquant_scale must be the same as weight.
* @li antiquant_offset: A Tensor List of antiquant_offset. Indicating the offset of the fake-quantization parameter.
* Format supports ND. The type support float16, bfloat16. Length of antiquant_offset must be the same as weight.
* @li group_list: a Tensor. Indicating the matmul size distribution along separated dimension.
* Format supports ND. The type support int64.
* @li per_token_scale: A Tensor of per_token_scale.
* Indicating the scaling factor of the quantization parameter, introduced by x quantization.
* Format supports ND. The type support float32, float8_e8m0.

* @par Attributes:
* @li split_item: An int. Indicate whether required separated y. Default: 0.
* @li dtype: An int. only invalid for quant case. -1, output data type is int8.
* 0, output data type is float16. 1, output data type is bfloat16. 2, output data type is int32. 3, output data type is float32. Default: 0.
* @li transpose_weight: A bool. Reserved parameter,
* indicate wether input weight is transposed, not enabled. Default: false.
* @li transpose_x: A bool. Reserved parameter,
* indicate wether input x is transposed, not enabled. Default: false.
* @li group_type: An int. Indicates the grouped dimension in group_list. -1, group_list is null.
* 0, grouped dimension is M. 1, grouped dimension is N, not supported currently. 2, grouped dimension is K. Default: -1.
* @li group_list_type: An int. Indicates whether the value in group_list is cumsum or count.
* 0, value in group_list is cumsum. 1, value in group_list is count. Default: 0.
* @li act_type: An int. Indicate activation function type. Value range 0-5. Default: 0.
* 0, no activation. 1, relu. 2, gelu_tanh. 3, gelu_err_func, not supported currently. 4, fastgelu. 5, silu.
* @li tuning_config: A ListInt. The first element in the array indicates the expected number of tokens processed by each expert.
* The operator tiling performs optimal tiling based on the first element in the array. Default: {0}.

* @par Outputs:
* y: A Tensor List. Format supports ND.
* The type support float16, bfloat16, int8, float32, int32.
*\n
*\n
* - The following are the supported data formats and corresponding data types (for Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component):
*\n
*\n
| Tensor    | x       | weight    | bias    | scale   | offset  | antiquant_scale | antiquant_offset | per_token_scale | y       |
| :-------: | :-----: | :-------: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| Format1   | ND      | ND        | ND      | ND      | ND      | ND      | ND      | ND      | ND      |
| Data Type | float16 | float16   | float16 | uint64  | float32 | float16 | float16 |    -    | float16 |
|           | float32 | float32   | float32 | uint64  | float32 | float16 | float16 |    -    | float32 |
|           | bfloat16| bfloat16  | float32 | uint64  | float32 | float16 | float16 |    -    | bfloat16|
|           | int8    | int8      | int32   | uint64  | float32 | float16 | float16 |    -    | int8    |
|           | int8    | int8      | int32   | bfloat16| float32 | float16 | float16 | float32 | bfloat16|
|           | int8    | int8      | int32   | float32 | float32 | float16 | float16 | float32 | float16 |
|           | int8    | int8      | int32   | uint64  | float32 | float16 | float16 |    -    | int32   |
|           | float16 | int8      | float16 | uint64  | float32 | float16 | float16 |    -    | float16 |
|           | bfloat16| int8      | float32 | uint64  | float32 | bfloat16| bfloat16|    -    | bfloat16|
|           | float16 | int4      | float16 | uint64  | float32 | float16 | float16 |    -    | float16 |
|           | bfloat16| int4      | float32 | uint64  | float32 | bfloat16| bfloat16|    -    | bfloat16|
| Format2   | ND      | FRACTAL_NZ| ND      | ND      | ND      | ND      | ND      | ND      | ND      |
| Data Type | int8    | int8      | int32   | bfloat16| float32 | float16 | float16 | float32 | bfloat16|
|           | int8    | int8      | int32   | float32 | float32 | float16 | float16 | float32 | float16 |
|           | int8    | int8      | int32   | uint64  | float32 | float16 | float16 |    -    | int32   |
*\n
*\n
* - The following are the supported data formats and data types (for Atlas Inference Series Product):
*\n
*\n
| Tensor    | x       | weight    | bias    | scale   | offset  | antiquant_scale | antiquant_offset | per_token_scale | y       |
| :-------: | :-----: | :-------: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| Format2   | ND      | FRACTAL_NZ| ND      | ND      | ND      | ND      | ND      | ND      | ND      |
| Data Type | float16 | float16   | float16 | uint64  | float32 | float16 | float16 |    -    | float16 |
*\n
*\n
* - The following are the supported data formats and data types (for Ascend 950 AI Processor):
*\n
| Tensor    | x                         | weight                                  | bias                   | scale            | offset  | antiquant_scale | antiquant_offset | per_token_scale | y   |
| :-------: | :-----------------------: | :-------------------------------------: | :--------------------: | :--------------: | :-----: | :---------: | :-----: | :---------: | :----------------------: |
| Format1   | ND                        | ND                                      | ND                     | ND               | ND      | ND          | ND      | ND          | ND                       |
| Data Type | int8                      | int8                                    | int32                  | uint64/int64     | float32 | float16     | float16 |      -      | float16/bfloat16/int8    |
|           | int8                      | int8                                    | int32                  | uint64/int64     | float32 | float16     | float16 |      -      | int32                    |
|           | int8                      | int8                                    | int32/float16/float32  | float32          | float32 | float16     | float16 | -/float32   | float16                  |
|           | int8                      | int8                                    | int32/bfloat16/float32 | bfloat16/float32 | float32 | float16     | float16 | -/float32   | bfloat16                 |
|           | hifloat8                  | hifloat8                                | float32                | uint64/int64     | float32 | float16     | float16 |      -      | float16/bfloat16/float32 |
|           | hifloat8                  | hifloat8                                | float32                | float32          | float32 | float16     | float16 | -/float32   | float16/bfloat16/float32 |
|           | float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2               | float32                | uint64/int64     | float32 | float16     | float16 |      -      | float16/bfloat16/float32 |
|           | float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2               | float32                | float32          | float32 | float16     | float16 | -/float32   | float16/bfloat16/float32 |
|           | float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2               | float32                | float8_e8m0      | float32 | float16     | float16 | float8_e8m0 | float16/bfloat16/float32 |
|           | float4_e2m1               | float4_e2m1                             | float32                | float8_e8m0      | float32 | float16     | float16 | float8_e8m0 | float16/bfloat16/float32 |
|           | float16                   | float16                                 | float16/float32        |       -          |    -    |      -      |    -    |      -      | float16                  |
|           | bfloat16                  | bfloat16                                | bfloat16/float32       |       -          |    -    |      -      |    -    |      -      | bfloat16                 |
|           | float16                   | int8/float8_e4m3fn/float8_e5m2/hifloat8 | float16                | uint64           | float32 | float16     | float16 |      -      | float16                  |
|           | bfloat16                  | int8/float8_e4m3fn/float8_e5m2/hifloat8 | bfloat16/float32       | uint64           | float32 | bfloat16    | bfloat16|      -      | bfloat16                 |
|           | float16                   | int4                                    | float16                | uint64           | float32 | float16     | float16 |      -      | float16                  |
|           | bfloat16                  | int4                                    | bfloat16/float32       | uint64           | float32 | bfloat16    | bfloat16|      -      | bfloat16                 |
| Format2   | ND                        | FRACTAL_NZ                              | ND                     | ND               | ND      | ND          | ND      | ND          | ND                       |
| Data Type | int8                      | int8                                    | int32                  | uint64/int64     | float32 | float16     | float16 |      -      | float16/bfloat16         |
|           | int8                      | int8                                    | int32/float16/float32  | float32          | float32 | float16     | float16 | -/float32   | float16                  |
|           | int8                      | int8                                    | int32/bfloat16/float32 | bfloat16/float32 | float32 | float16     | float16 | -/float32   | bfloat16                 |
|           | int8                      | int4                                    | float32                | float32          | float32 | float16     | float16 | float32     | float16/bfloat16         |
|           | float16                   | float4_e2m1/float32                     | float16                | uint64           | float32 | float8_e8m0 | float16 |      -      | float16                  |
|           | bfloat16                  | float4_e2m1/float32                     | bfloat16/float32       | uint64           | float32 | float8_e8m0 | bfloat16|      -      | bfloat16                 |
|           | float8_e4m3fn             | float4_e2m1/float32                     | float16                | uint64           | float32 | float8_e8m0 | float16 | float8_e8m0 | float16                  |
|           | float8_e4m3fn             | float4_e2m1/float32                     | bfloat16               | uint64           | float32 | float8_e8m0 | bfloat16| float8_e8m0 | bfloat16                 |
| Format3   | ND                        | FRACTAL_NZ_C0_16                        | ND                     | ND               | ND      | ND          | ND      | ND          | ND                       |
| Data Type | float16                   | float4_e2m1                             | float16                | uint64           | float32 | float8_e8m0 | float16 |      -      | float16                  |
|           | bfloat16                  | float4_e2m1                             | bfloat16/float32       | uint64           | float32 | float8_e8m0 | float16 |      -      | bfloat16                 |
| Format4   | ND                        | FRACTAL_NZ_C0_32                        | ND                     | ND               | ND      | ND          | ND      | ND          | ND                       |
| Data Type | float8_e4m3fn             | float4_e2m1                             | float16                | uint64           | float32 | float8_e8m0 | float16 | float8_e8m0 | float16                  |
|           | float8_e4m3fn             | float4_e2m1                             | bfloat16               | uint64           | float32 | float8_e8m0 | bfloat16| float8_e8m0 | bfloat16                 |
|           | int8                      | int32                                   | float32                | float32          | float32 | float16     | float16 | float32     | bfloat16/float16                |
*\n
* - The following are the supported data types and no quantization modes(for Ascend 950 AI Processor):
| x                         | weight                    | bias                   |   y                      | group_type |
| :-----------------------: | :-----------------------: | :--------------------: | :----------------------: | :--------: |
| float16                   | float16                   | float16/float32        | float16                  |-1/0/2      |
| bfloat16                  | bfloat16                  | bfloat16/float32       | bfloat16                 |-1/0/2      |
| float32                   | float32                   | float32/float32        | float32                  |-1/0/2      |
*\n
* - The following are the supported data types and quantization modes(for Ascend 950 AI Processor):
*\n
pertensor-perchannel && pertensor-pertensor:
*\n
  | x type                    | weight type               | per_token_scale type | scale type       | group_type |
  | ------------------------- | ------------------------- | -------------------- | ---------------- | ---------- |
  | int8                      | int8                      | -                    | uint64/int64     | 0          |
  | int8                      | int8                      | -                    | float32/bfloat16 | 0          |
  | float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2 | -                    | uint64/int64     | 0          |
  | hifloat8                  | hifloat8                  | -                    | uint64/int64     | 0          |
  | float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2 | -/float32            | float32          | 0/2        |
  | hifloat8                  | hifloat8                  | -/float32            | float32          | 0/2        |
*\n
pertoken-perchannel && pertoken-pertensor
*\n
  | x type                    | weight type               | per_token_scale type | scale type       | group_type |
  | ------------------------- | ------------------------- | -------------------- | ---------------- | ---------- |
  | int8                      | int8                      | float32              | float32/bfloat16 | 0          |
  | float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2 | float32              | float32          | 0/2        |
  | hifloat8                  | hifloat8                  | float32              | float32          | 0/2        |
*\n
pergroup-perblock(group_size_m = 1, group_size_k = 128, group_size_n = 128):
*\n
  | x type                    | weight type               | per_token_scale type | scale type       | group_type |
  | ------------------------- | ------------------------- | -------------------- | ---------------- | ---------- |
  | float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2 | float32              | float32          | 0/2        |
  | hifloat8                  | hifloat8                  | float32              | float32          | 0/2        |
*\n
mx quant(group_size_k = 32)：
*\n
  | x type                    | weight type               | per_token_scale type | scale type       | group_type |
  | ------------------------- | ------------------------- | -------------------- | ---------------- | ---------- |
  | float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2 | float8_e8m0          | float8_e8m0      | 0/2        |
  | float4_e2m1               | float4_e2m1               | float8_e8m0          | float8_e8m0      | 0          |
*\n
perchannel：
*\n
  | x type                    | weight type                             | antiquant_scale type | antiquant_offset type | group_type |
  | ------------------------- | --------------------------------------- | -------------------- | --------------------- | ---------- |
  | float16                   | int8/float8_e4m3fn/float8_e5m2/hifloat8 | float16              | float16               | 0          |
  | bfloat16                  | int8/float8_e4m3fn/float8_e5m2/hifloat8 | bfloat16             | bfloat16              | 0          |
  | float16                   | int4                                    | float16              | float16               | 0          |
  | bfloat16                  | int4                                    | bfloat16             | bfloat16              | 0          |
*\n
pertoken-(perchannel & pergroup)：
*\n
  | x type                    | weight type               | per_token_scale type | scale type            | antiquant_scale type | group_type |
  | ------------------------- | ------------------------- | -------------------- | --------------------- | -------------------- | ---------- |
  | int8                      | int4/int32                | float32              | float32               | float16              | 0          |
*\n
mx fake-quant(group_size_k = 32)：
*\n
  | x type                    | weight type                     | antiquant_scale type | antiquant_offset type | per_token_scale type | group_type |
  | ------------------------- | ------------------------------- | -------------------- | --------------------- | -------------------- | ---------- |
  | float16                   | float4_e2m1/float32             | float8_e8m0          | float16               |           -          | 0          |
  | bfloat16                  | float4_e2m1/float32             | float8_e8m0          | bfloat16              |           -          | 0          |
  | float8_e4m3fn             | float4_e2m1/float32             | float8_e8m0          | float16/bfloat16      | float8_e8m0          | 0          |
*\n
* - The following are the supported quantization modes, data types and shapes(for Ascend 950 AI Processor):
*\n
| quantization         | group_type | x type                             | scale type                    | x shape   | weight shape        | y shape   | scale shape                               | per_token_scale shape | bias shape |
|----------------------|------------|------------------------------------|-------------------------------|-----------|---------------------|-----------|-------------------------------------------|-----------------------|------------|
| pertensor-pertensor  | 0          | int8                               | uint64/int64/float32/bfloat16 | [(M,K)]   | [(B,N,K)]/[(B,K,N)] | [(M,N)]   | [(B,)]/[(B,1)]                            | -                     | -/[(B,N)]  |
| pertensor-pertensor  | 0          | float8_e4m3fn/float8_e5m2/hifloat8 | uint64/int64                  | [(M,K)]   | [(B,N,K)]/[(B,K,N)] | [(M,N)]   | [(B,)]/[(B,1)]                            | -                     | -          |
| pertensor-pertensor  | 0          | float8_e4m3fn/float8_e5m2/hifloat8 | float32                       | [(M,K)]   | [(B,N,K)]/[(B,K,N)] | [(M,N)]   | [(B,)]/[(B,1)]                            | -/(B,)/(B,1)          | -          |
| pertensor-pertensor  | 2          | float8_e4m3fn/float8_e5m2/hifloat8 | float32                       | [(K,M)]   | [(K,N)]             | [(B,M,N)] | [(B,)]/[(B,1)]                            | -/(B,)/(B,1)          | -          |
| pertensor-perchannel | 0          | int8                               | uint64/int64/float32/bfloat16 | [(M,K)]   | [(B,N,K)]/[(B,K,N)] | [(M,N)]   | [(B,N)]                                   | -                     | -/[(B,N)]  |
| pertensor-perchannel | 0          | float8_e4m3fn/float8_e5m2/hifloat8 | uint64/int64                  | [(M,K)]   | [(B,N,K)]/[(B,K,N)] | [(M,N)]   | [(B,N)]                                   | -                     | -          |
| pertensor-perchannel | 0          | float8_e4m3fn/float8_e5m2/hifloat8 | float32                       | [(M,K)]   | [(B,N,K)]/[(B,K,N)] | [(M,N)]   | [(B,N)]                                   | -/(B,)/(B,1)          | -          |
| pertensor-perchannel | 2          | float8_e4m3fn/float8_e5m2/hifloat8 | float32                       | [(K,M)]   | [(K,N)]             | [(B,M,N)] | [(B,N)]                                   | -/(B,)/(B,1)          | -          |
| pertoken-pertensor   | 0          | int8                               | float32/bfloat16              | [(M,K)]   | [(B,N,K)]/[(B,K,N)] | [(M,N)]   | [(B,)]/[(B,1)]                            | (M,)                  | -/[(B,N)]  |
| pertoken-pertensor   | 0          | float8_e4m3fn/float8_e5m2/hifloat8 | float32                       | [(M,K)]   | [(B,N,K)]/[(B,K,N)] | [(M,N)]   | [(B,)]/[(B,1)]                            | (M,)                  | -          |
| pertoken-pertensor   | 2          | float8_e4m3fn/float8_e5m2/hifloat8 | float32                       | [(K,M)]   | [(K,N)]             | [(B,M,N)] | [(B,)]/[(B,1)]                            | (B,M)                 | -          |
| pertoken-perchannel  | 0          | int8                               | float32/bfloat16              | [(M,K)]   | [(B,N,K)]/[(B,K,N)] | [(M,N)]   | [(B,N)]                                   | (M,)                  | -/[(B,N)]  |
| pertoken-perchannel  | 0          | float8_e4m3fn/float8_e5m2/hifloat8 | float32                       | [(M,K)]   | [(B,N,K)]/[(B,K,N)] | [(M,N)]   | [(B,N)]                                   | (M,)                  | -          |
| pertoken-perchannel  | 2          | float8_e4m3fn/float8_e5m2/hifloat8 | float32                       | [(K,M)]   | [(K,N)]             | [(B,M,N)] | [(B,N)]                                   | (B,M)                 | -          |
| pergroup-perblock    | 0          | float8_e4m3fn/float8_e5m2/hifloat8 | float32                       | [(M,K)]   | [(B,N,K)]           | [(M,N)]   | [(B,ceil(N/128),ceil(K/128))]             | (M,ceil(K/128))       | -          |
| pergroup-perblock    | 0          | float8_e4m3fn/float8_e5m2/hifloat8 | float32                       | [(M,K)]   | [(B,K,N)]           | [(M,N)]   | [(B,ceil(K/128),ceil(N/128))]             | (M,ceil(K/128))       | -          |
| pergroup-perblock    | 2          | float8_e4m3fn/float8_e5m2/hifloat8 | float32                       | [(K,M)]   | [(K,N)]             | [(B,M,N)] | [(K/128+B, ceil(N/128))]                  | (K/128+B,M)           | -          |
| mx                   | 0          | float8_e4m3fn/float8_e5m2          | float8_e8m0                   | [(M,K)]   | [(B,N,K)]/[(B,K,N)] | [(M,N)]   | [(B,N,ceil(K/64),2)]/[(B,ceil(K/64),N,2)] | (M,ceil(K/64),2)      | -          |
| mx                   | 2          | float8_e4m3fn/float8_e5m2          | float8_e8m0                   | [(K,M)]   | [(K,N)]             | [(B,M,N)] | [(K/64+B,N,2)]                            | (K/64+B,M,2)          | -          |
| mx                   | 0          | float4_e2m1                        | float8_e8m0                   | [(M,K)]   | [(B,N,K)]/[(B,K,N)] | [(M,N)]   | [(B,N,ceil(K/64),2)]/[(B,ceil(K/64),N,2)] | (M,ceil(K/64),2)      | -/[(B,N)]  |

| quantization                      | group_type | x type        | weight type                        | per_token_scale type | scale type | antiquant_scale type | antiquant_offset type | x shape | weight shape                      | y shape | per_token_scale shape | scale shape | antiquant_scale shape       | antiquant_offset shape |
|---------------------------------- |------------|---------------|------------------------------------|----------------------|------------|----------------------|-----------------------|---------------------------------------------|---------|-----------------------|-------------|-----------------------------|------------------------|
| perchannel                        | 0          | float16       | int8                               | -                    | uint64     | float16              | float16               | [(M,K)] | [(B,N,K)]/[(B,K,N)]               | [(M,N)] | -                     | [(0)]       | [(B,N,)]                    | [(0)]/[(B,N,)]         |
| perchannel                        | 0          | bfloat16      | int8                               | -                    | uint64     | bfloat16             | bfloat16              | [(M,K)] | [(B,N,K)]/[(B,K,N)]               | [(M,N)] | -                     | [(0)]       | [(B,N,)]                    | [(0)]/[(B,N,)]         |
| perchannel                        | -1         | float16       | int8                               | -                    | uint64     | float16              | float16               | [(M1,K1),(M2,K2),...] | [(N1,K1),(N2,K2),...]/[(K1,N1),(K2,N2),...] | [(M1,N1),(M2,N2),...] | - | [(0)] | [(N1,),(N2,)...] | [(0)]/[(N1,),(N2,)...] |
| perchannel                        | -1         | bfloat16      | int8                               | -                    | uint64     | bfloat16             | bfloat16              | [(M1,K1),(M2,K2),...] | [(N1,K1),(N2,K2),...]/[(K1,N1),(K2,N2),...] | [(M1,N1),(M2,N2),...] | - | [(0)] | [(N1,),(N2,)...] | [(0)]/[(N1,),(N2,)...] |
| perchannel                        | 0          | float16       | float8_e4m3fn/float8_e5m2/hifloat8 | -                    | uint64     | float16              | float16               | [(M,K)] | [(B,N,K)]                         | [(M,N)] | -                     | [(0)]       | [(B,N,)]                    | [(0)]                  |
| perchannel                        | 0          | bfloat16      | float8_e4m3fn/float8_e5m2/hifloat8 | -                    | uint64     | bfloat16             | bfloat16              | [(M,K)] | [(B,N,K)]                         | [(M,N)] | -                     | [(0)]       | [(B,N,)]                    | [(0)]                  |
| perchannel                        | 0          | float16       | int4                               | -                    | uint64     | float16              | float16               | [(M,K)] | [(B,N,K)]/[(B,K,N)]               | [(M,N)] | -                     | [(0)]       | [(B,N,)]                    | [(0)]/[(B,N,)]         |
| perchannel                        | 0          | bfloat16      | int4                               | -                    | uint64     | bfloat16             | bfloat16              | [(M,K)] | [(B,N,K)]/[(B,K,N)]               | [(M,N)] | -                     | [(0)]       | [(B,N,)]                    | [(0)]/[(B,N,)]         |
| perchannel                        | -1         | float16       | int4                               | -                    | uint64     | float16              | float16               | [(M1,K1),(M2,K2),...] | [(N1,K1),(N2,K2),...]/[(K1,N1),(K2,N2),...] | [(M1,N1),(M2,N2),...] | - | [(0)] | [(N1,),(N2,)...] | [(0)]/[(N1,),(N2,)...] |
| perchannel                        | -1         | bfloat16      | int4                               | -                    | uint64     | bfloat16             | bfloat16              | [(M1,K1),(M2,K2),...] | [(N1,K1),(N2,K2),...]/[(K1,N1),(K2,N2),...] | [(M1,N1),(M2,N2),...] | - | [(0)] | [(N1,),(N2,)...] | [(0)]/[(N1,),(N2,)...] |
| pertoken-(perchannel & pergroup)  | 0          | int8          | int4                               | float32              | float32    | float16              | float16               | [(M,K)] | [(B,ceil(N/32),ceil(K/16),16,32)] | [(M,N)] | (M,)                  | [(B,N,)]    | [(B,ceil(K/group_size),N,)] | [(0)]                  |
| mx                                | 0          | float16       | float4_e2m1                        | -                    | uint64     | float8_e8m0          | float16               | [(M,K)] | [(B,ceil(N/16),ceil(K/16),16,16)] | [(M,N)] | -                     | [(0)]       | [(B,ceil(K/32),N,)]         | [(0)]                  |
| mx                                | 0          | bfloat16      | float4_e2m1                        | -                    | uint64     | float8_e8m0          | bfloat16              | [(M,K)] | [(B,ceil(N/16),ceil(K/16),16,16)] | [(M,N)] | -                     | [(0)]       | [(B,ceil(K/32),N,)]         | [(0)]                  |
| mx                                | 0          | float16       | float32                            | -                    | uint64     | float8_e8m0          | float16               | [(M,K)] | [(B,ceil(N/16),ceil(K/16),16,2)]  | [(M,N)] | -                     | [(0)]       | [(B,ceil(K/32),N,)]         | [(0)]                  |
| mx                                | 0          | bfloat16      | float32                            | -                    | uint64     | float8_e8m0          | bfloat16              | [(M,K)] | [(B,ceil(N/16),ceil(K/16),16,2)]  | [(M,N)] | -                     | [(0)]       | [(B,ceil(K/32),N,)]         | [(0)]                  |
| mx                                | 0          | float8_e4m3fn | float4_e2m1                        | float8_e8m0          | uint64     | float8_e8m0          | float16               | [(M,K)] | [(B,ceil(K/32),ceil(N/16),16,32)] | [(M,N)] | (M,ceil(K/64)*2)      | [(0)]       | [(B,N,ceil(K/64)*2,)]       | [(0)]                  |
| mx                                | 0          | float8_e4m3fn | float4_e2m1                        | float8_e8m0          | uint64     | float8_e8m0          | bfloat16              | [(M,K)] | [(B,ceil(K/32),ceil(N/16),16,32)] | [(M,N)] | (M,ceil(K/64)*2)      | [(0)]       | [(B,N,ceil(K/64)*2,)]       | [(0)]                  |
| mx                                | 0          | float8_e4m3fn | float32                            | float8_e8m0          | uint64     | float8_e8m0          | float16               | [(M,K)] | [(B,ceil(K/32),ceil(N/16),16,4)]  | [(M,N)] | (M,ceil(K/64)*2)      | [(0)]       | [(B,N,ceil(K/64)*2,)]       | [(0)]                  |
| mx                                | 0          | float8_e4m3fn | float32                            | float8_e8m0          | uint64     | float8_e8m0          | bfloat16              | [(M,K)] | [(B,ceil(K/32),ceil(N/16),16,4)]  | [(M,N)] | (M,ceil(K/64)*2)      | [(0)]       | [(B,N,ceil(K/64)*2,)]       | [(0)]                  |
*\n
* @attention Constraints
* @li single-tensor: for tensor list type of input and output, tensors for different groups are not separated, and only one tensor in the tensor list.
* @li multi-tensor: for tensor list type of input and output, tensors for different groups are separated, and multi separated tensors in the tensor list.
* @li If group_list is passed, when group_list_type is 0, it must be a non-negative monotone non-decreasing array; when group_list_type is 1, it must be a non-negative array.
* @li Quantization and fake-quantization are supported only when group_type is set to -1 or 0 for Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component and Atlas Inference Series Product.
* Quantization is supported only when group_type is set to 0 or 2 for Ascend 950 AI Processor.
* @li Quantization with y data type bfloat16 or float16 or float32 is only supported in single-tensor x, single-tensor weight, single-tensor y cases.
* @li Each axis for tensors in x and weight for each group of matmul should be less or equal to 2147483647 (maximum of data type int32) after aligning to 32 byte.
* @li When the dtype of the weight is int4, the K axis size should be even if the weight is transposed, otherwise, the N axis size should be even.
* @li In following tables, "S" stands for single-tensor, and "M" stands for multi-tensor, expressed in the sequence of x, weight, y.
* For example, "SMS" indicates single-tensor x, multi-tensor weight, and single-tensor y;
* "optional-dynamic inputs" stands for DYNAMIC_INPUT not needed in all scenarios, which includes bias/scale/antiquant_scale/antiquant_offset. Offset is not needed in all scenarios.
*\n
* The following are the supported shapes and constrains for different scenarios(for Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component and Atlas Inference Series Product):
*\n
| group_type | supported scenario | x shape | weight shape | y shape | optional-dynamic inputs shape if needed | group_list shape if passed | per_token_scale shape if passed |    constrains    |
| :--------: | :----------------: | :-----: | :----------: | :-----: | :-------------------------------------: | :------------------------: | :-----------------------------: | :--------------: |
| -1         |        MMM         | [(M1,K1),(M2,K2),...] | [(K1,N1),(K2,N2),...] | [(M1,N1),(M2,N2),...] | [(N1),(N2),...] | not support | not support |1) K1,K2,... < 65536.<br>2) N1,N2,... < 65536.<br>3) B <= 128.|
| 0          |        SSS         | [(M,K)] | [(B,K,N)] | [(M,N)] | [(B,N)] | (B) | (M) |1) K < 65536.<br>2) N < 65536.<br>3) B <= 1024.|
| 0          |        SMS         | [(M,K)] | [(K,N),(K,N),...] | [(M,N)] | [(N),(N),...] | (B) | not support |1) K < 65536.<br>2) N < 65536.<br>3) B <= 128.|
| 0          |        MMS         | [(M1,K1),(M2,K2),...] | [(K1,N),(K2,N),...] | [(M,N)] | [(N),(N),...] | (B) | not support |1) K1,K2,... < 65536.<br>2) N < 65536.<br>3) B <= 128.|
*\n
*\n
* The following are the supported shapes and constrains for different scenarios(for for Ascend 950 AI Processor):
*\n
| group_type | supported scenario | x shape | weight shape | y shape | optional-dynamic inputs shape if needed | group_list shape if passed | per_token_scale shape if passed |    constrains    |
| :--------: | :----------------: | :-----: | :----------: | :-----: | :-------------------------------------: | :------------------------: | :-----------------------------: | :--------------: |
| -1         |        MMM         | [(M1,K1),(M2,K2),...] | [(K1,N1),(K2,N2),...] | [(M1,N1),(M2,N2),...] | [(N1),(N2),...] | not support | not support |1) B <= 128.|
| 0          |        SSS         | [(M,K)] | [(B,K,N)]/[(B,N,K)] | [(M,N)] | See the constrains below | (B) | See the constrains below |1) B <= 1024.|
| 0          |        SMS         | [(M,K)] | [(K,N),(K,N),...] | [(M,N)] | [(N),(N),...] | (B) | not support |1) B <= 128.|
| 0          |        MMS         | [(M1,K1),(M2,K2),...] | [(K1,N),(K2,N),...] | [(M,N)] | [(N),(N),...] | (B) | not support |1) B <= 128.|
| 2          |        SSS         | [(K,M)] | [(K,N)] | [(B,M,N)] | See the constrains below | (B) | See the constrains below |1) B <= 1024.|
*\n
* The constrains of inputs' shape in quantization(for for Ascend 950 AI Processor) can be found in the table "The following are the supported quantization modes, data types and shapes(for Ascend 950 AI Processor)" above.
* @li Shape of offset and not needed optional-dynamic inputs is [(0)].
* @li Shape of weight indicated in above table corresponds to data format ND. Currently, only single-tensor x, single-tensor weight, single-tensor y with group_type 0 case supports weight with format NZ.
* @li When weight has format NZ, N axis should align to 32 bytes, i.e. if weight has data type int8 , N axis align to 32; if weight has data type float16 , N axis align to 16.
* @li In the MXFP4 scenario, the constraint that k must be even and not equal 2. Under non-transposed weight, n must also be even.
*\n
*\n
* @li The following are the supported group_type and constrains for different scenarios:
| group_type | supported scenario |    constrains    |
| :--------: | :----------------: | :--------------: |
| -1         |        MMM         |1) tensors in x support dim num 2-6, tensors in weight support dim num 2, tensors in y should have the same dim num with tensor in x.<br>2) group_list must be passed as null.|
| 0          |        SSS         |1) tensor in x and y should have dim num 2, tensor in weight should have dim num 3.<br>2) group_list must be passed, if group_list_type equals to 0, the last value in group_list must be no greater than the first dimension of tensor in x; if group_list_type equals to 1, sum of values in group_list must be no greater than the first dimension of tensor in x.|
| 0          |        SMS         |1) tensors in x, weight and y should have dim num 2.<br>2) group_list must be passed, if group_list_type equals to 0, the last value in group_list must be no greater than the first dimension of tensor in x; if group_list_type equals to 1, sum of values in group_list must be no greater than the first dimension of tensor in x.<br>3) The K axis and N axis of each tensor in weight must be the same.|
| 0          |        MMS         |1) tensors in x, weight and y should have dim num 2.<br>2) if group_list is passed, when group_list_type equals to 0, difference between two adjacent value in group_list should be consistent with the first dimension of each tensor in x; when group_list_type equals to 1, values in group_list should be consistent with the first dimension of each tensor in x.<br>3) The N axis of each tensor in weight must be the same.|
| 2          |        SSS         |1) tensor in x and weight should have dim num 2, tensor in y should have dim num 3.<br>2) group_list must be passed, if group_list_type equals to 0, the last value in group_list must be no greater than the first dimension of tensor in x; if group_list_type equals to 1, sum of values in group_list must be no greater than the first dimension of tensor in x.|
*\n
* @li Atlas Inference Series Product Constraints: \n
* - Atlas Inference Series Product only supports single-tensor x, single-tensor weight, single-tensor y, and group_type 0 cases. \n
* - Atlas Inference Series Product only supports x, weight and y have data type float16, and N axis should align to 16. \n
*/
  REG_OP(GroupedMatmul)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8, DT_FLOAT4_E2M1}))
    .DYNAMIC_INPUT(weight, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT, DT_INT4, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8, DT_FLOAT4_E2M1}))
    .DYNAMIC_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_BF16}))
    .DYNAMIC_INPUT(scale, TensorType({DT_UINT64, DT_INT64, DT_BF16, DT_FLOAT32, DT_FLOAT8_E8M0}))
    .DYNAMIC_INPUT(offset, TensorType({DT_FLOAT32}))
    .DYNAMIC_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT8_E8M0}))
    .DYNAMIC_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(group_list, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(per_token_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT, DT_INT32}))
    .ATTR(split_item, Int, 0)
    .ATTR(dtype, Int, 0)
    .ATTR(transpose_weight, Bool, false)
    .ATTR(transpose_x, Bool, false)
    .ATTR(group_type, Int, -1)
    .ATTR(group_list_type, Int, 0)
    .ATTR(act_type, Int, 0)
    .ATTR(tuning_config, ListInt, {0})
    .OP_END_FACTORY_REG(GroupedMatmul)

/**
* @brief Function QuantGroupedMatmulInplaceAdd. This op fuse GroupedMatmul with InplaceAdd.

* @par Inputs:
* @li x1: A tensor. Format supports ND. Data type supports hifloat8, float8_e5m2, float8_e4m3fn.
* @li x2: A tensor. Format supports ND. Data type supports hifloat8, float8_e5m2, float8_e4m3fn.
* @li scale2: A tensor. Indicating scaling factor of quantization parameter, introduced by x2 quantization.
* Format supports ND. Data type supports float32, float8_e8m0.
* @li group_list: A tensor. Indicating the matmul size distribution along separated dimension.
* Format supports ND. Data type supports int64. The first dimension of group_list should not be greater than 1024, meaning it supports up to 1024 groups.
* @li y: A tensor. Format supports ND. Data type supports float32.
* @li scale1: A tensor. Indicating the scaling factor of the quantization parameter, introduced by x1 quantization.
* Format supports ND. The data type supports float32, float8_e8m0.

* @par Attributes:
* @li group_list_type: An int. Indicates whether the value in group_list is cumsum or count.
* 0, value in group_list is cumsum. 1, value in group_list is count. Default: 0.
* @li group_size: An int. Default: 0. Value only supports 0 for now.

* @par Outputs:
* y: A Tensor. Format supports ND.
* The data type supports float32.

*\n
*\n
* - The following are the supported quantization modes, data types and shapes(for Ascend 950 AI Processor):
*\n
| quantization         | x1 type                   | x2 type                   | scale1 type                    | scale2 type                    | x1 shape  | x2 shape            | y shape   | scale2 shape                  | scale1 shape          |
|----------------------|---------------------------|---------------------------|--------------------------------|--------------------------------|-----------|---------------------|-----------|-------------------------------|-----------------------|
| pertensor-perchannel | hifloat8                  | hifloat8                  | float32                        | float32                        | (K,M)     | (K,N)               | (B,M,N)   | (B,N)                         | (B,)/(B,1)            |
| mx                   | float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2 | float8_e8m0                    | float8_e8m0                    | (K,M)     | (K,N)               | (B,M,N)   | (K/64+B,N,2)                  | (K/64+B,M,2)          |

*\n
* @attention Constraints
* @li When group_list_type is 0, group_list must be a non-negative monotone non-decreasing array, when group_list_type is 1, it must be a non-negative array.
* @li Each axis for tensor x1 and tensor x2 should be less than or equal to 2147483647 (maximum of data type int32) after aligning to 32 byte.
* @li Input tensor x1 and x2 should have dim num 2, input tensor y should have dim num 3.
* @li The group_list must be passed, if group_list_type equals to 0, the last value in group_list must be no greater than the first dimension of tensor x1,
* if group_list_type equals to 1, sum of values in group_list must be no greater than the first dimension of tensor x1. \n
*/
  REG_OP(QuantGroupedMatmulInplaceAdd)
    .INPUT(x1, TensorType({DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .INPUT(x2, TensorType({DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .INPUT(scale2, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .INPUT(group_list, TensorType({DT_INT64}))
    .INPUT(y, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(scale1, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(group_list_type, Int, 0)
    .ATTR(group_size, Int, 0)
    .OP_END_FACTORY_REG(QuantGroupedMatmulInplaceAdd)

  /**
  * @brief Function TomeUnmerge. \n

  * @par Inputs:
  * @li attention: A Tensor List, attention out. Shape is (B, S, H). S = S2 + S1 - (S2 + S1) * top_rate
  * @li ori_index_a: A Tensor List of origin index A. Shape is (B, S1, H), Value range [0, S1 + S2), no dup and cant dup with ori_index_b.
  * @li ori_index_b: A Tensor List of origin index B. Shape is (B, S2, H), Value range [0, S1 + S2), no dup and cant dup with ori_index_a.
  * @li topk_indice: A Tensor List of topK indice. Shape is (B, S1, H), S1 must equal with ori_index_a.
  * @li arg_max: A Tensor List of ArgMax. Shape is (B, S1, H), S1 must equal with ori_index_a.

  * @par Attributes:
  * @li top_rate: A Float. rate to calculate how many rows of token_a merge to token_b. default is "0.5".

  * @par Outputs:
  * unzip_token: A Tensor List, restore by ori_index_a and ori_index_b.
  * @par Restrictions:
  * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
  */
  REG_OP(TomeUnmerge)
      .INPUT(attention, TensorType({DT_FLOAT16}))
      .INPUT(ori_index_a, TensorType({DT_INT64}))
      .INPUT(ori_index_b, TensorType({DT_INT64}))
      .INPUT(topk_indice, TensorType({DT_INT64}))
      .INPUT(arg_max, TensorType({DT_INT64}))
      .OUTPUT(unzip_token, TensorType({DT_FLOAT16}))
      .ATTR(top_rate, Float, 0.5)
      .OP_END_FACTORY_REG(TomeUnmerge)

/**
* @brief Function GroupedMatMulAllReduce. This op computes multi groups of matmuls on multi-cards environment.

* @par Inputs:
* @li x: A Tensor List, contains all left matrixs of inputs for matmuls. For each tensor, the data type of elements supports float16 or bfloat16; the format supports ND. The maximum length allowed is 64.
* 32B-aligned size of each dim should be smaller than 2147483647. The size of inner axis should be smaller than 65536.
* @li weight: A Tensor List of weight, contains all right matrixs of inputs for matmul. For each tensor, the data type of elements supports float16 or bfloat16; the format supports ND. The maximum length allowed is 64.
* 32B-aligned size of each dim should be smaller than 2147483647. The size of inner axis should be smaller than 65536.
* @li bias: A Tensor List of bias, contains all bias of inputs for matmul. For each tensor, the data type of elements supports float16 or float32; the format supports ND. The maximum length allowed is 64.
* @li group_list: a Tensor, indicates M-axis distributation of groups of matmuls for inputs and outputs.
* Data type of elements is int64. Format: ND. The maximum length allowed is 64.

* @par Attributes:
* @li splitItem: An int64, indicates whether do tensor split for inputs and outputs.
* 0: no split for inputs and outputs; 1: inputs need tensor split; 2: outputs need tensor split;
* 3: both inputs and outputs need tensor split. Default value is 0.
* @li group: A string. A required String identifying the group of ranks.
* @li reduceOp: A string. A required string identifying the reduction operation to
 perform. support "sum".
* @li commTurn: An int64. Number of communications with AICPU. Default: 0.

* @par Outputs:
* y: A Tensor List, contains all result of groups of matmuls. For each tensor,
* the data type of elements supports float16 or bfloat16; the format supports ND. The maximum length allowed is 64.

* @attention Constraints:
* Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
*/
  REG_OP(GroupedMatMulAllReduce)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(weight, TensorType({DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(group_list, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .ATTR(splitItem, Int, 0)
    .REQUIRED_ATTR(group, String)
    .ATTR(reduceOp, String, "sum")
    .ATTR(commTurn, Int, 0)
    .OP_END_FACTORY_REG(GroupedMatMulAllReduce)

/**
* @brief Fusion of alltoallv and grouped matmul.
* @par Inputs:
* @li gmm_x: A matrix Tensor of shape [BSK, H1]. The data type of elements supports float16, bfloat16 or hifloat8; the format supports ND.
* @li gmm_weight: A matrix Tensor of shape [e, H1, N1]. The data type of elements supports float16, bfloat16 or hifloat8 and should match that of gmm_x; the format supports ND.
* @li send_counts_tensor: A Tensor of shape [e * ep]. The data type of elements supports int32 or int64; the format supports ND.
* @li recv_counts_tensor: A Tensor of shape [e * ep]. The data type of elements supports int32 or int64; the format supports ND.
* @li mm_x: A matrix Tensor of shape [BS, H1]. The data type of elements supports float16, bfloat16 or hifloat8; the format supports ND.
* @li mm_weight: A matrix Tensor of shape [H2, N2]. The data type of elements supports float16, bfloat16 or hifloat8 and should match that of mm_x; the format supports ND.
* @li gmm_x_scale: A matrix Tensor. The type support float32. The format supports ND.
* @li gmm_weight_scale: A matrix Tensor. The type support float32. The format supports ND.
* @li gmm_x_offset: A matrix Tensor. The type support float32. The format supports ND.
* @li gmm_weight_offset: A matrix Tensor. The type support float32. The format supports ND.
* @li mm_x_scale: A matrix Tensor. The type support float32. The format supports ND.
* @li mm_weight_scale: A matrix Tensor. The type support float32. The format supports ND.
* @li mm_x_offset: A matrix Tensor. The type support float32. The format supports ND.
* @li mm_weight_offset: A matrix Tensor. The type support float32. The format supports ND.
*
* @par Attributes:
* @li group: A required String identifying the expert group of ranks.
* @li ep_world_size: A required int identifying the number of expert parallel group rank num.
* @li send_counts: An int list. A list containing amount of data to be sent.
* @li recv_counts: An int list. A list containing amount of data to be received.
* @li trans_gmm_weight: A boolean value. Whether gmm_weight is transposed. True indicates transposition. Default: false.
* @li trans_mm_weight: A boolean value. Whether mm_weight is transposed. True indicates transposition. Default: false.
* @li permute_out_flag: A boolean value. Whether to output permute_out. True indicates that output permute_out is required. Default: false.
* @li gmm_x_quant_mode: An int. Quantization mode of gmm_x. Default: 0.
*        - 0：No Quantization
*        - 1：PerTensor Quantization
*        - 2：PerChannel Quantization
*        - 3：PerToken Quantization
*        - 4：PerGroup Quantization
*        - 5：PerBlock Quantization
*        - 6：Mx Quant Quantization
* @li gmm_weight_quant_mode: An int. Quantization mode of gmm_weight. Default: 0.
* @li mm_x_quant_mode: An int. Quantization mode of mm_x. Default: 0.
* @li mm_weight_quant_mode: An int. Quantization mode of mm_weight. Default: 0.
* @li group_size: An int. Default: 0.
* @li y_dtype: An int. Declare the gmm_y dtype. Default: static_cast<int64_t>(ge::DT_UNDEFINED) is 28.
* @li mm_dtype: An int. Declare the mm_y dtype. Default: static_cast<int64_t>(ge::DT_UNDEFINED) is 28.
*
* @par Outputs:
* @li gmm_y: A matrix tensor of shape [A, N1] containing result of grouped matmul. The data type of elements supports float16 or bfloat16; the format supports ND.
* @li mm_y: A matrix tensor of shape [BS, N2] containing result of matmul. The data type of elements supports float16 or bfloat16; the format supports ND.
* @li permute_out: A matrix tensor of shape [BSK, H1] containing result of permutation if permute_out_flag == true. The data type of elements supports float16, bfloat16 or hifloat8; the format supports ND.
*/
REG_OP(AlltoAllvGroupedMatMul)
      .INPUT(gmm_x, TensorType({DT_FLOAT16, DT_BF16, DT_HIFLOAT8}))
      .INPUT(gmm_weight, TensorType({DT_FLOAT16, DT_BF16, DT_HIFLOAT8}))
      .OPTIONAL_INPUT(send_counts_tensor, TensorType({DT_INT32, DT_INT64}))
      .OPTIONAL_INPUT(recv_counts_tensor, TensorType({DT_INT32, DT_INT64}))
      .OPTIONAL_INPUT(mm_x, TensorType({DT_FLOAT16, DT_BF16, DT_HIFLOAT8}))
      .OPTIONAL_INPUT(mm_weight, TensorType({DT_FLOAT16, DT_BF16, DT_HIFLOAT8}))
      .OPTIONAL_INPUT(gmm_x_scale, TensorType({DT_FLOAT}))
      .OPTIONAL_INPUT(gmm_weight_scale, TensorType({DT_FLOAT}))
      .OPTIONAL_INPUT(gmm_x_offset, TensorType({DT_FLOAT}))
      .OPTIONAL_INPUT(gmm_weight_offset, TensorType({DT_FLOAT}))
      .OPTIONAL_INPUT(mm_x_scale, TensorType({DT_FLOAT}))
      .OPTIONAL_INPUT(mm_weight_scale, TensorType({DT_FLOAT}))
      .OPTIONAL_INPUT(mm_x_offset, TensorType({DT_FLOAT}))
      .OPTIONAL_INPUT(mm_weight_offset, TensorType({DT_FLOAT}))
      .OUTPUT(gmm_y, TensorType({DT_FLOAT16, DT_BF16}))
      .OUTPUT(mm_y, TensorType({DT_FLOAT16, DT_BF16}))
      .OUTPUT(permute_out, TensorType({DT_FLOAT16, DT_BF16, DT_HIFLOAT8}))
      .REQUIRED_ATTR(group, String)
      .REQUIRED_ATTR(ep_world_size, Int)
      .REQUIRED_ATTR(send_counts, ListInt)
      .REQUIRED_ATTR(recv_counts, ListInt)
      .ATTR(trans_gmm_weight, Bool, false)
      .ATTR(trans_mm_weight, Bool, false)
      .ATTR(permute_out_flag, Bool, false)
      .ATTR(gmm_x_quant_mode, Int, 0)
      .ATTR(gmm_weight_quant_mode, Int, 0)
      .ATTR(mm_x_quant_mode, Int, 0)
      .ATTR(mm_weight_quant_mode, Int, 0)
      .ATTR(group_size, Int, 0)
      .ATTR(y_dtype, Int, 28)
      .ATTR(mm_dtype, Int, 28)
      .OP_END_FACTORY_REG(AlltoAllvGroupedMatMul)

/**
* @brief Fusion of grouped matmul and alltoallv.

* @par Inputs:
* @li gmm_x: A matrix tensor of shape [A, H1]. The data type of elements supports float16 or bfloat16; the format supports ND.
* @li gmm_weight: A matrix tensor of shape [e, H1, N1]. The data type of elements supports float16 or bfloat16 and should match that of gmm_x; the format supports ND.
* @li send_counts_tensor: A tensor of shape [e * ep]. The data type of elements supports int32 or int64; the format supports ND.
* @li recv_counts_tensor: A tensor of shape [e * ep]. The data type of elements supports int32 or int64; the format supports ND.
* Data type of elements is int64. Format: ND.
* @li mm_x: A matrix tensor of shape [BS, H2]. The data type of elements supports float16 or bfloat16; the format supports ND.
* @li mm_weight: gmm_weight: A matrix tensor of shape [H2, N2]. The data type of elements supports float16 or bfloat16 and should match that of mm_x; the format supports ND.

* @par Attributes:
* @li group: A required String identifying the expert group of ranks
* @li ep_world_size: A required int identifying the number of expert parallel group rank num.
* @li send_counts: An int list. A list containing amount of data to be sent.
* @li recv_counts: An int list. A list containing amount of data to be received.
* @li trans_gmm_weight: A boolean value. Indicating whether gmm_weight is transposed.
* @li trans_mm_weight: A boolean value. Indicating whether mm_weight is transposed.

* @par Outputs:
* @li y: A matrix tensor of shape [BSK, N1] containing result of grouped matmul. The data type of elements supports float16 or bfloat16; the format supports ND.
* @li mm_y_optional: A matrix tensor of shape [BS, N2] containing result of matmul. The data type of elements supports float16 or bfloat16; the format supports ND.
*/
REG_OP(GroupedMatMulAlltoAllv)
      .INPUT(gmm_x, TensorType({DT_FLOAT16, DT_BF16}))
      .INPUT(gmm_weight, TensorType({DT_FLOAT16, DT_BF16}))
      .OPTIONAL_INPUT(send_counts_tensor, TensorType({DT_INT32, DT_INT64}))
      .OPTIONAL_INPUT(recv_counts_tensor, TensorType({DT_INT32, DT_INT64}))
      .OPTIONAL_INPUT(mm_x, TensorType({DT_FLOAT16, DT_BF16}))
      .OPTIONAL_INPUT(mm_weight, TensorType({DT_FLOAT16, DT_BF16}))
      .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
      .OUTPUT(mm_y, TensorType({DT_FLOAT16, DT_BF16}))
      .REQUIRED_ATTR(group, String)
      .REQUIRED_ATTR(ep_world_size, Int)
      .REQUIRED_ATTR(send_counts, ListInt)
      .REQUIRED_ATTR(recv_counts, ListInt)
      .ATTR(trans_gmm_weight, Bool, false)
      .ATTR(trans_mm_weight, Bool, false)
      .OP_END_FACTORY_REG(GroupedMatMulAlltoAllv)

  /**
   * @brief compute init routing grad for moe input.
   * @par Inputs:
   * @li grad_expanded_x: A 2D or 3D Tensor, tokens squences. Type is:BFloat16, Float16 or Float32. Format support ND.
   * @li expanded_row_idx: A 1D Tensor, token indices in grad_expanded_x. Type is:Int32. Format support ND.
   * @par Outputs:
   * @li grad_x: A 2D Tensor, reverse gradient result. Type is:BFloat16, Float16 or Float32，which is same as that of
   *             grad_expanded_x. axis 0 of grad_x should be same as the value that axis 0 of expanded_row_idx divide
   *             top_k, axis 1 of grad_x should be same as -1 axis of grad_expanded_x. Format support ND.
   * @par Attributes:
   * @li top_k: Required parameter. Type is:Int32. The value must be greater than 0 and can be exactly divided by axis 0
   *            of expanded_row_idx.
   * @li drop_pad_mode: Optional parameter, identify the dropless or drop/pad scenario. Type is:Int32. The value is
   *                    0 (dropless scenario) or 1 (drop/pad scenario).
   * @li active_num: Optional parameter, identify activate scenario. Type is:Int32. The value 0 indicates a non-active
   *                 scenario, and a value greater than 0 indicates an active scenario. In the active scenario, the size
   *                 of axis 0 of grad_expanded_x must be equal to the value of active_num.
   */
    REG_OP(MoeInitRoutingV2Grad)
    .INPUT(grad_expanded_x, "T1")
    .INPUT(expanded_row_idx, "T2")
    .OUTPUT(grad_x, "T1")
    .DATATYPE(T1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T2, TensorType({DT_INT32}))
    .REQUIRED_ATTR(top_k, Int)
    .ATTR(drop_pad_mode, Int, 0)
    .ATTR(active_num, Int, 0)
    .OP_END_FACTORY_REG(MoeInitRoutingV2Grad)

  /**
   * @brief compute init routing for moe input.
   * @par Inputs:
   * @li x: A 2D Tensor. represents the input token. shape is (num_rows, H), num_rows is the number of tokens, H is the
   *        length of each token. Type is:BFloat16, Float16 or Float32. Format support ND.
   * @li expert_idx: A 2D Tensor. represents the id of top k experts for each tokens. shape is (num_row, K),
   *                 it can be (num_row, ) in Ascend 950 AI Processor.
   *                 When drop_pad_mode is 1, or drop_pad_mode is 0 but expert_tokens_count_or_cumsum_flag is not 0,
   *                 The range of value is [0, expert_num). Other scenario the value of expert_idx cannot be less than
   *                 0. Type is:Int32 or Int64, Int64 only support in Ascend 950 AI Processor. Format support ND.
   * @par Outputs:
   * @li expanded_x: A 2D or 3D Tensor. Type is:BFloat16, Float16 or Float32.
   *                 The data type must be the same as that of x. Format support ND.
   *                 dropless scenario:
   *                 the first dim is val=min(num_rows \* K, active_num), so shape is (val, H)
   *                 dropPad scenario:
   *                 shape is (expert_num, expert_capacity, H)
   * @li expanded_row_idx: A 1D Tensor. Type is:Int32. shape is (num_rows \* K,). Format support ND.
   * @li expert_tokens_count_or_cumsum: A 1D Tensor. represents the number of tokens processed by each expert and the
   * cumulative value. The value is controlled by expert_tokens_count_or_cumsum_flag to output. Type is:Int32. shape
   * is (expert_num,). Format support ND.
   * @li expert_tokens_before_capacity: A 1D Tensor. represents the number of tokens processed by each expert before
   * drop. The value is controlled by expert_tokens_before_capacity_flag to output. Type is:Int32. shape is
   * (expert_num,). Format support ND.
   * @par Attributes:
   * @li active_num: Optional parameter. Type is:Int32. identify activate scenario. The value 0 indicates a non-active
   *                 scenario, and a value greater than 0 indicates an active scenario. In the active scenario, the size
   *                 of axis 0 of grad_expanded_x must be equal to the value of active_num. Default: 0.
   * @li expert_capacity: Optional parameter. Type is:Int32. The max tokens count of every expert. Default: 0.
   * @li expert_num: Optional parameter. Type is:Int32. Default: 0.
   * @li drop_pad_mode: Optional parameter. Type is:Int32. The value is 0(dropless) or 1(dropPad). Default: 0.
   * @li expert_tokens_count_or_cumsum_flag: Optional parameter. Type is:Int32. The value is 0 (no token count),
   *                                         1(compute token count) or 2(compute token cumsum), which in dropless
   *                                         scenario. Default: 0.
   * @li expert_tokens_before_capacity_flag: Optional parameter. Type is:Bool. The value is true (no tokens count) or
   *                                         1(compute token count), which in dropPad scenario. Default: false.
   */
    REG_OP(MoeInitRoutingV2)
    .INPUT(x, "T1")
    .INPUT(expert_idx, "T3")
    .OUTPUT(expanded_x, "T1")
    .OUTPUT(expanded_row_idx, "T2")
    .OUTPUT(expert_tokens_count_or_cumsum, "T2")
    .OUTPUT(expert_tokens_before_capacity, "T2")
    .DATATYPE(T1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T2, TensorType({DT_INT32}))
    .DATATYPE(T3, TensorType({DT_INT32, DT_INT64}))
    .ATTR(active_num, Int, 0)
    .ATTR(expert_capacity, Int, 0)
    .ATTR(expert_num, Int, 0)
    .ATTR(drop_pad_mode, Int, 0)
    .ATTR(expert_tokens_count_or_cumsum_flag, Int, 0)
    .ATTR(expert_tokens_before_capacity_flag, Bool, false)
    .OP_END_FACTORY_REG(MoeInitRoutingV2)

  /**
   * @brief compute init routing quant for moe input.
   * @par Inputs:
   * @li x: A 2D Tensor. Type is:BFloat16, Float16 or Float32. Format support ND.
   * @li row_idx: A 2D Tensor: A Tensor. Type is:Int32. Format support ND.
   * @li expert_idx: A 2D Tensor. Type is:Int32. Format support ND.
   * @par Outputs:
   * @li expanded_x: A 2D Tensor. Type is:Int8. The data type must be the same as that of x.
   *                 The first dim must be the first dim of row_idx multiply the second dim of row_idx.
   *                 The second dim must be the second dim of x. Format support ND.
   * @li expanded_row_idx: A 1D Tensor. Type is:Int32. The dim must be  the first dim of row_idx multiply the second
   *                       dim of row_idx. Format support ND.
   * @li expanded_expert_idx: A 1D Tensor. Type is:Int32. The Shape is same as expanded_row_idx. Format support ND.
   * @par Attributes:
   * @li active_num: Required parameter. Type is:Int32. The value 0 indicates a non-active
   *                 scenario, and a value greater than 0 indicates an active scenario. In the active scenario, the
   *                 size of axis 0 of expanded_x must be equal to the value of active_num.
   * @li scale: Required parameter. Type is:Float.
   * @li offset: Required parameter. Type is:Float.
   */
    REG_OP(MoeInitRoutingQuant)
    .INPUT(x, "T1")
    .INPUT(row_idx, "T2")
    .INPUT(expert_idx, "T2")
    .OUTPUT(expanded_x, "T3")
    .OUTPUT(expanded_row_idx, "T2")
    .OUTPUT(expanded_expert_idx, "T2")
    .DATATYPE(T1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T2, TensorType({DT_INT32}))
    .DATATYPE(T3, TensorType({DT_INT8}))
    .REQUIRED_ATTR(active_num, Int)
    .REQUIRED_ATTR(scale, Float)
    .REQUIRED_ATTR(offset, Float)
    .OP_END_FACTORY_REG(MoeInitRoutingQuant)

  /**
   * @brief compute init routing quant for moe input.
   * @par Inputs:
   * @li x: A 2D Tensor. represents the input token. shape is (num_rows, H), num_rows is the number of tokens, H is the
   *        length of each token. Type is:BFloat16, Float16 or Float32. Format support ND.
   * @li expert_idx: A 2D Tensor. represents the id of top k experts for each tokens. shape is (num_row, K).
   *                 When drop_pad_mode is 1, or drop_pad_mode is 0 but expert_tokens_count_or_cumsum_flag is not 0,
   *                 The range of value is [0, expert_num). Other scenario the value of expert_idx cannot be less than
   *                 0. Type is:Int32. Format support ND.
   * @li scale: An Optional Tensor. represents the parameters used to compute the result of the quant, which is required
   *            to be a 1D shape (1,) for the static quant scenario. The dynamic quant scene if not input, indicates
   *            that scale is not used in the calculation process; if a 2D Tensor is input, the shape is required to be
   *            (expertNum, H) or (1, H). Type is:Float32. Format support ND.
   * @li offset: An Optional Tensor. Indicates the offset value used to compute the result, it`s only used in the static
   *             quant scenario, which is required to be a 1D shape (1,). Type is:Float32. Format support ND.
   * @par Outputs:
   * @li expanded_x: A 2D or 3D Tensor. Type is:Int8. Format support ND.
   *                 dropless scenario:
   *                 the first dim is val=min(num_rows * K, active_num), so shape is (val, H)
   *                 dropPad scenario:
   *                 shape is (expert_num, expert_capacity, H)
   * @li expanded_row_idx:A 1D Tensor. Type is:Int32. shape is (num_rows * K,). Format support ND.
   * @li expert_tokens_count_or_cumsum: A 1D Tensor. represents the number of tokens processed by each expert and the
   * cumulative value. The value is controlled by expert_tokens_count_or_cumsum_flag to output. Type is:Int32. shape
   * is (expert_num,). Format support ND.
   * @li expert_tokens_before_capacity: A 1D Tensor. represents the number of tokens processed by each expert before
   * drop. The value is controlled by expert_tokens_before_capacity_flag to output. Type is:Int32. shape is
   * (expert_num,). Format support ND.
   * @li dynamic_quant_scale: A 1D Tensor. represents tthe outputs the intermediate value of the dynamic quant
   *                          computation process, which is only output in dynamic quant scenarios.Type is:Float32.
   *                          The shape is expanded_x_shape[:-1]. Format support ND.
   * @par Attributes:
   * @li active_num: Optional parameter. Type is:Int32. identify activate scenario. The value 0 indicates a non-active
   *                 scenario, and a value greater than 0 indicates an active scenario. In the active scenario, the size
   *                 of axis 0 of grad_expanded_x must be equal to the value of active_num. Default: 0.
   * @li expert_capacity: Optional parameter. Type is:Int32. The max tokens count of every expert. Default: 0.
   * @li expert_num: Optional parameter. Type is:Int32. Default: 0.
   * @li drop_pad_mode: Optional parameter. Type is:Int32. The value is 0(dropless) or 1(dropPad). Default: 0.
   * @li expert_tokens_count_or_cumsum_flag: Optional parameter. Type is:Int32. The value is 0 (no token count),
   *                                         1(compute token count) or 2(compute token cumsum), which in dropless
   *                                         scenario. Default: 0.
   * @li expert_tokens_before_capacity_flag: Optional parameter. Type is:Bool. The value is true (no tokens count) or
   *                                         1(compute token count), which in dropPad scenario. Default: false.
   * @li quant_mode: Optional parameter. Type is:Int32. The value is 0(static quant) or 1(dynamic quant). Default: 0.
   */
    REG_OP(MoeInitRoutingQuantV2)
    .INPUT(x, "T1")
    .INPUT(expert_idx, "T2")
    .OPTIONAL_INPUT(scale, "T3")
    .OPTIONAL_INPUT(offset, "T3")
    .OUTPUT(expanded_x, "T4")
    .OUTPUT(expanded_row_idx, "T2")
    .OUTPUT(expert_tokens_count_or_cumsum, "T2")
    .OUTPUT(expert_tokens_before_capacity, "T2")
    .OUTPUT(dynamic_quant_scale, "T3")
    .DATATYPE(T1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T2, TensorType({DT_INT32}))
    .DATATYPE(T3, TensorType({DT_FLOAT}))
    .DATATYPE(T4, TensorType({DT_INT8}))
    .ATTR(active_num, Int, 0)
    .ATTR(expert_capacity, Int, 0)
    .ATTR(expert_num, Int, 0)
    .ATTR(drop_pad_mode, Int, 0)
    .ATTR(expert_tokens_count_or_cumsum_flag, Int, 0)
    .ATTR(expert_tokens_before_capacity_flag, Bool, false)
    .ATTR(quant_mode, Int, 0)
    .OP_END_FACTORY_REG(MoeInitRoutingQuantV2)

/**
* @brief In MoE computation, the final step involves processing and merging the output results of the MoE FNN.
* @par Inputs:
* @li expanded_x: A 2D Tensor. Type is:BFloat16, Float16 or Float32. Shape support(NUM\_ROWS \* K, H).
* @li x1: A 2D Tensor. Type is:BFloat16, Float16 or Float32. The data type requirement of A is consistent
      with expandedX,and the shape requirements are consistent with the shape of out.
* @li x2: An optional 2D Tensor. Type is:BFloat16, Float16 or Float32. The data type requirement of A is consistent
      with expandedX,and the shape requirements are consistent with the shape of out. If the parameter A is not entered,
      the parameter B can also not be entered.
* @li bias: A 2D Tensor. Type is:BFloat16, Float16 or Float32.The data type requirement of A is consistent
      with expandedX.Shape support(E, H). E is the total number of experts, and H is the number of columns.
* @li scales: A 2D Tensor. Type is:BFloat16, Float16 or Float32. The data type requirement of A is consistent
      with expandedX.Shape support(NUM\_ROWS, K).
* @li expanded_row_idx: A 1D Tensor. Type is:Int32.Shape support(NUM\_ROWS \* K).Values in Tensor are
      [0,NUM\_ROWS \* K-1].
* @li expanded_expert_idx: A 2D Tensor. Type is Int32. Shape support(NUM\_ROWS, K).
      Values in Tensor are [0, E-1].
* @par Outputs:
* @li y: A 2D Tensor. Type is:BFloat16, Float16 or Float32. Shape support(NUM\_ROWS, H).
*/
REG_OP(MoeFinalizeRouting)
    .INPUT(expanded_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(scales, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(expanded_row_idx, TensorType({DT_INT32}))
    .INPUT(expanded_expert_idx, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(MoeFinalizeRouting)

/**
* @brief In MoE computation, the final step involves processing and merging the output results of the MoE FNN.
* @par Inputs:
* @li expanded_x: expandedX in the formula, represents the token sequences. A 2D or 3D Tensor. Type is:BFloat16, Float16 or Float32. Format support
      ND. Dropless scenario shape is (NUM\_ROWS \* K, H), dropPad scenario shape is (expert_num, expert_capacity, H).
* @li expanded_row_idx: A 1D Tensor, represents the token indexes of expanded_x. Type is:Int32. Shape support(NUM\_ROWS \* K).Values in Tensor are [0, NUM\_ROWS \* K – 1]
      when drop_pad_mode is 0,2; Values in Tensor are [-1, NUM\_ROWS \* K – 1] when drop_pad_mode is 1, 3.
* @li x1: An optional 2D Tensor. Type is:BFloat16, Float16 or Float32.The data type requirement of A is consistent
      with expandedX,and the shape requirements are consistent with the shape of out.
* @li x2: An optional 2D Tensor. Type is:BFloat16, Float16 or Float32.The data type requirement of A is consistent
      with expandedX,and the shape requirements are consistent with the shape of out.If the parameter A is not entered,
      the parameter B can also not be entered.
* @li bias: An optional 2D Tensor, represents the bias of expanded_x. Type is:BFloat16, Float16 or Float32.The data type requirement of A is consistent
      with expandedX.Shape support(E, H). E is the total number of experts, and H is the number of columns.
* @li scales: An optional 2D Tensor, represents the scale of expanded_x. Type is:BFloat16, Float16 or Float32. The data type of scales is required to be consistent
      with expandedX in non-mixed precision mode, except in Ascend 950 AI Processor, and supports Float32 for scales when expandedX is BFloat16 in mixed precision mode. Shape support(NUM\_ROWS, K), When scales is null, K is 1.
* @li expert_idx: An optional 2D Tensor, represents the indexes of bias. Type is Int32.Shape support(NUM\_ROWS, K).Values in Tensor are [0, E-1], if bias exists, expert_idx must exist.
* @par Outputs:
* @li y: A 2D Tensor. Type is:BFloat16, Float16 or Float32.Shape support(NUM\_ROWS, H).
* @par Attributes:
* @li drop_pad_mode: drop mode. Type is Int32. Default: 0, range [0,3].
      0 (dropless scenario, expanded_row_idx column arrangement), 1 (drop or pad scenario, expanded_row_idx column arrangement),
      2 (dropless scenario, expanded_row_idx line arrangement), 3 (drop or pad scenario, expanded_row_idx line arrangement).
*/
REG_OP(MoeFinalizeRoutingV2)
    .INPUT(expanded_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(expanded_row_idx, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(scales, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(expert_idx, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(drop_pad_mode, Int, 0)
    .OP_END_FACTORY_REG(MoeFinalizeRoutingV2)

/**
* @brief In Multi-Head Attention(MHA) computation, bias and rescale qkv tensor.
* @par Inputs:
* @li qkv:  A 3D tensor. Type is:BFloat16, Float16 or Float32. Format support
      ND. shape is (batch, token, 3 * num_heads * dim_per_head).
* @li qkv_bias: A 1D tensor. Type is:BFloat16, Float16 or Float32. Format support
      ND. shape is (3 * num_heads * dim_per_head).
* @par Outputs:
* @li q: A 4D tensor.Type is:BFloat16, Float16 or Float32. Format support
      ND. shape is (batch, num_heads,token, dim_per_head).
* @li k: A 4D tensor.Type is:BFloat16, Float16 or Float32. Format support
      ND. shape is (batch, num_heads,token, dim_per_head).
* @li v: A 4D tensor.Type is:BFloat16, Float16 or Float32. Format support
      ND. shape is (batch, num_heads,token, dim_per_head).
* @par Attributes:
      num_heads: head nums. Type is Int64.
*/
REG_OP(TransformBiasRescaleQkv)
.INPUT(qkv, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
.INPUT(qkv_bias, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
.OUTPUT(q, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
.OUTPUT(k, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
.OUTPUT(v, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
.REQUIRED_ATTR (num_heads, Int)
.OP_END_FACTORY_REG(TransformBiasRescaleQkv)


  /**
   * @brief compute init routing for moe input.
   * @par Inputs:
   * @li x: A 2D Tensor. Type is:BFloat16, Float16 or Float32. Format support ND.
   * @li row_idx: A 2D Tensor: A Tensor. Type is:Int32. Format support ND.
   * @li expert_idx: A 2D Tensor. Type is:Int32. Format support ND.
   * @par Outputs:
   * @li expanded_x: A 2D Tensor. Type is:BFloat16, Float16 or Float32. The data type must be the same as that of x.
   *                 The first dim must be the first dim of row_idx multiply the second dim of row_idx or active_num.
   *                 The second dim must be the second dim of x. Format support ND.
   * @li expanded_row_idx: A 1D Tensor. Type is:Int32. The dim must be  the first dim of row_idx multiply the second
   *                       dim of row_idx. Format support ND.
   * @li expanded_expert_idx: A 1D Tensor. Type is:Int32. The Shape is same as expanded_row_idx. Format support ND.
   * @par Attributes:
   * @li active_num: Required parameter. Type is:Int32. The value 0 indicates a non-active
   *                 scenario, and a value greater than 0 indicates an active scenario. In the active scenario, the size
   *                 of axis 0 of expanded_x must be equal to the value of active_num.
   */
    REG_OP(MoeInitRouting)
    .INPUT(x, "T1")
    .INPUT(row_idx, "T2")
    .INPUT(expert_idx, "T2")
    .OUTPUT(expanded_x, "T1")
    .OUTPUT(expanded_row_idx, "T2")
    .OUTPUT(expanded_expert_idx, "T2")
    .DATATYPE(T1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T2, TensorType({DT_INT32}))
    .REQUIRED_ATTR(active_num, Int)
    .OP_END_FACTORY_REG(MoeInitRouting)

    /**
       * @brief compute init routing for moe.
       * @par Inputs:
       * @li x: A 2D tensor. Shape is: (B*S, H). Type is:Int8, BFloat16, Float16 or Float32. Format support ND.
       * @li expert_idx: A 2D tensor. Shape is: (B*S, K). Type is:Int32. Expert index. Format support ND.
       * @li scale: A 1D or 2D tensor. Shape is: (B*S) or (B*S, H). Type is:Float32. Format support ND.
       * @li offset: A 2D tensor. Shape is: (expert_end - expert_start, 1) or (expert_end - expert_start, H).
                     Type is:Float32. Format support ND.
       * @par Outputs:
       * @li expanded_x: A 2D tensor. Shape is: (B*S*K, H). Type is:Int8, BFloat16, Float16 or Float32.
                         The data type must be the same as that of x. Format support ND.
       * @li expanded_row_idx: A 1D tensor. Shape is: (B*S*K). Type is:Int32. Format support ND.
       * @li expert_tokens_count_or_cumsum: A 1D tensor. represents the number of tokens processed by each expert and the
                                            cumulative value. The value is controlled by expert_tokens_num_flag to output.
                                            Type is:Int64. shape is (expert_end - expert_start, ). Format support ND.
       * @li expanded_scale: A 1D tensor. Shape is: (B*S*K). Type is:Float32.
                             The data type must be the same as that of scale. Format support ND.
       * @par Attributes:
       * @li active_num: Optional parameter. Type is:Int32. identify activate scenario. The value 0 indicates a non-active
       *                 scenario, and a value greater than 0 indicates an active scenario. In the active scenario, the size
       *                 of axis 0 of grad_expanded_x must be equal to the value of active_num. Default: -1.
       * @li expert_capacity: Optional parameter. Type is:Int32. The max tokens count of every expert. Default: -1.
       * @li expert_num: Optional parameter. Type is:Int32. Number of experts. Default: -1.
       * @li drop_pad_mode: Optional parameter. Type is:Int32. The value is 0(dropless) or 1(dropPad). Default: 0.
       * @li expert_tokens_num_type: Optional parameter. Type is:Int32. The value is 0(compute tokens cumsum) or
                                     1(compute tokens count), which in dropPad scenario. Default: 0.
       * @li expert_tokens_num_flag: Optional parameter. Type is:Bool. The value is true (compute tokens) or
                                     false(do not compute tokens), which in dropPad scenario. Default: false.
       * @li quant_mode: Optional parameter. Type is:Int. The value is -1(no quant) or 0(static quant) or 1(dynamic quant). Default: -1.
       * @li active_expert_range: Optional parameter. Type is:ListInt. Like [expert_start, expert_end].
                                  expert_start must be greater than or equal to 0, expert_end must be less than or equal to 10240,
                                  expert_start must be less than expert_end. Default: [].
       * @li row_idx_type: Optional parameter. Type is:Int. The value is 0(gather) or 1(scatter). Default: 0.
       */
    REG_OP(MoeInitRoutingV3)
      .INPUT(x, TensorType({DT_INT8, DT_FLOAT16, DT_FLOAT, DT_BF16}))
      .INPUT(expert_idx, TensorType({DT_INT32}))
      .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT}))
      .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
      .OUTPUT(expanded_x, TensorType({DT_INT8, DT_FLOAT16, DT_FLOAT, DT_BF16}))
      .OUTPUT(expanded_row_idx, TensorType({DT_INT32}))
      .OUTPUT(expert_tokens_count_or_cumsum, TensorType({DT_INT64}))
      .OUTPUT(expanded_scale, TensorType({DT_FLOAT}))
      .ATTR(active_num, Int, -1)
      .ATTR(expert_capacity, Int, -1)
      .ATTR(expert_num, Int, -1)
      .ATTR(drop_pad_mode, Int, 0)
      .ATTR(expert_tokens_num_type, Int, 0)
      .ATTR(expert_tokens_num_flag, Bool, false)
      .ATTR(quant_mode, Int, -1)
      .ATTR(active_expert_range, ListInt, {})
      .ATTR(row_idx_type, Int, 0)
      .OP_END_FACTORY_REG(MoeInitRoutingV3)

    /**
       * @brief Compute renorm(sigmoid) and topk for moe input.
       *
       * @par Inputs:
       * @li x: A 2D tensor which moe gating topk is applied, The shape is: (B*S, E), format supports ND, and data type must be float16, float or bfloat16. E(Expert num) can not be greater than 2048. E(Expert num) should be divisible by group_count.
       * @li bias: A 1D tensor which is "bias" in moe gating topk. The shape is: (E), format supports ND, and data type must be the same as that of x.
       *
       * @par Outputs:
       * @li y: A 2D tensor which is the topk value result of moe gating topk, format supports ND, and data type must be the same as that of x.
             The size of the non-1 axis must be the same as that of the corresponding axis of x.
             The size of the -1 axis must be the same as that of k.
       * @li expert_idx: A 2D tensor which is the topk index result of moe gating topk, format supports ND, and data type must be int. The shape must be the same as that of y.
       * @li out: A 2D tensor which is the renorm result of moe gating topk, format supports ND, and data type must be float. The shape must be the same as that of x.
       *
       * @par Attributes:
       * @li k: A required attribute of type int. The value must greater than 0 and less than or equal to expert_num / group_count * k_group, idicating the topk value.
       * @li k_group: An optional attribute of type int. It can not be less than 1, and can not be greater than group_count, indicating the topk group value. The default value is 1.
       * @li group_count: An optional attribute of type int. It can not be less than 1, indicating the group count. The group_count * align_32(expert_num / group_count) can not be greater than 2048. The default value is 1.
       * @li group_select_mode: An optional attribute of type int. 0 indicating that sort group by max values, 1 indicating that sort group by sum of top-2 values. The default value is 0.
       * @li renorm: An optional attribute of type int. It can only be 0 now, indicating that norm firstly and then topk. The default value is 0.
       * @li norm_type: An optional attribute of type int. 0 indicating that the softmax function is used, 1 indicating that the sigmoid function is used. The default value is 0.
       * @li out_flag: An optional attribute of type bool. true indicating that has renorm output, false indicating that does not have renorm output. The default value is false.
       * @li routed_scaling_factor: An optional attribute of type float, indicating the routed_scaling_factor coefficient in use. The default value is 1.0.
       * @li eps: An optional attribute of type float, indicating the eps coefficient in use. The default value is 1e-20.
       */
    REG_OP(MoeGatingTopK)
      .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
      .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
      .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
      .OUTPUT(expert_idx, TensorType({DT_INT32}))
      .OUTPUT(out, TensorType({DT_FLOAT}))
      .REQUIRED_ATTR(k, Int)
      .ATTR(k_group, Int, 1)
      .ATTR(group_count, Int, 1)
      .ATTR(group_select_mode, Int, 0)
      .ATTR(renorm, Int, 0)
      .ATTR(norm_type, Int, 0)
      .ATTR(out_flag, Bool, false)
      .ATTR(routed_scaling_factor, Float, 1.0)
      .ATTR(eps, Float, 1e-20)
      .OP_END_FACTORY_REG(MoeGatingTopK)

  /**
   * @brief compute softmax and topk for moe input.
   * @par Inputs:
   * @li x: A 2D or 3D Tensor. Type is:BFloat16, Float16 or Float32. Format support ND.
   * @li finished: A Tensor. Type is:Bool. Shape is x_shape[:-1]. Format support ND.
   * @par Outputs:
   * @li y: A Tensor. Type is:BFloat16, Float16 or Float32. The data type must be the same as that of x.
         The size of the non-1 axis must be the same as that of the corresponding axis of x.
         The size of the -1 axis must be the same as that of k. Format support ND.
   * @li expert_idx: A Tensor. Type is:Int32. The shape must be the same as that of y. Format support ND.
   * @li row_idx: A Tensor. Type is:Int32. The shape must be the same as that of y. Format support ND.
   * @par Attributes:
   * @li k: Required parameter. Type is:Int32. The value must greater than 0 and less than or equal to the size
         of the -1 axis of x, and k must not greater than 1024.
   */
    REG_OP(MoeGatingTopKSoftmax)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(finished, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(expert_idx, TensorType({DT_INT32}))
    .OUTPUT(row_idx, TensorType({DT_INT32}))
    .REQUIRED_ATTR(k, Int)
    .OP_END_FACTORY_REG(MoeGatingTopKSoftmax)

  /**
   * @brief compute softmax and topk for moe input.
   * @par Inputs:
   * @li x: A 2D or 3D tensor. Type is:BFloat16, Float16 or Float32. Format support ND.
   * @li finished: An optional tensor. Type is:Bool. Shape is x_shape[:-1]. Format support ND.
   * @par Outputs:
   * @li y: A tensor. Type is:BFloat16, Float16 or Float32. The data type must be the same as that of x.
         The size of the non-1 axis must be the same as that of the corresponding axis of x.
         The size of the -1 axis must be the same as that of k. Format support ND.
   * @li expert_idx: A tensor. Type is:Int32. The shape must be the same as that of y. Format support ND.
   * @li softmax_result: A tensor. Type is:Float32. The shape must be the same as that of x. Format support ND.
   * @par Attributes:
   * @li k: Required parameter. Type is:Int32. The value must greater than 0 and less than or equal to the size
         of the -1 axis of x, and k must not greater than 1024.
   * @li renorm: Optional parameter. Type is:Int32. The value must be 0(non-renorm) or 1(renorm)
   * @li output_softmax_result_flag: Optional parameter. Type is:Bool. The value must be true or false.
         When renorm is 0, output_softmax_result_flag is true indicates that the Softmax result is output.
         When renorm is 0, output_softmax_result_flag is false indicates that the Softmax result is not output.
         When renorm is 1, this parameter does not take effect, the Softmax result is not output.
   */
    REG_OP(MoeGatingTopKSoftmaxV2)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(finished, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(expert_idx, TensorType({DT_INT32}))
    .OUTPUT(softmax_result, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(k, Int)
    .ATTR(renorm, Int, 0)
    .ATTR(output_softmax_result_flag, Bool, false)
    .OP_END_FACTORY_REG(MoeGatingTopKSoftmaxV2)

/**
* @brief Backwards calculation of MoeFinalizeRoutingV2.

* @par Inputs:
* @li grad_y: A 2D Tensor, represents the gradient of output of MoeFinalizeRoutingV2. Type is BFloat16, Float16 or
      Float32. Shape supports (R, H). Format supports ND.
* @li expanded_row_idx: A 1D Tensor, represents the token indexes of expanded_x. Type is Int32. Shape supports (R * K),
      when scales is not passed in, K must be 1. If drop_pad_mode is 0, the value range is [0, R * K - 1], and there are
      no duplicate indexes. When drop_pad_mode is 1, the value range is [-1, expert_num * expert_capacity - 1], and
      duplicate indexes are not allowed except -1. Format supports ND.
* @li expanded_x: An optional 2D or 3D Tensor, represents the token sequences. Type should be the same as the type of
      grad_y. When scales is passed in, it should be passed in. When drop_pad_mode is 0, it should be a 2D Tensor, and
      when active_num is between (0, R * K), the shape is (active_num, H), otherwise the shape is (R * K, H). When
      drop_pad_mode is 1, it should be a 3D Tensor, the shape is (expert_num, expert_capacity, H). Format supports ND.
* @li scales: An optional 2D Tensor, represents the scale of expanded_x. Type should be the same as the type of grad_y
      except in Ascend 950 AI Processor. Shape supports (R, K). Format supports ND.
* @li expert_idx: An optional 2D Tensor, represents the indexes of bias. Type should be the same as the type of
      expanded_row_idx. When bias is passed in, it should be passed in. Shape supports (R, K). the value range is
      [0, E - 1], E >= 1, and duplicate indexes are allowed. Format supports ND.
* @li bias: An optional 2D Tensor, represents the bias of expanded_x. Type should be the same as the type of grad_y.
      Shape supports (E, H). Format supports ND.

* @par Outputs:
* @li grad_expanded_x: A 2D or 3D Tensor, represents the gradient of expanded_x. Type should be the same as the type of
      grad_y. When drop_pad_mode is 0, it should be a 2D Tensor, when active_num is between (0, R * K), the shape is
      (active_num, H), otherwise the shape is (R * K, H). When drop_pad_mode is 1, it should be a 3D Tensor, the shape is
      (expert_num, expert_capacity, H). Format supports ND.
* @li grad_scales: A 2D Tensor, represents the gradient of scales. Type should be the same as the type of grad_y except
      in Ascend 950 AI Processor. Shape supports (R, K). This output only makes sense when scales is passed in.
      Format supports ND.

* @par Attributes:
* @li drop_pad_mode: An optional integer, represents the dropless or drop/pad mode. Type is Int32. Default: 0. Value
      supports 0 or 1.
* @li active_num: An optional integer, represents the active tokens of expanded_x. Type is Int32. Default: 0. When
      drop_pad_mode is 0, it takes effect only when it is between (0, R * K). When drop_pad_mode is 1, it does not take
      effect.
* @li expert_num: An optional integer, represents the number of expert. Type is Int32. Default: 0. When drop_pad_mode
      is 0, it does not take effect. When drop_pad_mode is 1, it should be equal to E when bias is passed in, otherwise
      it should be greater than 0.
* @li expert_capacity: An optional integer, represents the capacity of expert. Type is Int32. Default: 0. When
      drop_pad_mode is 0, it does not take effect. When drop_pad_mode is 1, it should be greater than 0.
*/
REG_OP(MoeFinalizeRoutingV2Grad)
    .INPUT(grad_y, "T1")
    .INPUT(expanded_row_idx, "T2")
    .OPTIONAL_INPUT(expanded_x, "T1")
    .OPTIONAL_INPUT(scales, "T1")
    .OPTIONAL_INPUT(expert_idx, "T2")
    .OPTIONAL_INPUT(bias, "T1")
    .OUTPUT(grad_expanded_x, "T1")
    .OUTPUT(grad_scales, "T1")
    .DATATYPE(T1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T2, TensorType({DT_INT32}))
    .ATTR(drop_pad_mode, Int, 0)
    .ATTR(active_num, Int, 0)
    .ATTR(expert_num, Int, 0)
    .ATTR(expert_capacity, Int, 0)
    .OP_END_FACTORY_REG(MoeFinalizeRoutingV2Grad)

/**
* @brief Binary finds the position of the last row processed by each expert in the sorted_experts array.
* @par Inputs:
* @li sorted_experts: An 1D Tensor, sorted expert array. Type is:Int32.
* @par Outputs:
* @li total_rows_before_expert: A Tensor. Type is:Int32.
* @par Attributes:
* @li num_experts: Required parameter. Type is:Int. The value must be more than 0 and less than 2147483647.
*/
REG_OP(MoeComputeExpertTokens)
    .INPUT(sorted_experts, "T")
    .OUTPUT(total_rows_before_expert, "T")
    .REQUIRED_ATTR(num_experts, Int)
    .DATATYPE(T, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(MoeComputeExpertTokens)

/**
* @brief Rearrange tokens from rank order to expert order
* @par Inputs:
* @li tokens: A 2D tensor, represents tokens in rank-order. Type is BFloat16, Float16, DT_FLOAT8_E5M2,
      DT_FLOAT8_E4M3FN or Int8. Shape supports (A, H). Format supports ND.
      Notices: The value of H must be greater than 0 and less than 16384.(0 < H <16384)
* @li expert_token_num_per_rank: A 2D tensor, represents numbers of tokens belong to an expert on specific rank.
      Type is Int32 or Int64. Shape supports (N, E). Format supports ND.
* @li per_token_scales: A 1D or 2D tensor, optional, represents tokens scale in rank-order. Type is Float32, DT_FLOAT8_E8M0.
      Shape supports (A) or (A,S). Format supports ND. If tokens is FLOAT8, per_token_scales must be DT_FLOAT8_E8M0. \n
  The Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component support 1D. \n
  The Atlas A3 Training Series Product/Atlas A3 Inference Series Product support 1D. \n
  Ascend 950 support 1D or 2D. \n
* @par Outputs:
* @li permute_tokens: A 2D tensor, represents tokens in expert-order. Type is BFloat16, Float16 or
      Int8. Shape supports (A, H). Format supports ND.
* @li permute_per_token_scales: A 1D or 2D tensor, represents tokens scale in expert-order. Type is Float32, DT_FLOAT8_E8M0.
      Shape supports (A) or (A,S). Format supports ND.
  The Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component support 1D. \n
  The Atlas A3 Training Series Product/Atlas A3 Inference Series Product support 1D. \n
  Ascend 950 support 1D or 2D. \n
* @li permute_token_idx: A 1D tensor, represents token idx in rank-order. Type is Int32.
      Shape supports (A). Format supports ND.
* @li expert_token_num: A 1D tensor, represents tokens nums of experts. Type is Int32 or Int64.
      Shape supports (E). Format supports ND.
* @par Attributes:
* @li expert_token_num_type: Optional integer, represents the cumsum or count mode. Type is Int. Default: 1. Value
      supports 0-cumsum or 1-count.
* @li idx_type: Optional integer, represents the gather or scatter index. Type is Int. Default: 0. Value
      supports 0-gather idx or 1-scatter idx.
*/
REG_OP(MoeReRouting)
    .INPUT(tokens, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .INPUT(expert_token_num_per_rank, TensorType({DT_INT32, DT_INT64}))
    .OPTIONAL_INPUT(per_token_scales, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OUTPUT(permute_tokens, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .OUTPUT(permute_per_token_scales, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OUTPUT(permute_token_idx, TensorType({DT_INT32}))
    .OUTPUT(expert_token_num, TensorType({DT_INT32, DT_INT64}))
    .ATTR(expert_token_num_type, Int, 1)
    .ATTR(idx_type, Int, 0)
    .OP_END_FACTORY_REG(MoeReRouting)

/**
* @brief The fusion operator of Gelu activation function and quantum quantization.
* @par Inputs:
* @li x: A tensor. Type is DT_FLOAT32, DT_FLOAT16, DT_BF16.
      Shape supports at least 2 dimensions (M,K1), and at most 8 dimensions.
* @li input_scale: An optional tensor. When quant_mode is "static",it is a required tensor.
*     When quant_mode is "dynamic", it is an optional tensor.
*     Data type is one of DT_FLOAT32, DT_FLOAT16, DT_BF16, and is consistent with x or has higher accuracy.
*     The shape can only be one-dimensional, and the size can only be the last axis of x or 1.
* @li input_offset: An optional tensor. It is an optional input when quant_mode is "static".
*     Data type is one of DT_FLOAT32, DT_FLOAT16, DT_BF16.
*     The shape and type should be the same as input_scale. It can also be null.
* @par Outputs:
* @li y: A tensor. Type is DT_INT8, DT_FLOAT8_E4M3_FN, DT_FLOAT8_E5M2, DT_HIFLOAT8. Shape size is the same as x.
* @li out_scale: A tensor. Type is DT_FLOAT32. Represents Scale used for quantization. The value is
      output only when quant_mode is dynamic.
      The shape of out_scale matches the shape of x across all dimensions except for the last dimension.
* @par Attributes:
* @li approximate: Optional string parameter. Which formula used for activation computation.
      Type is String. The value must be none or tanh. Defaults to none.
* @li quant_mode: Optional string parameter. Which formula used for quantized computation
      Type is String. The value must be dynamic or static. Defaults to dynamic.
* @li dst_type: Optional int32 parameter, specifying the output data type.
*     The value range is [DT_INT8, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_HIFLOAT8]. Defaults to DT_INT8.
* @li round_mode: Optional string parameter, specifying the cast type.
*     The value range is ["rint", "round", "hybrid"]. Defaults to "rint".
*/
REG_OP(GeluQuant)
    .INPUT(x, TensorType({DT_FLOAT32, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(input_scale, TensorType({DT_FLOAT32, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(input_offset, TensorType({DT_FLOAT32, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_INT8, DT_FLOAT8_E4M3_FN, DT_FLOAT8_E5M2, DT_HIFLOAT8}))
    .OUTPUT(out_scale, TensorType({DT_FLOAT32}))
    .ATTR(approximate, String, "none")
    .ATTR(quant_mode, String, "dynamic")
    .ATTR(dst_type, Int, DT_INT8)
    .ATTR(round_mode, String, "rint")
    .OP_END_FACTORY_REG(GeluQuant)


/**
* @brief
   swin_transformer model specific structure.Operator only supports swin_transformer. \n
* @par Inputs:
* Eight inputs, including:
* @li x: A Tensor. Must be one of the following types: float16.
* @li gamma: A Tensor. Must be one of the following types: float16.
* @li beta: A Tensor. Must be one of the following types: float16.
* @li weight: A Tensor. Must be one of the following types: int8.
* @li bias: A Tensor. Must be one of the following types: float16.
* @li quant_scale: A Tensor. Must be one of the following types: float16.
* @li quant_offset: A Tensor. Must be one of the following types: float16.
* @li dequant_scale: A Tensor. Must be one of the following types: uint64. \n

* @par Attributes:
* @li head_num: A required attribute, the type is int. Defaults to 1.
* @li seq_length: A required attribute, the type is int. Defaults to 32.
* @li epsilon: A required attribute, the type is float. Defaults to 0.000001.
* @li ori_height: A required attribute, the type is int. Defaults to 7
* @li ori_weight: A required attribute, the type is int. Defaults to 7. \n
* @li h_win_szie: A required attribute, the type is int. Defaults to 7. \n
* @li w_win_size: A required attribute, the type is int. Defaults to 7. \n
* @li weight_transpose: A required attribute, the type is bool. Defaults to true. \n

* @par Outputs:
* Three outputs, including:
* @li query_output: A Tensor. Must be one of the following types: float16.
* @li key_output: A Tensor. Must be one of the following types: float16.
* @li value_output: A Tensor. Must be one of the following types: float16. \n
*/
REG_OP(SwinTransformerLnQkvQuant)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(gamma, TensorType({DT_FLOAT16}))
    .INPUT(beta, TensorType({DT_FLOAT16}))
    .INPUT(weight, TensorType({DT_FLOAT16}))
    .INPUT(bias, TensorType({DT_FLOAT16}))
    .INPUT(quant_scale, TensorType({DT_FLOAT16}))
    .INPUT(quant_offset, TensorType({DT_FLOAT16}))
    .INPUT(dequant_scale, TensorType({DT_UINT64}))
    .OUTPUT(query_output, TensorType({DT_FLOAT16}))
    .OUTPUT(key_output, TensorType({DT_FLOAT16}))
    .OUTPUT(value_output, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(head_num, Int)
    .REQUIRED_ATTR(seq_length, Int)
    .REQUIRED_ATTR(epsilon, Float)
    .REQUIRED_ATTR(ori_height, Int)
    .REQUIRED_ATTR(ori_weight, Int)
    .REQUIRED_ATTR(h_win_szie, Int)
    .REQUIRED_ATTR(w_win_size, Int)
    .REQUIRED_ATTR(weight_transpose, Bool)
    .OP_END_FACTORY_REG(SwinTransformerLnQkvQuant)

/**
* @brief The quant fusion operator of SwinAttentionScoreQuant.

* @par Inputs:
* @li query: A matrix Tensor. The type support int8.
* @li key: A matrix Tensor. The type support int8.
* @li value: A matrix Tensor. The type support int8.
* @li scale_quant: A Tensor. The type support fp16.
* @li scale_dequant1: A Tensor. The type support uint64.
* @li scale_dequant2: A Tensor. The type support uint64.
* @li bias_quant: A Tensor. The type support fp16.
* @li bias_dequant1: A Tensor. The type support int32.
* @li bias_dequant2: A Tensor. The type support int32.
* @li padding_mask1: A matrix Tensor. The type support fp16.
* @li padding_mask2: A matrix Tensor. The type support fp16.
* @li attention_score: A matrix Tensor. The type support fp16.

* @par Attributes:
* @li query_transpose: A bool. Whether query is transposed. Default: false.
* @li key_transpose: A bool. Whether key is transposed. Default: false.
* @li value_transpose: A bool. Whether value is transposed. Default: false.
* @li softmax_axes: An int. Which axes to calculate softmax. Default: -1.

* @par Outputs:
* @li attention_score: A matrix Tensor. The type support fp16. \n
*/
REG_OP(SwinAttentionScoreQuant)
    .INPUT(query, TensorType({DT_INT8}))
    .INPUT(key, TensorType({DT_INT8}))
    .INPUT(value, TensorType({DT_INT8}))
    .INPUT(scale_quant, TensorType({DT_FLOAT16}))
    .INPUT(scale_dequant1, TensorType({DT_UINT64}))
    .INPUT(scale_dequant2, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(bias_quant, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(bias_dequant1, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(bias_dequant2, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(padding_mask1, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(padding_mask2, TensorType({DT_FLOAT16}))
    .OUTPUT(attention_score, TensorType({DT_FLOAT16}))
    .ATTR(query_transpose, Bool, false)
    .ATTR(key_transpose, Bool, false)
    .ATTR(value_transpose, Bool, false)
    .ATTR(softmax_axes, Int, -1)
    .OP_END_FACTORY_REG(SwinAttentionScoreQuant)

/**
* @brief Fusion op DequantRopeQuantKvcache.

* @par Inputs:
* thirteen inputs, including:
* @li x: A Tensor with shape (B, S, H) or (B, H), H is (Nq+Nkv+Nkv)*D, format support ND.
* The type support float16, bf16, int32.
* @li cos: A Tensor with shape (B, S, 1, D) or (B, D). The type support float16, bf16, format support ND.
* @li sin: A Tensor with shape (B, S, 1, D) or (B, D). The type support float16, bf16, format support ND.
* @li k_cache: A Tensor with shape (C_1, C_2, Nkv, D) indicates kcache for in-place updates.
* The type support int8, format support ND.
* @li v_cache: A Tensor with shape (C_1, C_2, Nkv, D) indicates vcache for in-place updates.
* The type support int8, format support ND.
* @li indices: A Tensor with shape (B) when cache_mode is contiguous with shape (B * S) when cache_mode is page.
* The type support int32, format support ND.
* @li scale_k: A Tensor with shape (Nkv, D). The type support float32, format support ND.
* @li scale_v: A Tensor with shape (Nkv, D). The type support float32, format support ND.
* @li offset_k: A Tensor with shape (Nkv, D). An optional input parameter. The type support float32.
* format support ND.
* @li offset_v: A Tensor with shape (Nkv, D). An optional input parameter. The type support float32.
* format support ND.
* @li weight_scale: A Tensor with shape (D) indicates the weight scale factor of the dequantization parameter.
* An optional input parameter. The type support float32.
* @li activation_scale: A Tensor with shape (B * S) or (B) indicates the activation scale factor of the dequantization parameter.
* An optional input parameter. The type support float32.
* @li bias: A Tensor with shape (D). An optional input parameter. The type support float32, bf16, float16, int32.

* @par Attributes:
* @li size_splits: A list of int. Specifies the size of spliting qkv.
* @li quant_mode: A string. A optional attribute. Specifies the method of quant. Default: "static".
* @li layout: A string. A optional attribute. Specifies the format of input. Default: "BSND".
* @li kv_output: A bool. A optional attribute. Whether to output kv. Default: "false".
* @li cache_mode:  A string. A optional attribute. Specifies the cache mode for kcache and vcache.
*    Should be "contiguous" or "page", default is "contiguous".

* @par Outputs:
* @li q: A Tensor with shape (B, S, Nq, D) or (B, Nq, D). The type support float16, bf16.
* @li k: A Tensor with shape (B, S, Nkv, D) or (B, Nkv, D). The type support float16, bf16.
* @li v: A Tensor with shape (B, S, Nkv, D) or (B, Nkv, D). The type support float16, bf16.
* @li k_cache: A Tensor with shape (C_1, C_2, Nkv, D). The type support int8, format support ND.
* @li v_cache: A Tensor with shape (C_1, C_2, Nkv, D). The type support int8, format support ND.
*/
REG_OP(DequantRopeQuantKvcache)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16, DT_INT32}))
    .INPUT(cos, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(sin, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(k_cache, TensorType({DT_INT8}))
    .INPUT(v_cache, TensorType({DT_INT8}))
    .INPUT(indices, TensorType({DT_INT32, DT_INT64}))
    .INPUT(scale_k, TensorType({DT_FLOAT32}))
    .INPUT(scale_v, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(offset_k, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(offset_v, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(weight_scale, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(activation_scale, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT32, DT_BF16, DT_FLOAT16, DT_INT32}))
    .OUTPUT(q, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(k, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(v, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(k_cache, TensorType({DT_INT8}))
    .OUTPUT(v_cache, TensorType({DT_INT8}))
    .REQUIRED_ATTR(size_splits, ListInt)
    .ATTR(quant_mode, String, "static")
    .ATTR(layout, String, "BSND")
    .ATTR(kv_output, Bool, false)
    .ATTR(cache_mode, String, "contiguous")
    .OP_END_FACTORY_REG(DequantRopeQuantKvcache)

/**
* @brief DequantBias. \n

* @par Inputs:
* @li x: A 2D tensor. Input tensor representing the inverse quantization operation.
* Supported format "ND". The shape is [M, N], and the data type supports int32.
* @li weight_scale: A 1D tensor. Indicates the weight of the multiplication on the N-dimensional input of the anti-quantization operation.
* The shape is [N], and the length is consistent with the N-dimensional length of x. The data type supports float32, bfloat16.
* @li activate_scale: A 1D tensor. The data type supports float32.
* Indicates the weight of the multiplication on the M dimension of the input for the anti-quantization operation.
* The shape is [M], with a length consistent with the M dimension of x, and the data type supports float32.
* Supported format "ND".
* @li bias: A 1D tensor. Indicates the weight of the addition on the N-dimensional input of the anti-quantization operation.
* The shape is [N], with a length consistent with the N-dimensional length of x.
* The data type supports float32, bfloat16, float16, int32. Supported format "ND".

* @par Attributes:
* output_dtype: An int attr. Indicates the data type of the output out. The value is [1, 27].
* A value of 1 indicates that the output type is float16, and a value of 27 indicates that the output type is bfloat16.
* When the weight_scale data type is float32, this parameter is set to 1; when the weight_scale data type is bfloat16,
* this parameter is set to 27.

* @par Outputs:
* y: A 2D tensor. The output tensor of the quantization operation.
* The shape is [M, N], and the data type supports float16, bfloat16. Supported format "ND". \n
*/
REG_OP(DequantBias)
    .INPUT(x, TensorType({DT_INT32}))
    .INPUT(weight_scale, TensorType({DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(activate_scale, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(bias, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT32, DT_INT32}))
    .REQUIRED_ATTR(output_dtype, Int)
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(DequantBias)

/**
* @brief AddLora, the operation kernel for batch gather matmul.
* @par Inputs:
* @li y: A tensor. Indicates the tensor to be updated by accumulation. The data type is float16. Shape dimension is 2D: [B, H3].
* Supported format "ND". The first dimension needs to be consistent with the first dimension of x, both represented by B.
* @li x: A tensor. Input tensor before grouping. Data type supports float16. Shape dimension is 2D: [B, H1],
* where H1 is a multiple of 16. Supported format "ND".
* @li weightB: A weight tensor of type float16. Supported format "ND". Represents the second weight matrix for matrix multiplication.
* Shape dimension is 4D: [W, L, H2, R]. The third dimension needs to be smaller than the second dimension of y (H2<H3),
* and H2 is an integer multiple of 16.
* @li indices: A tensor. Indicates the group index of the input x. The data type is int32. Shape dimension is 1D: [B],
* which must be the same as the first dimension of x and y. Both of them are represented by B. Supported format "ND".
* @li weightA: A optional weight tensor of type float16. Indicates the first weight matrix for matrix multiplication.
* If the value is null, the first matrix multiplication is skipped.
* Shape dimension is 4D: [W, L, R, H1]. The first two dimensions must be consistent with the first two dimensions of weightB,
* which are represented by W and L. The third dimension must be consistent with the fourth dimension of weightB,
* and both dimensions are represented by R. The fourth dimension must be the same as the second dimension of x,
* both are represented by H1, and must be an integer multiple of 16. Supported format "ND".

* @par Attributes:
* @li layer_idx: A optional int, default value is 0, indicates the layer id of weight tensors.
* The value must be less than the second dimension L of weightB.
* @li scale: A optional float, default value is 1e-3, scales up the multiplication results.
* @li y_offset: A optional int, default value is 0,  represents the offset of y.
* The value needs to be less than the second dimension H3 of y.
* @li y_slice_size: A optional int, default value is -1, represents the slice_size of y to be updated.
* The value needs to be less than the second dimension H3 of y.

* @par Outputs:
* y_out: A tensor of type float16, the shape requirements are consistent with the shape of y.
* The shape dimension is two dimensions. Supported format "ND". Has the same as type as y.
*/
REG_OP(AddLora)
    .INPUT(y, TensorType({DT_FLOAT16}))
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(weightB, TensorType({DT_FLOAT16}))
    .INPUT(indices, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(weightA, TensorType({DT_FLOAT16}))
    .ATTR(layer_idx, Int, 0)
    .ATTR(scale, Float, 1e-3)
    .ATTR(y_offset, Int, 0)
    .ATTR(y_slice_size, Int, -1)
    .OUTPUT(y_out, TensorType({DT_FLOAT16}))
    .OP_END_FACTORY_REG(AddLora)


/**
* @brief Update multi output of RingAttention.

* @par Inputs:
* seven inputs, including:
* @li prev_attn_out: A matrix Tensor. The type support float16, bf16, float32.
* @li prev_softmax_max: A matrix Tensor. The type support float32.
* @li prev_softmax_sum: A matrix Tensor. The type support float32.
* @li cur_attn_out: A matrix Tensor. An optional input parameter. The type support float16, bf16, float32.
* @li cur_softmax_max: A matrix Tensor. An optional input parameter. The type support float32.
* @li cur_softmax_sum: A matrix Tensor. An optional input parameter. The type support float32.
* @li actual_seq_qlen: A matrix Tensor. An optional input parameter. The type support int64. If used,
* layout need to be setted TND. ex. If the attn_out seqlen is [2,2,2,2,2], this parameter need be setted [2,4,6,8,10].

* @par Attributes:
* @li input_layout: A string. A optional attribute. Specifies the layout of `attn_out`,
* the value must be one of ["SBH"]. Default: "SBH".

* @par Outputs:
* @li attn_out: A matrix Tensor. The type support float16, bf16, float32.
* @li softmax_max: A matrix Tensor. The type support float32.
* @li softmax_sum: A matrix Tensor. The type support float32.
*/
REG_OP(RingAttentionUpdate)
    .INPUT(prev_attn_out, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .INPUT(prev_softmax_max, TensorType({DT_FLOAT32}))
    .INPUT(prev_softmax_sum, TensorType({DT_FLOAT32}))
    .INPUT(cur_attn_out, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .INPUT(cur_softmax_max, TensorType({DT_FLOAT32}))
    .INPUT(cur_softmax_sum, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(actual_seq_qlen, TensorType({DT_INT64}))
    .OUTPUT(attn_out, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(softmax_max, TensorType({DT_FLOAT32}))
    .OUTPUT(softmax_sum, TensorType({DT_FLOAT32}))
    .ATTR(input_layout, String, "SBH")
    .OP_END_FACTORY_REG(RingAttentionUpdate)

/**
* @brief Multiplies dynamic asymmetric quantize and sparse updates into a variable reference.

* @par Inputs:
* Five inputs, including:
* @li x: A tensor which layout need to be setted BSH. Shape is (B, 1, H).
* The type support float16, bfloat16, format support ND.
* @li indices: A 1D tensor with shape (B). Indices of scatter. The type support int32, format support ND.
* @li var: A tensor with shape (B, S, 1, H). Target tensor to which the quantization results are scattered.
* The type support int4, format support ND.
* @li var_scale: A tensor with shape (B, S). Target tensor to which the quantization scales are scattered.
* The type support float32, format support ND.
* @li var_offset: A tensor with shape (B, S). Target tensor to which the quantization offsets are scattered.
* The type support float32, format support ND.

* @par Outputs:
* @li var: A tensor with shape (B, S, H). Result tensor after an in-place scatter.
* The type support int4, format support ND.
* @li var_scale: A tensor with shape (1, B, S). Result tensor after an in-place scatter.
* The type support float32, format support ND.
* @li var_offset: A tensor with shape (1, B, S). Result tensor after an in-place scatter.
* The type support float32, format support ND.
*/
REG_OP(DynamicQuantUpdateScatterV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(var, TensorType({DT_INT4}))
    .INPUT(var_scale, TensorType({DT_FLOAT}))
    .INPUT(var_offset, TensorType({DT_FLOAT}))
    .OUTPUT(var, TensorType({DT_INT4}))
    .OUTPUT(var_scale, TensorType({DT_FLOAT}))
    .OUTPUT(var_offset, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(DynamicQuantUpdateScatterV2)

/**
* @brief SwiGlu and DynamicQuant are integrated to implement quantization. Only MOE group quantization is supported.

* @par Inputs:
* @li x: A matrix tensor. Must be one of the following types: float32,float16,bfloat16, has format ND.
* @li smooth_scales: A optional tensor. Describing the result of dynamic quantize scales.
        A tensor of type float32, has format ND.
* @li offsets: A optional tensor, describing the data of offsets, a tensor of type float32, has format ND.
* @li group_index: A optional tensor, described grouping data, a tensor of type int32, has format ND.

* @par Attributes:
* @li activate_left: A optional bool.
*     The SwiGlu activate_left algorithm to use:
*     'false' (activate right) or 'true' (activate left), defalut is 'false' (activate right).
* @li quant_mode: Optional parameter, which formula used for quantized computation.
      Type is String, the value must be "dynamic" or "static" or "dynamic_msd", "static" indicates static quantization,
      "dynamic" indicates dynamic quantization, and "dynamic_msd" indicates dynamic mean squared displacement quantization, defaults to dynamic.
      Now only support "dynamic" and "static" mode.
* @li group_list_type: Optional parameter, which used to describe group_index mode.
      Type is Int, the value must be 0 or 1, 0 indicates "cumsum" mode, 1 indicates "count" mode.
      Now only support "cumsum" and "count" mode.
* @li dst_type: Optional parameter, which used to describe quant output mode.
      Type is Int, the value must be 2(DT_INT8 enum value) or 29(DT_INT4 enum value), 2 indicates "int8 quant output" mode, 29 indicates "int4 quant output" mode.
      Now only support "int8 quant output" and "int4 quant output" mode.

* @par Outputs:
* @li y: A tensor ,type is int8 or int4, the size of the last dimension of output y is half of the size of input x.
       And the size of other dimensions is the same as that of input x, now only support DT_INT8.
* @li scale: A tensor. Type is float32.
      The shape of scale matches the shape of x across all dimensions except for the last dimension.
*/
REG_OP(SwiGluQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(smooth_scales, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(offsets, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(group_index, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT8, DT_INT4}))
    .OUTPUT(scale, TensorType({DT_FLOAT}))
    .ATTR(activate_left, Bool, false)
    .ATTR(quant_mode, String, "dynamic")
    .ATTR(group_list_type, Int, 0)
    .ATTR(dst_type, Int, DT_INT8)
    .OP_END_FACTORY_REG(SwiGluQuant)



/**
* @brief RopeWithSinCosCache.

* @par Inputs:
* @li positions: A tensor of type int32 of int64.
* @li queryIn: A tensor of type float, bf16, float16.
* @li keyIn: A tensor of type float, bf16, float16.
* @li cosSinCache: A tensor of type float, bf16, float16.

* @par Attributes:
* @li numQHeads: A int attr.
* @li numKHeads: A int attr.
* @li headSize: A int attr.
* @li mropeSection: A ListInt attr.
* @li qStride: A int attr.
* @li KStride: A int attr.
* @li isNeoxStyle: A bool attr.

* @par Outputs:
* @li queryOut: A tensor of type FP16/FP32/BF16.
* @li keyOut: A tensor of type FP16/FP32/BF16.
*/

REG_OP(RopeWithSinCosCache)
      .INPUT(positions, TensorType({DT_INT32, DT_INT64}))
      .INPUT(queryIn, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
      .INPUT(keyIn, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
      .INPUT(cosSinCache, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
      .OUTPUT(queryOut, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
      .OUTPUT(keyOut, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
      .REQUIRED_ATTR(numQHeads, Int)
      .REQUIRED_ATTR(numKHeads, Int)
      .REQUIRED_ATTR(headSize, Int)
      .ATTR(mropeSection, ListInt, {0,0,0})
      .ATTR(qStride, Int, 0)
      .ATTR(kStride, Int, 0)
      .ATTR(isNeoxStyle, Bool, true)
      .OP_END_FACTORY_REG(RopeWithSinCosCache)

/**
 * @brief Recurrent Gated Delta Rule operator interface implementation.
 *
 * @par Inputs:
 * @li query: queries of shape [T, Nk, Dk]. Support dtype: bfloat16. Support format: ND.
 * @li key: keys of shape [T, Nk, Dk]. Support dtype: bfloat16. Support format: ND.
 * @li value: values of shape [T, Nv, Dv]. Support dtype: bfloat16. Support format: ND.
 * @li beta: betas of shape [T, Nv]. Support dtype: bfloat16. Support format: ND.
 * @li state: initial states of shape [BlockNum, Nv, Dv, Dk]. Support dtype: bfloat16. Support format: ND.
 * @li actual_seq_lengths: actual sequence length of shape [B,] used for variable-length training. Support dtype: int32. Support format: ND.
 * @li ssm_state_indices: indices to map the input sequences to the states of shape [T,]. Support dtype: int32. Support format: ND.
 * @li g: decays of shape [T, Nv], alpha = e^g. Support dtype: float32. Support format: ND.
 * @li gk: key decays of shape [T, Nv, Dk], alpha = e^g. Support dtype: float32. Support format: ND.
 * @li num_accepted_tokens: number of accepted tokens for each sequence during decoding of shape (B,). Support dtype: int32. Support format: ND.
 *
 * @par Attributes:
 * scale_value: scale factor for the RetNet attention scores, usually 1/sqrt(Dk), the default value is 1.0. dtype: float32.
 *
 * @par Output:
 * @li out: outputs of shape [T, Nv,Dv]. Support dtype: bfloat16. Support format: ND.
 * @li state: final states of shape [BlockNum, Nv, Dv, Dk]. Support dtype: bfloat16. Support format: ND.
 */
REG_OP(RecurrentGatedDeltaRule)
.INPUT(query, "T1")
.INPUT(key, "T1")
.INPUT(value, "T1")
.INPUT(beta, "T1")
.INPUT(state, "T1")
.INPUT(actual_seq_lengths, "T2")
.INPUT(ssm_state_indices, "T2")
.OPTIONAL_INPUT(g, "T3")
.OPTIONAL_INPUT(gk, "T3")
.OPTIONAL_INPUT(num_accepted_tokens, "T2")
.OUTPUT(out, "T1")
.OUTPUT(state, "T1")
.ATTR(scale_value, Float, 1.0)
.DATATYPE(T1, TensorType({DT_BF16}))
.DATATYPE(T2, TensorType({DT_INT32}))
.DATATYPE(T3, TensorType({DT_FLOAT}))
.OP_END_FACTORY_REG(RecurrentGatedDeltaRule)

/**
* @brief Function GroupedMatmulFinalizeRouting. This op mixs GroupedMatmul和MoeFinalizeRouting. After the calculation of GroupedMatmul, perform a combine operation on the output according to the index, and support the format where w is in the Ascend affinity data layout.

* @par Inputs:
* @li x: A tensor, which is the input x in the formula, supports the ND data format (refer to Data Format), and the shape supports 2 dimensions with the dimension being (m, k). The data type supports INT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_HIFLOAT8.
* @li w: A tensor of weight, which supports the Ascend affinity data layout format as described in Data Format. The data type supports INT8, INT4, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, and the shape supports 3 or 5 dimensions. On the Ascend 950 AI Processor, five-dimensional shapes are supported only when the data type is INT8, DT_FLOAT8_E4M3FN or DT_HIFLOAT8.
* When transposew is false, each dimension is represented as: (e, n1, k1, k0, n0), where k0 = 16, n0 = 32. The k in the x shape and the k1 in the w shape need to satisfy the following relationship: ceilDiv(k, 16) = k1.
* The aclnnCalculateMatmulWeightSizeV2 interface and the aclnnTransMatmulWeight interface can be used to complete the conversion of the input format from the ND format to the Ascend affinity data layout format.
* @li scale: It represents the scaling factor in the quantization parameters. It supports the ND data format as described in Data Format. The supported data type is FLOAT32, INT64, BFLOAT16, DT_FLOAT8_E8M0.
* and if w is INT4, the shape is three-dimensional (e, 1, n), others scenarios is two-dimensional (e, n), where the values of e and n are consistent with those of e and n in w.  The supported data type is BFLOAT16.
* @li bias: A tensor of bias, contains all bias of inputs for matmul. For each tensor, the data type of elements supports float32; the format supports ND..
* @li offset: A tensor of offset, only supported when w is INT4, the shape is three-dimensional (e, 1, n), the format supports ND. The supported data types is FLOAT32.
* @li pertoken_scale: The dequantization parameters for matrix calculation support the ND data format (refer to Data Format). They correspond to the x matrix, with a dimension of (m). The supported data type is FLOAT32, DT_FLOAT8_E8M0. Non-contiguous tensors are not supported.
* @li group_list: a tensor, indicates M-axis distributation of groups of matmuls for inputs and outputs.
* Data type of elements is int64. Format: ND. The first dimension of group_list should not be greater than 1024, meaning it supports up to 1024 groups.
* @li shared_input: In the MoE (Mixture of Experts) calculation, the output of the shared experts needs to undergo a combine operation with the output of the MoE experts. The supported data types are bfloat16.
* @li logit: In the MoE (Mixture of Experts), for the logit magnitudes of each token, the output of the matrix multiplication is multiplied by these logits, and then combined according to the indices. The supported data type is float32.
* @li row_index: The outputs of the MoE (Mixture of Experts) are combined according to the rowIndex, where the values in rowIndex serve as the indices for the scatter add operation during the combination. The supported data types are int64 and int32.
* @par Attributes:
* @li dtype: The type of GroupedMatmul. The type is int, which output:0：FLOAT32；1：FLOAT16；2：BFLOAT16.
* @li shared_input_weight: The coefficients for combining the shared experts and the MoE experts. The shareInput is multiplied first with these coefficients, and then the result is accumulated with the output of the MoE experts. The supported data type is float32.
* @li shared_input_offset: The offset of the output of the shared experts in the total output. The supported data type is int64.
* @li transpose_x: Whether the left matrix is transposed. Default value: false(not transposed).
* @li transpose_w: Whether the right matrix is transposed. Default value: false(not transposed).
* @li output_bs: The size of the highest dimension of the output.
* @li group_list_type: Group type of GroupedMatmul. Default value: 1. When configured as 0: It is in the cumsum mode, which means it is the prefix sum. When configured as 1: It is in the count mode. The supported data type is int64. \n

* @par Outputs:
* y: A tensor List, contains all result of groups of matmuls. For each tensor,
* the data type of elements supports float32; the format supports ND. \n

* @attention Constraints:
* Support combinations of data type: \n
*  | x             | w             |  scale           | bias     | offset  | pertokenScale | groupList | sharedInput | logit   | rowIndex    |  out    | \n
*  | ----          | ----          | -------          | -------  | ------- | ------------- | --------- | ----------- | ------- | --------    | ------- | \n
*  | INT8          | INT4          | INT64            | FLOAT32  | FLOAT32 | FLOAT32       | INT64     | BFLOAT16    | FLOAT32 | INT64       | FLOAT32 | \n
*  | INT8          | INT4          | INT64            | FLOAT32  | null    | FLOAT32       | INT64     | BFLOAT16    | FLOAT32 | INT64       | FLOAT32 | \n
*  | INT8          | INT8          | FLOAT32          | null     | null    | FLOAT32       | INT64     | BFLOAT16    | FLOAT32 | INT64       | FLOAT32 | \n
*  | INT8          | INT8          | FLOAT32          | null     | null    | FLOAT32       | INT64     | null        | null    | INT64       | FLOAT32 | \n
*  | INT8          | INT8          | FLOAT32          | BFLOAT16 | null    | FLOAT32       | INT64     | null        | null    | INT64       | FLOAT32 | \n
*  | INT8          | INT8          | BFLOAT16         | BFLOAT16 | null    | FLOAT32       | INT64     | null        | null    | INT64       | FLOAT32 | \n
*  | INT8          | INT8          | FLOAT32/BFLOAT16 | BFLOAT16 | null    | FLOAT32/null  | INT64     | BFLOAT16    | FLOAT32 | INT64/INT32 | FLOAT32 | \n
*  | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT32/BFLOAT16 | BFLOAT16 | null    | FLOAT32/null  | INT64     | BFLOAT16    | FLOAT32 | INT64       | FLOAT32 | \n
*  | HIFLOAT8      | HIFLOAT8      | FLOAT32/BFLOAT16 | BFLOAT16 | null    | FLOAT32/null  | INT64     | BFLOAT16    | FLOAT32 | INT64       | FLOAT32 | \n
* Among them: \n
* The dimension m = batch * topk, and the value range is [1, 16 * 1024 * 8]. The functionality is not guaranteed if it exceeds this range. \n
* k supports the value include of 256、512、1024、1408、2048. The functionality is not guaranteed if it exceeds this range. \n
* n supports the value include of 2048、7168、7680. The functionality is not guaranteed if it exceeds this range. \n
* The value range of e is [1, 256]. The functionality is not guaranteed if it exceeds this range. \n
* The value range of bs/p is [1, 2 * 1024], and p = [8, 16, 32, 48, 64, 96, 128, 144, 288]. \n
* The value range of bs is [1, 16 * 1024]. The functionality is not guaranteed if it exceeds this range. \n
* The sum of the values in grouplist is less than or equal to m. \n
* When group_list_type is 0, group_list must be a non-negative monotone non-decreasing array; when group_list_type is 1, it must be a non-negative array. \n

* @attention MX Quantization Constraints:
*  |             x             |            w              |     scale     |      bias    | offset  | pertokenScale | groupList |  sharedInput  | logit   | rowIndex |  out    | \n
*  | ------------------------- | ------------------------- | ------------- | ------------ | ------- | ------------- | --------- | ------------- | ------- | -------- | ------- | \n
*  | FLOAT8_E4M3FN/FLOAT8_E5M2 | FLOAT8_E4M3FN/FLOAT8_E5M2 | FLOAT8_E8M0   | FLOAT32/NULL | NULL    | FLOAT8_E8M0   | INT64     | BFLOAT16/NULL | FLOAT32 | INT64    | FLOAT32 | \n
*  | FLOAT4_E2M1               | FLOAT4_E2M1               | FLOAT8_E8M0   | FLOAT32/NULL | NULL    | FLOAT8_E8M0   | INT64     | BFLOAT16/NULL | FLOAT32 | INT64    | FLOAT32 | \n
* PertokenScale dimension is represented as: (m, ceil(k/64), 2). The data type supports FLOAT8_E8M0, and the shape supports 3 dimensions.
* Scale dimension is represented as: (e, ceil(k/64), n, 2) or (e, n, ceil(k/64), 2). The data type supports FLOAT8_E8M0, and the shape supports 4 dimensions.
* In the MXFP4 scenario, the constraint that k must be even. Under non-transposed x2, n must also be even.
* In MXFP4/MXFP8 scenarios, weight(w) transposition is supported. The transposition property of w must be consistent with that of scale.
* e must be less than or equal to 1024.
* In the MXFP4 scenario, k cannot be equal to 2.
*/
REG_OP(GroupedMatmulFinalizeRouting)
.INPUT(x, TensorType({DT_INT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3, DT_FLOAT4_E2M1, DT_HIFLOAT8}))
.INPUT(w, TensorType({DT_INT8, DT_INT4, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3, DT_FLOAT4_E2M1, DT_HIFLOAT8}))
.OPTIONAL_INPUT(scale, TensorType({DT_FLOAT, DT_INT64, DT_BF16, DT_FLOAT8_E8M0}))
.OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_BF16}))
.OPTIONAL_INPUT(pertoken_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
.OPTIONAL_INPUT(group_list, TensorType({DT_INT64}))
.OPTIONAL_INPUT(shared_input, TensorType({DT_BF16}))
.OPTIONAL_INPUT(logit, TensorType({DT_FLOAT}))
.OPTIONAL_INPUT(row_index, TensorType({DT_INT64, DT_INT32}))
.OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
.OUTPUT(y, TensorType({DT_FLOAT}))
.ATTR(dtype, Int, 0)
.ATTR(shared_input_weight, Float, 1.0)
.ATTR(shared_input_offset, Int, 0)
.ATTR(transpose_x, Bool, false)
.ATTR(transpose_w, Bool, false)
.ATTR(output_bs, Int, 0)
.ATTR(group_list_type, Int, 1)
.ATTR(tuning_config, ListInt, {0})
.OP_END_FACTORY_REG(GroupedMatmulFinalizeRouting)

/**
* @brief The fusion operator of antiquant and matmul for A8W4, MxA8W4, and A8W8.

* @par Inputs:
* @li x1: A matrix Tensor. The shape supports (m, k), and the format supports ND.
* The data type supports float8_e5m2, float8_e4m3fn, int8. The m and k value must be in [1, 65535]. The k value
* must be in [1, 65535] and must be a multiple of 64.
* @li x2: A matrix Tensor of quantized weight. The shape supports (n, k), and the format supports ND/FRACTAL_NZ.\n
* - In MxA8W4 scenario:
* For ND format, the data type supports float4_e2m1. For FRACTAL_NZ format, the data type supports float4_e2m1.\n
* - In A8W4 scenario:
* The data type supports float4_e2m1.
* The k, n value must be in [1, 65535] and k, n value must be a multiple of 64.
* The shape must be (k1, n1, n0, k0) (n0 = 16, k0 = 32) when the format is FRACTAL_NZ.
* - In A8W8 scenario: The data type supports int8.
* @li bias: An Optional Tensor.
* The shape supprts(1, n), format supports ND, the type supports bfloat16, float16, float32.
* @li x1_scale: An Optional Tensor for quantization parameters.
* The type supports float8_e8m0, float16, bfloat16, format supports ND.
* The shape supprts(m, k/group_size) when type is float8_e8m0, float32.
* The type bfloat16, bfloat16 is not supported yet.
* @li x2_scale: An Optional Tensor for quantization parameters.
* The type supports float8_e8m0, bfloat16, float16, uint64, format supports ND, The shape supports (n, ceildiv(k, group_size)).
* - In A8W8 scenario: The shape supports (ceildiv(k, group_size_k), ceildiv(n, group_size_n)). The type supports float32.
* @li y_scale: An Optional Tensor for quantization parameters.
* The type support uint64, format supports ND, The shape supprts(1, n).
* @li x1_offset: An Optional Tensor for quantization parameters. It's not supported yet.
* @li x2_offset: An Optional Tensor for quantization parameters. It's not supported yet.
* @li y_offset: An Optional Tensor for quantization parameters. It's not supported yet.
* @li x2_table: An Optional Tensor for quantization parameters. It's not supported yet.
*
* @par Attributes:
* @li dtype: An int32. The data type of y. It supports 1(float16) and 27(bfloat16).
* @li compute_type: An int32. It's compute type. The value must be -1(auto) or 4(f8f4). Default -1.
* @li transpose_x1: A bool. x1 is transposed if true. Default: false.
* When transpose_x1 is true, x1's shape is (k, m). Currently, it should always be false.
* @li transpose_x2: A bool. x2 is transposed if true. Default: false.
* When transpose_x2 is true, x2's shape is (n, k), x2_scale's shape should be (n, k / group_size).
* Currently, it should always be true.
* In A8W8 scenario, when transpose_x2 is true, x2_scale's shape should be
* (ceildiv(n, group_size_n), ceildiv(k, group_size_k)).
* @li group_size: An int32. Default: -1.

* @par Outputs:
* y: A matrix Tensor. The data type support bfloat16, float16. The format supports ND.
*/
REG_OP(QuantBatchMatmulV4)
    .INPUT(x1, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_INT8}))
    .INPUT(x2, TensorType({DT_FLOAT4_E2M1, DT_INT4, DT_INT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT32}))
    .OPTIONAL_INPUT(x1_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT8_E8M0, DT_FLOAT32}))
    .OPTIONAL_INPUT(x2_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT8_E8M0, DT_UINT64, DT_FLOAT32}))
    .OPTIONAL_INPUT(y_scale, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(x1_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(x2_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(y_offset, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(x2_table, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(dtype, Int)
    .ATTR(compute_type, Int, -1)
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(group_size, Int, -1)
    .OP_END_FACTORY_REG(QuantBatchMatmulV4)

/**
* @brief The fusion operator of QuantBatchMatmul and ReduceSum.

* @par Inputs:
* @li x1: A matrix tensor. The shape supports (batch, m, k), and the format supports ND. The data type supports int8.
* @li x2: A matrix tensor of quantized weight. The shape supports (batch, k, n), and the format supports FRACTAL_NZ. The data type supports int8.
* @li dims: A 1D list or tuple of int64. Specifies the dimensions to reduce.
* @li bias: Reserved parameter, not supported now, please pass nullptr.
* @li x1_scale: A tensor for x1 quantization parameters. The type supports float32, format supports ND, The shape supprts(b, m).
* @li x2_scale: A tensor for x2 quantization parameters. The type supports bfloat16, format supports ND. The shape supprts(n,).
* @li y_scale: Reserved parameter, not supported now, please pass nullptr.
* @li x1_offset: Reserved parameter, not supported now, please pass nullptr.
* @li x2_offset: Reserved parameter, not supported now, please pass nullptr.
* @li y_offset: Reserved parameter, not supported now, please pass nullptr.
* @li x2_table: Reserved parameter, not supported now, please pass nullptr. \n

* @par Attributes:
* @li dtype: A Int. Declare the output dtype, supports 27(bfloat16).
* @li compute_type: Reserved attributes, not supported now. Default is -1.
* @li transpose_x1: A bool. If true, changes the shape of "x1" from [m, k] to [k, m] before multiplication. Default: false. Only supports false now.
* @li transpose_x2: A bool. If true, changes the shape of "x2" from [k, n] to [n, k] before multiplication. Default: false. Only supports false now.
* @li group_size: Reverved attributes, not supported now. Default is -1.
* @li keep_dims: A bool. If true, keeps the original input dims in the output. Default: false. Only support false now. \n

* @par Outputs:
* y: A matrix tensor. The data type is bfloat16. The format supports ND. \n

* Atlas A2 Trainging Series Product/Atlas 800I A2 Inference Product or Atlas A3 Training Series Product: \n
* | x1   | x2   | x1Scale  | x2Scale  | yScale | x1Offset | x2Offset | yOffset | bias  | out     |
* |------|------|----------|----------|--------|----------|----------|---------|-------|---------|
* | INT8 | INT8 | FLOAT32  | BFLOAT16 | null   | null     | null     | null    | null  |BFLOAT16 |
*/
REG_OP(QuantMatmulReduceSum)
    .INPUT(x1, TensorType({DT_INT8}))
    .INPUT(x2, TensorType({DT_INT8}))
    .INPUT(dims, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(bias, TensorType({DT_BF16}))
    .OPTIONAL_INPUT(x1_scale, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(x2_scale, TensorType({DT_BF16}))
    .OPTIONAL_INPUT(y_scale, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(x1_offset, TensorType({DT_BF16}))
    .OPTIONAL_INPUT(x2_offset, TensorType({DT_BF16}))
    .OPTIONAL_INPUT(y_offset, TensorType({DT_BF16}))
    .OPTIONAL_INPUT(x2_table, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_BF16}))
    .REQUIRED_ATTR(dtype, Int)
    .ATTR(compute_type, Int, -1)
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(group_size, Int, -1)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(QuantMatmulReduceSum)

/**
* @brief MoeDistributeDispatchV2 operator interface implementation.

* @par Inputs
* Five inputs, including:
* @li x: A tensor. Support dtype: float16,bfloat16, dimension must be 2. Shape supports (BS, H), support format: ND.
* @li expertIds: A tensor. Support dtype: int32, indicates top k experts of each token, dimension must be 2. Shape supports (BS, K), support format: ND.
* @li scales: An optional tensor. Support dtype: float32, dimension must be 2, support format: ND.
* @li x_active_mask: An optional tensor. Support dtype: bool, support format: ND.
* @li expert_scales: An optional tensor. Support dtype: float32. Shape supports (BS, K), support format: ND.
* @li performance_info: An optional tensor. Support dtype: int64. Shape supports (BS, ), support format: ND.

* @par Attributes
* @li group_ep: Required. Input ep comm group name, ep means experts parallelism, dtype: String.
* @li ep_world_size: Required. Input ep comm world size, dtype: int64.
* @li ep_rank_id: Required. Input ep comm rank Id, dtype: int64.
* @li moe_expert_num: Required. Input moe expert num, dtype: int64.
* @li group_tp: Input tp comm group name, tp means tensor parallelism, dtype: String.
* @li tp_world_size: Input tp comm world size, dtype: int64.
* @li tp_rank_id: Input tp comm rank Id, dtype: int64.
* @li expert_shard_type: Input moe shard type, dtype: int64.
* @li shared_expert_num: Input shared expert num, dtype: int64.
* @li shared_expert_rank_num: Input shared expert rank num, dtype: int64.
* @li quant_mode: Input quant mode. The options are 0 (non-quantization), 1 (static quantization), and 2 (dynamic quantization). dtype: int64.
* @li global_bs: Input global batch size, dtype: int64.
* @li expert_token_nums_type: Input expert token nums type, dtype: int64.
* @li comm_alg: Input comm alg type, dtype: String.

* @par Outputs
* Seven outputs, including:
* @li expand_x: A tensor. Result of each expert after dispatching. Support dtype: float16,bfloat16,int8,float8_e4m3,float8_e5m2，hifloat8. Shape supports (A, H), support format: ND.
* @li dynamic_scales: If quant is enabled, scale value of each token. A tensor. Support dtype: float32,float8_e8m0. Shape supports (A, ), support format: ND.
* @li assist_info_for_combine: A tensor. Support dtype: int32. Shape supports (A * 128), support format: ND.
* @li expert_token_nums: A tensor. Tokens nums of expand_x. Support dtype: int64, support format: ND.
* @li ep_recv_count: A tensor. Received token nums after dispatching. Support dtype: int32, support format: ND.
* @li tp_recv_count: A tensor. Received token nums after allgather. Support dtype: int32, support format: ND.
* @li expand_scales: A tensor. Scales of each token to sum for combine. Support dtype: float32. Shape supports (A, ), support format: ND.
*/
REG_OP(MoeDistributeDispatchV2)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(expert_ids, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(scales, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(x_active_mask, TensorType({DT_BOOL}))
    .OPTIONAL_INPUT(expert_scales, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(elastic_info, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(performance_info, TensorType({DT_INT64}))
    .OUTPUT(expand_x, TensorType({DT_BF16, DT_INT8, DT_FLOAT16, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .OUTPUT(dynamic_scales, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OUTPUT(assist_info_for_combine, TensorType({DT_INT32}))
    .OUTPUT(expert_token_nums, TensorType({DT_INT64}))
    .OUTPUT(ep_recv_count, TensorType({DT_INT32}))
    .OUTPUT(tp_recv_count, TensorType({DT_INT32}))
    .OUTPUT(expand_scales, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(group_ep, String)
    .REQUIRED_ATTR(ep_world_size, Int)
    .REQUIRED_ATTR(ep_rank_id, Int)
    .REQUIRED_ATTR(moe_expert_num, Int)
    .ATTR(group_tp, String, "")
    .ATTR(tp_world_size, Int, 0)
    .ATTR(tp_rank_id, Int, 0)
    .ATTR(expert_shard_type, Int, 0)
    .ATTR(shared_expert_num, Int, 1)
    .ATTR(shared_expert_rank_num, Int, 0)
    .ATTR(quant_mode, Int, 0)
    .ATTR(global_bs, Int, 0)
    .ATTR(expert_token_nums_type, Int, 1)
    .ATTR(comm_alg, String, "")
    .ATTR(zero_expert_num, Int, 0)
    .ATTR(copy_expert_num, Int, 0)
    .ATTR(const_expert_num, Int, 0)
    .ATTR(y_dtype, Int, 28)
    .OP_END_FACTORY_REG(MoeDistributeDispatchV2)

/**
* @brief MoeDistributeCombineV2 operator interface implementation.

* @par Inputs
* Ten inputs, including:
* @li expand_x: A tensor. Support dtype: float16, bfloat16, int32, dimension must be 2, Support Shape: (A * world_size, H), support format: ND.
* @li expert_ids: A tensor. Support dtype: int32, dimension must be 2, Support Shape: (BS, K), support format: ND.
* @li assist_info_for_combine: A tensor. Support dtype: int32, dimension must be 1, Support Shape: (A * 128), support format: ND.
* @li ep_send_counts: A tensor. Support dtype: int32, Support Shape: (expert_nums + 2 * globalBs * K * server_num, ), support format: ND.
* @li expert_scales: A tensor. Support dtype: float32, Support Shape: (BS, K), support format: ND.
* @li tp_send_counts: A tensor. Support dtype: int32, support format: ND.
* @li x_active_mask: An optional tensor. Support dtype: bool, support format: ND.
* @li activation_scale: An optional tensor. Support dtype: float32, support format: ND.
* @li weight_scale: An optional tensor. Support dtype: float32, support format: ND.
* @li group_list: An optional tensor. Support dtype: int64, support format: ND.
* @li expand_scales: A tensor. Support dtype: float32, Support Shape: (A, ), support format: ND.
* @li shared_expert_x: A tensor. Support dtype: float16, bfloat16, int32, support format: ND.
* @li performance_info: A tensor. Support dtype: int64, support format: ND.

* @par Attributes
* @li group_ep: Input ep comm group name, ep means experts parallelism, dtype: String.
* @li ep_world_size: Input ep comm world size, dtype: Int64.
* @li ep_rank_id: Input ep comm rank Id, dtype: Int64.
* @li moe_expert_num: Input moe expert num, dtype: Int64.
* @li group_tp: Input tp comm group name, tp means tensor parallelism, dtype: String.
* @li tp_world_size: Input tp comm world size, dtype: Int64.
* @li tp_rank_id: Input tp comm rank Id, dtype: Int64.
* @li expert_shard_type: Input moe shard type, dtype: Int64.
* @li shared_expert_num: Input shared expert num, dtype: Int64.
* @li shared_expert_rank_num: Input shared expert rank num, dtype: Int64.
* @li global_bs: Input global batch size, dtype: Int64.
* @li out_dtype: Dtype of output, 0 for bfloat16, 1 for float16, dtype: Int64.
* @li comm_quant_mode: communication quantization mode, 1 for enable, 0 for disable, dtype: Int64.
* @li group_list_type: type of input group_list, dtype: Int64.
* @li comm_alg: Input comm alg type, dtype: String.

* @par Outputs
* One outputs, including:
* @li x: A tensor. Result of combine. Support dtype: float16,bfloat16,  Support Shape: (BS, H), support format: ND.
*/
REG_OP(MoeDistributeCombineV2)
    .INPUT(expand_x, TensorType({DT_BF16, DT_FLOAT16, DT_INT32}))
    .INPUT(expert_ids, TensorType({DT_INT32}))
    .INPUT(assist_info_for_combine, TensorType({DT_INT32}))
    .INPUT(ep_send_counts, TensorType({DT_INT32}))
    .INPUT(expert_scales, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(tp_send_counts, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(x_active_mask, TensorType({DT_BOOL}))
    .OPTIONAL_INPUT(activation_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(weight_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(group_list, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(expand_scales, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(shared_expert_x, TensorType({DT_BF16, DT_FLOAT16, DT_INT32}))
    .OPTIONAL_INPUT(elastic_info, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(ori_x, TensorType({DT_BF16, DT_FLOAT16, DT_INT32}))
    .OPTIONAL_INPUT(const_expert_alpha_1, TensorType({DT_BF16, DT_FLOAT16, DT_INT32}))
    .OPTIONAL_INPUT(const_expert_alpha_2, TensorType({DT_BF16, DT_FLOAT16, DT_INT32}))
    .OPTIONAL_INPUT(const_expert_v, TensorType({DT_BF16, DT_FLOAT16, DT_INT32}))
    .OPTIONAL_INPUT(performance_info, TensorType({DT_INT64}))
    .OUTPUT(x, TensorType({DT_BF16, DT_FLOAT16}))
    .REQUIRED_ATTR(group_ep, String)
    .REQUIRED_ATTR(ep_world_size, Int)
    .REQUIRED_ATTR(ep_rank_id, Int)
    .REQUIRED_ATTR(moe_expert_num, Int)
    .ATTR(group_tp, String, "")
    .ATTR(tp_world_size, Int, 0)
    .ATTR(tp_rank_id, Int, 0)
    .ATTR(expert_shard_type, Int, 0)
    .ATTR(shared_expert_num, Int, 1)
    .ATTR(shared_expert_rank_num, Int, 0)
    .ATTR(global_bs, Int, 0)
    .ATTR(out_dtype, Int, 0)
    .ATTR(comm_quant_mode, Int, 0)
    .ATTR(group_list_type, Int, 0)
    .ATTR(comm_alg, String, "")
    .ATTR(zero_expert_num, Int, 0)
    .ATTR(copy_expert_num, Int, 0)
    .ATTR(const_expert_num, Int, 0)
    .OP_END_FACTORY_REG(MoeDistributeCombineV2)

/**
* @brief DistributeBarrier operator interface implementation.

* @par Inputs
* Three inputs, including:
* @li x_ref: An optional tensor, reserved. Support dtype:bfloat16, float16, float32, bool, int8, int16, int32, int64, uint8, uint16, uint32, uint64. Support format: ND.
* @li time_out: An optional tensor. Support dtype:int32. Support format: ND.
* @li elastic_info: An optional tensor. Support dtype:int32. Support format: ND.

* @par Attributes
* @li group: Required. Input comm group name, means experts parallelism, dtype: String.
* @li world_size: Required. Input comm world size, dtype: int64.

* @par Outputs
* One outputs, including:
* @li x_ref: A tensor. reserved. Support dtype:bfloat16, float16, float32, bool, int8, int16, int32, int64, uint8, uint16, uint32, uint64. Support format: ND.
*/
REG_OP(DistributeBarrier)
    .INPUT(x_ref, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_BOOL, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .OPTIONAL_INPUT(time_out, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(elastic_info, TensorType({DT_INT32}))
    .OUTPUT(x_ref, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_BOOL, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .REQUIRED_ATTR(group, String)
    .REQUIRED_ATTR(world_size, Int)
    .OP_END_FACTORY_REG(DistributeBarrier)

/**
*@brief Data Scanning Operator for the FFN Side in Decoupled Attention-FFN Deployment Scenarios，This operator polls data from AttentionToFFN to confirm its readiness state.
*@par Inputs:
*@li  schedule_context: A tensor. The type support int8. Its shape must be 1D (1024). Format: ND

*@par Attributes
*@li sync_group_size: Sessions handled per group, with a default value of 1. The type support int32.
*@li execute_mode: Kernel execution mode. Currently only supports value 0. The type support int32.

*@par Outputs:
*schedule_context: A ND Tensor that holds the new value of schedule_context after the value has been assigned.
*Has the same shape and dtype and format as the input "schedule_context". \n
*/
REG_OP(FfnWorkerScheduler)
    .INPUT(schedule_context, TensorType({DT_INT8}))
    .OUTPUT(schedule_context, TensorType({DT_INT8}))
    .ATTR(sync_group_size, Int, 1)
    .ATTR(execute_mode, Int, 0) // 1: loop, 0: once
    .OP_END_FACTORY_REG(FfnWorkerScheduler)

/**
* @brief MoeDistributeCombineAddRmsNorm operator interface implementation.

* @par Inputs
* Fourteen inputs, including:
* @li expand_x: A tensor. The characteristics of the extended token based on expandIds. Support dtype: bfloat16, dimension must be 2, Support Shape: (A * world_size, H), support format: ND.
* @li expert_ids: A tensor. The topK expert index for each token. Support dtype: int32, dimension must be 2, Support Shape: (BS, K), support format: ND.
* @li assist_info_for_combine: A tensor. An output of MoeDistributeDispatch. Support dtype: int32, dimension must be 1, Support Shape: (A * 128), support format: ND.
* @li ep_send_counts: A tensor. An output of MoeDistributeDispatch. Support dtype: int32, Support Shape: (expert_nums + 2 * globalBs * K * server_num, ), support format: ND.
* @li expert_scales: A tensor. The weight of the topK experts for each token. Support dtype: float32, Support Shape: (BS, K), support format: ND.
* @li residual_x: A tensor. The right matrix of the add in AddRmsNorm. Support dtype: bfloat16, Support Shape: (BS, 1, H), support format: ND.
* @li gamma: A tensor. An input of addRmsNorm. Support dtype: bfloat16, Support Shape: (H), support format: ND.
* @li tp_send_counts: A tensor. The output tpRecvCounts in MoeDistributeDispatch. Support dtype: int32, support format: ND.
* @li x_active_mask: An optional tensor. Whether the token participates in the communication. True: participate, False: do not participate. Support dtype: bool, support format: ND.
* @li activation_scale: An optional tensor. Support dtype: float32, support format: ND. It's not supported yet.
* @li weight_scale: An optional tensor. Support dtype: float32, support format: ND. It's not supported yet.
* @li group_list: An optional tensor. Support dtype: int64, support format: ND. It's not supported yet.
* @li expand_scales: A tensor. An output of MoeDistributeDispatch. Support dtype: float32, Support Shape: (A, ), support format: ND.
* @li shared_expert_x: A tensor. Share the tokens calculated by experts. Support dtype: bfloat16, support format: ND.

* @par Attributes
* @li group_ep: Input ep comm group name, ep means experts parallelism, dtype: String. Support string length [1, 128), could not bee the same with group_tp.
* @li ep_world_size: Input ep comm world size, dtype: Int64. Support value range: [1, 384].
* @li ep_rank_id: Input ep comm rank Id, dtype: Int64. Support value range: [0, ep_world_size), could not be duplicated within the same EP domain.
* @li moe_expert_num: Input moe expert num, dtype: Int64. Support value range: [0, 512).
* @li group_tp: Input tp comm group name, tp means tensor parallelism, dtype: String. Support string length [1, 128), could not bee the same with group_ep. Default value: "".
* @li tp_world_size: Input tp comm world size, dtype: Int64. Support value range: [0, 2]. Default value: 0.
* @li tp_rank_id: Input tp comm rank Id, dtype: Int64. Support value range: [0, tp_world_size), could not be duplicated within the same TP domain. Default value: 0.
* @li expert_shard_type: Input moe shard type, dtype: Int64. Only support 0 now. Default value: 0.
* @li shared_expert_num: Input shared expert num, dtype: Int64. Support value range: [0, 4].D efault value: 1.
* @li shared_expert_rank_num: Input shared expert rank num, dtype: Int64. Support value range: [0, ep_world_size - 1). Default value: 0.
* @li global_bs: Input global batch size, dtype: Int64. When bs is the same on all the ranks, globalBs = bs * ep_world_size or globalBs = 0;
      when bs is not the same on all the ranks, globalBs = maxBs * ep_world_size. Default value: 0.
* @li out_dtype: Dtype of output, 0 for bfloat16, 1 for float16, dtype: Int64. Default value: 0. It's not supported yet.
* @li comm_quant_mode: Communication quantization mode, 1 for enable, 0 for disable, dtype: Int64. Default value: 0. It's not supported yet.
* @li group_list_type: Type of input group_list, dtype: Int64. Default value: 0. It's not supported yet.
* @li comm_alg: Input comm alg type, dtype: String. Default value: "0". It's not supported yet.
* @li norm_eps: Used to prevent AddRmsNorm by 0 error, dtype:Float. Default value: 1e-6.

* @par Outputs
* Three outputs, including:
* @li y: A tensor. Result of (combine - Add) + Add + RmsNorm. Support dtype: bfloat16,  Support Shape: (BS, 1, H), support format: ND.
* @li rstd: A tensor. Result of standard deviation. Support dtype: float32,  Support Shape: (BS, 1, 1), support format: ND.
* @li x: A tensor. Result of (combine - Add) + Add. Support dtype: bfloat16,  Support Shape: (BS, 1, H), support format: ND.
*/
REG_OP(MoeDistributeCombineAddRmsNorm)
    .INPUT(expand_x, TensorType({DT_BF16}))
    .INPUT(expert_ids, TensorType({DT_INT32}))
    .INPUT(assist_info_for_combine, TensorType({DT_INT32}))
    .INPUT(ep_send_counts, TensorType({DT_INT32}))
    .INPUT(expert_scales, TensorType({DT_FLOAT}))
    .INPUT(residual_x, TensorType({DT_BF16}))
    .INPUT(gamma, TensorType({DT_BF16}))
    .OPTIONAL_INPUT(tp_send_counts, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(x_active_mask, TensorType({DT_BOOL}))
    .OPTIONAL_INPUT(activation_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(weight_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(group_list, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(expand_scales, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(shared_expert_x, TensorType({DT_BF16}))
    .OPTIONAL_INPUT(elastic_info, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(ori_x, TensorType({DT_BF16}))
    .OPTIONAL_INPUT(const_expert_alpha_1, TensorType({DT_BF16}))
    .OPTIONAL_INPUT(const_expert_alpha_2, TensorType({DT_BF16}))
    .OPTIONAL_INPUT(const_expert_v, TensorType({DT_BF16}))
    .OUTPUT(y, TensorType({DT_BF16}))
    .OUTPUT(rstd, TensorType({DT_FLOAT}))
    .OUTPUT(x, TensorType({DT_BF16}))
    .REQUIRED_ATTR(group_ep, String)
    .REQUIRED_ATTR(ep_world_size, Int)
    .REQUIRED_ATTR(ep_rank_id, Int)
    .REQUIRED_ATTR(moe_expert_num, Int)
    .ATTR(group_tp, String, "")
    .ATTR(tp_world_size, Int, 0)
    .ATTR(tp_rank_id, Int, 0)
    .ATTR(expert_shard_type, Int, 0)
    .ATTR(shared_expert_num, Int, 1)
    .ATTR(shared_expert_rank_num, Int, 0)
    .ATTR(global_bs, Int, 0)
    .ATTR(out_dtype, Int, 0)
    .ATTR(comm_quant_mode, Int, 0)
    .ATTR(group_list_type, Int, 0)
    .ATTR(comm_alg, String, "")
    .ATTR(norm_eps, Float, 1e-6f)
    .ATTR(zero_expert_num, Int, 0)
    .ATTR(copy_expert_num, Int, 0)
    .ATTR(const_expert_num, Int, 0)
    .OP_END_FACTORY_REG(MoeDistributeCombineAddRmsNorm)

/**
 * @brief MoeUpdateExpert operator interface implementation.
 *
 * @par Inputs:
 * @li expert_ids: A tensor. The topK expert index for each token. Support dtype: int32, int64. dimension must be 2, support format: ND.
 * @li eplb_table: A tensor. Mapping table of logical rank_id to physical rank_id. Support dtype: int32. dimension must be 2, support format: ND.
 * @li expert_scales: A tensor. Scales for top k experts for each token. Support dtype: float16, bfloat16, float32. dimension must be 2, support format: ND.
 * @li pruning_threshold: A tensor. Threshold for expert scales. Support dtype: float32. dimension must be 1 or 2, support format: ND.
 * @li active_mask: A tensor. Indicates if token is involved in communication. Support dtype: bool. dimension must be 1, support format: ND.
 *
 * @par Attributes:
 * @li local_rank_id: Required. The rank id in the current communication domain. dtype: Int64.
 * @li world_size: Required. The num of physical rank in the current communication domain. dtype: Int64.
 * @li balance_mode: Optional. Balanced by rank(0) or by token(1), and the current default value is 0. dtype: Int64
 *
 * @par Output:
 * balance_expert_ids: A tensor. Convert the logical rank_id in expert_ids to physical rank_id using the eplb_table. Support dtype: int32, int64. Support format: ND.
 * balanced_active_mask: A tensor. Get balanced active mask through tailored expert scales and pruning threshold. Support dtype: bool. Support format: ND.
 */
REG_OP(MoeUpdateExpert)
      .INPUT(expert_ids, TensorType({DT_INT32, DT_INT64}))
      .INPUT(eplb_table, TensorType({DT_INT32}))
      .OPTIONAL_INPUT(expert_scales, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
      .OPTIONAL_INPUT(pruning_threshold, TensorType({DT_FLOAT}))
      .OPTIONAL_INPUT(active_mask, TensorType({DT_BOOL}))
      .OUTPUT(balanced_expert_ids, TensorType({DT_INT32, DT_INT64}))
      .OUTPUT(balanced_active_mask, TensorType({DT_BOOL}))
      .ATTR(local_rank_id, Int, -1)
      .ATTR(world_size, Int, -1)
      .ATTR(balance_mode, Int, 0)
      .OP_END_FACTORY_REG(MoeUpdateExpert)

/**
 * @brief GroupedMatmulSwigluQuantV2 operator interface implementation.
 *
 * @par Inputs:
 * @li x: A tensor. The format supports ND. The type supports int8, float8_e5m2, float8_e4m3fn, float4_e2m1, hifloat8.
 * @li x_scale: A tensor. Scales for x for each token. The format supports ND. The type supports float32, float8_e8m0.
 * @li group_list: A tensor. Indicates the matmul size distribution along separated dimension.
 * The format supports ND. The type supports int64. The first dimension of group_list should not be greater than 1024, meaning it supports up to 1024 groups.
 * @li weight: A tensor List. The format supports ND/NZ. The type supports int8, float8_e5m2, float8_e4m3fn, float4_e2m1, hifloat8.
 * @li weight_scale: A tensor List. Format supports ND. The type supports float32, float8_e8m0, bfloat16, float16.
 * @li weight_assist_matrix: A tensor List. The format supports ND. The type supports float32.
 * @li bias: A tensor. The format supports ND. The type supports float32.
 * @li smooth_scale: A tensor. The format supports ND. The type supports float32.
 *
 * @par Attributes:
 * @li dequant_mode: Optional. Indicates the mode of dequant. dtype: Int64. Default: 0. Reserved parameter.
 * @li dequant_dtype: Optional. Indicates the dtype of dequant. dtype: Int64. Default: 0. Reserved parameter.
 * @li quant_mode: Optional. Indicates the mode of quant. dtype: Int64. Default: 0. Reserved parameter.
 * @li quant_dtype: Optional. Indicates the dtype of quant. dtype: Int64. Default: 0. Reserved parameter.
 * @li transpose_weight: A bool. Default: false. Reserved parameter.
 * @li group_list_type: An int. Indicates whether the value in group_list is cumsum or count.
 * 0, value in group_list is cumsum. 1, value in group_list is count. Default: 0.
 * @li tuning_config: A ListInt. The first element in the array indicates the expected number of tokens processed by each expert.
 * The operator tiling performs optimal tiling based on the first element in the array. Default: {0}.
 *
 * @par Output:
 * @li y: A tensor. The type supports int8, float8_e5m2, float8_e4m3fn, float4_e2m1, hifloat8. The format supports ND.
 * @li y_scale: A tensor. The type supports float32, float8_e8m0. The format supports ND.

*\n
*\n
* - The following are the supported format, data types and shapes(for Ascend 950 AI Processor):
*\n
| Tensor    | x                         | x_scale             | group_list | weight                    | weight_scale                                    | y                                                    | y_scale                  |
| :-------: | :-----------------------: | :-----------------: | :--------: | :-----------------------: | :---------------------------------------------: | :--------------------------------------------------: |:-----------------------: |
| Format    | ND                        | ND                  | ND         | ND                        | ND                                              | ND                                                   | ND                       |
| Dtype     | float8_e4m3fn/float8_e5m2 | float8_e8m0         | int64      | float8_e4m3fn/float8_e5m2 | float8_e8m0                                     | float8_e4m3fn/float8_e5m2                            | float8_e8m0              |
| Shape     | (M,K)                     | (M,ceil(K/64),2)    | (B,)       | [(B,K,N)]/[(B,N,K)]       | [(B,ceil(K/64),N,2)]/[(B,N,ceil(K/64),2)]       | (M,N/2)                                              | (M,ceil((N/2)/64),2)     |
| Dtype     | float4_e2m1               | float8_e8m0         | int64      | float4_e2m1               | float8_e8m0                                     | float8_e4m3fn/float8_e5m2/float4_e2m1                | float8_e8m0              |
| Shape     | (M,K)                     | (M,ceil(K/64),2)    | (B,)       | [(B,K,N)]/[(B,N,K)]       | [(B,ceil(K/64),N,2)]/[(B,N,ceil(K/64),2)]       | (M,N/2)                                              | (M,ceil((N/2)/64),2)     |
| Dtype     | int8                      | float32             | int64      | int8                      | bf16/float32/float16                            | int8                                                 | float32                  |
| Shape     | (M,K)                     | (M,)                | (B,)       | [(B,K,N)]/[(B,N,K)]       | (B,N,2)                                         | (M,N/2)                                              | (M,)                     |
| Dtype     | float8_e4m3fn/float8_e5m2 | float32             | int64      | float8_e4m3fn/float8_e5m2 | bf16/float32                                    | float8_e4m3fn/float8_e5m2                            | float32                  |
| Shape     | (M,K)                     | (M,)                | (B,)       | [(B,K,N)]/[(B,N,K)]       | (B,N,2)                                         | (M,N/2)                                              | (M,)                     |
| Dtype     | hifloat8                  | float32             | int64      | hifloat8                  | bf16/float32                                    | hifloat8                                             | float32                  |
| Shape     | (M,K)                     | (M,)                | (B,)       | [(B,K,N)]/[(B,N,K)]       | (B,N,2)                                         | (M,N/2)                                              | (M,)                     |
*\n
*\n
* @attention Constraints:
* @li The value of N must be 128 aligned, and this constraint applies only to the Ascend 950 AI Processor.
* @li When group_list_type is 0, group_list must be a non-negative monotone non-decreasing array. The last value in group_list must be no greater than the first dimension of tensor x1;
* When group_list_type is 1, it must be a non-negative array. Sum of values in group_list must be no greater than the first dimension of tensor x1. \n
*/
REG_OP(GroupedMatmulSwigluQuantV2)
      .INPUT(x, TensorType({DT_INT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_HIFLOAT8}))
      .INPUT(x_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
      .INPUT(group_list, TensorType{DT_INT64})
      .DYNAMIC_INPUT(weight, TensorType({DT_INT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_HIFLOAT8}))
      .DYNAMIC_INPUT(weight_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0, DT_BF16, DT_FLOAT16}))
      .DYNAMIC_INPUT(weight_assist_matrix, TensorType{DT_FLOAT})
      .OPTIONAL_INPUT(bias, TensorType{DT_FLOAT})
      .OPTIONAL_INPUT(smooth_scale, TensorType{DT_FLOAT})
      .ATTR(dequant_mode, Int, 0)
      .ATTR(dequant_dtype, Int, 0)
      .ATTR(quant_mode, Int, 0)
      .ATTR(quant_dtype, Int, 0)
      .ATTR(transpose_weight, Bool, false)
      .ATTR(group_list_type, Int, 0)
      .ATTR(tuning_config, ListInt, {0})
      .OUTPUT(y, TensorType({DT_INT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_HIFLOAT8}))
      .OUTPUT(y_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
      .OP_END_FACTORY_REG(GroupedMatmulSwigluQuantV2)

/**
 * @brief Fusion op of quant all reduce.
 * @par Inputs:
 * two inputs, including:
 * @li x: A matrix tensor. The type support int8, hifloat8, float8_e4m3fn, float8_e5m2, float4_e1m2, float4_e2m1. The format supports ND.
 * @li scale: A matrix tensor. The type support float32, float8_e8m0. The format supports ND.
 *
 * @par Outputs:
 * out_put: A matrix tensor. The type support float16, bfloat16, float32. The format supports ND.
 *
 * @par Attributes:
 * @li group: A required string identifying the group of ranks participating in the op.
 * @li reduce_op: An optional string identifying the reduction operation to perform. Default: "sum".
 * @li output_dtype: An optional int identifying the data type of output. The type support 0(float), 1(float16), 27(bfloat16). Default: 27(bfloat16).
 * @li world_size: A required int identifying the rank size.
 */
REG_OP(QuantAllReduce)
    .INPUT(x, "T1")
    .INPUT(scales, "T2")
    .OUTPUT(out_put, "T3")
    .DATATYPE(T1, TensorType({DT_INT8, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E1M2, DT_FLOAT4_E2M1}))
    .DATATYPE(T2, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .DATATYPE(T3, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .REQUIRED_ATTR(group, String)
    .ATTR(reduce_op, String, "sum")
    .ATTR(output_dtype, Int, DT_BF16)
    .REQUIRED_ATTR(world_size, Int)
    .OP_END_FACTORY_REG(QuantAllReduce)

/**
 * @brief Fusion op of quant and reduce scatter.
 * @par Inputs:
 * two inputs, including:
 * @li x: A matrix tensor. The type support int8, hifloat8, float8_e4m3fn, float8_e5m2, float4_e1m2, float4_e2m1. The format supports ND.
 * @li scale: A matrix tensor. The type support float32, float8_e8m0. The format supports ND.
 *
 * @par Outputs:
 * out_put: A matrix tensor. The type support float16, bfloat16, float32. The format supports ND.
 *
 * @par Attributes:
 * @li group: A required string identifying the group of ranks participating in the op.
 * @li reduce_op: An optional string identifying the reduction operation to perform. Default: "sum".
 * @li output_dtype: An optional int identifying the data type of output. The type support 0(float), 1(float16), 27(bfloat16). Default: 27(bfloat16).
 * @li world_size: A required int identifying the rank size.
 */
REG_OP(QuantReduceScatter)
    .INPUT(x, "T1")
    .INPUT(scales, "T2")
    .OUTPUT(out_put, "T3")
    .DATATYPE(T1, TensorType({DT_INT8, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E1M2, DT_FLOAT4_E2M1}))
    .DATATYPE(T2, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .DATATYPE(T3, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .REQUIRED_ATTR(group, String)
    .ATTR(reduce_op, String, "sum")
    .ATTR(output_dtype, Int, DT_BF16)
    .REQUIRED_ATTR(world_size, Int)
    .OP_END_FACTORY_REG(QuantReduceScatter)

REG_OP(AttentionToFFN)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(session_id, TensorType({DT_INT32}))
    .INPUT(micro_batch_id, TensorType({DT_INT32}))
    .INPUT(layer_id, TensorType({DT_INT32}))
    .INPUT(expert_ids, TensorType({DT_INT32}))
    .INPUT(expert_rank_table, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(scales, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(active_mask, TensorType({DT_BOOL}))
    .REQUIRED_ATTR(group, String)
    .REQUIRED_ATTR(world_size, Int)
    .REQUIRED_ATTR(ffn_token_info_table_shape, ListInt)
    .REQUIRED_ATTR(ffn_token_data_shape, ListInt)
    .REQUIRED_ATTR(attn_token_info_table_shape, ListInt)
    .REQUIRED_ATTR(moe_expert_num, Int)
    .ATTR(quant_mode, Int, 0)
    .ATTR(sync_flag, Int, 0)
    .ATTR(ffn_start_rank_id, Int, 0)
    .OP_END_FACTORY_REG(AttentionToFFN)

REG_OP(FFNToAttention)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(session_ids, TensorType({DT_INT32}))
    .INPUT(micro_batch_ids, TensorType({DT_INT32}))
    .INPUT(token_ids, TensorType({DT_INT32}))
    .INPUT(expert_offsets, TensorType({DT_INT32}))
    .INPUT(actual_token_num, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(attn_rank_table, TensorType({DT_INT32}))
    .REQUIRED_ATTR(group, String)
    .REQUIRED_ATTR(world_size, Int)
    .REQUIRED_ATTR(token_info_table_shape, ListInt)
    .REQUIRED_ATTR(token_data_shape, ListInt)
    .OP_END_FACTORY_REG(FFNToAttention)

/**
* @brief Function KvQuantSparseFlashAttention.

* @par Inputs:
* @li query: A matrix tensor. The type support float16, bfloat16.
* Query for attention structure.
* @li key: A matrix tensor. The type support int8, fp8_e4m3fn, hifloat8.
* Key for attention structure.
* @li value: A matrix tensor. The type support int8, fp8_e4m3fn, hifloat8.
* Value for attention structure.
* @li sparse_indices: A matrix tensor. The type support int32.
* The indices of sparse kv cache.
* @li key_dequant_scale: A matrix tensor. The type support float32.
* The dequantization factor of key.
* @li value_dequant_scale: A matrix tensor. The type support float32.
* The dequantization factor of value.
* @li block_table: A matrix tensor. The type support int32.
* The block mapping table used in KV storage of PageAttention.
* @li actual_seq_lengths_query: A matrix tensor. The type support int32.
* Efective sequence length of query in different batches.
* @li actual_seq_lengths_kv: A matrix tensor. The type support int32.
* Effective sequence length of key/value in different batches.

* @par Attributes:
* @li scale_value: A float. The scale value.
* @li key_quant_mode: An int. Antiquantization mode of key.
* - 0: per-channel
* - 1: per-token+per-head
* - 2: per-tile
* @li value_quant_mode: An int. Antiquantization mode of value. The mode number is the same as key_quant_mode.
* @li sparse_block_size: An int. The block size in the sparse phase. Default: 1.
* @li layout_query: A string. Specifies the layout of `query`, the value must be one of ["BSND", "TND"]. Default: "BSND".
* @li layout_kv: A string. Specifies the layout of `key/value`, the value must be one of ["BSND", "TND", "PA_BSND"]. Default: "BSND".
* @li sparse_mode: Sparse mode. Default: 3.
* - 0: default mask
* - 3: rightDownCausal make
* - 4: band mask
* @li pre_tokens: An int. Previous tokens. Default: 9223372036854775807.
* @li next_tokens: An int. Next tokens. Default: 9223372036854775807.
* @li attention_mode: An int. Attention mode, 0: GQA/MHA; 1: MLA-naive; 2: MLA-absorb. Default: 0.
* @li quant_scale_repo_mode: An int. Quant scale repo mode, 0: separate; 1: combine. Default: 1.
* @li tile_size: An int. Tile size. Default: 128.
* @li rope_head_dim: An int. Rope head dim. Default: 64.

* @par Outputs:
* @li attention_out: A matrix tensor. The type support float16, bfloat16.
* The output of attention structure.
*/
REG_OP(KvQuantSparseFlashAttention)
    .INPUT(query, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(key, TensorType({DT_INT8, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .INPUT(value, TensorType({DT_INT8, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .INPUT(sparse_indices, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(key_dequant_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(value_dequant_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(block_table, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(actual_seq_lengths_query, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(actual_seq_lengths_kv, TensorType({DT_INT32}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(scale_value, Float)
    .REQUIRED_ATTR(key_quant_mode, Int)
    .REQUIRED_ATTR(value_quant_mode, Int)
    .ATTR(sparse_block_size, Int, 1)
    .ATTR(layout_query, String, "BSND")
    .ATTR(layout_kv, String, "BSND")
    .ATTR(sparse_mode, Int, 3)
    .ATTR(pre_tokens, Int, 9223372036854775807)
    .ATTR(next_tokens, Int, 9223372036854775807)
    .ATTR(attention_mode, Int, 0)
    .ATTR(quant_scale_repo_mode, Int, 1)
    .ATTR(tile_size, Int, 128)
    .ATTR(rope_head_dim, Int, 64)
    .OP_END_FACTORY_REG(KvQuantSparseFlashAttention)

/**
* @brief Function LightningIndexer.

* @par Inputs:
* @li query: A matrix tensor. The type support float16, bfloat16.
* Query for attention structure.
* @li key: A matrix tensor. The type support float16, bfloat16.
* Key for attention structure.
* @li weights: A matrix tensor. The type support float16, bfloat16.
* Weights for attention structure.
* @li actual_seq_lengths_query: A matrix tensor. The type support int32.
* Efective sequence length of query in different batches.
* @li actual_seq_lengths_key: A matrix tensor. The type support int32.
* Effective sequence length of key in different batches.
* @li block_table: A matrix tensor. The type support int32.
* The block mapping table used in KV storage of PageAttention.

* @par Attributes:
* @li layout_query: A string. Specifies the layout of `query`, the value must be one of ["BSND", "TND"]. Default: "BSND".
* @li layout_key: A string. Specifies the layout of `key`, the value must be one of ["BSND", "TND", "PA_BSND"]. Default: "BSND".
* @li sparse_count: An int. The sparse count. Default: 2048.
* @li sparse_mode: Sparse mode. Default: 3.
* - 0: default mask
* - 3: rightDownCausal make
* - 4: band mask
* @li pre_tokens: An int. Previous tokens. Default: 9223372036854775807.
* @li next_tokens: An int. Next tokens. Default: 9223372036854775807.
* @li return_values: An bool. Need return sparse value. Default: false.

* @par Outputs:
* @li sparse_indices: A matrix tensor. The type support int32.
* The indices of sparse kv cache.
* @li sparse_values: A matrix tensor. The type support float16, bfloat16.
* The values of sparse kv cache.
*/
REG_OP(LightningIndexer)
    .INPUT(query, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(key, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(weights, TensorType({DT_BF16, DT_FLOAT16}))
    .OPTIONAL_INPUT(actual_seq_lengths_query, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(actual_seq_lengths_key, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(block_table, TensorType({DT_INT32}))
    .OUTPUT(sparse_indices, TensorType({DT_INT32}))
    .OUTPUT(sparse_values, TensorType({DT_BF16, DT_FLOAT16}))
    .ATTR(layout_query, String, "BSND")
    .ATTR(layout_key, String, "BSND")
    .ATTR(sparse_count, Int, 2048)
    .ATTR(sparse_mode, Int, 3)
    .ATTR(pre_tokens, Int, 9223372036854775807)
    .ATTR(next_tokens, Int, 9223372036854775807)
    .ATTR(return_values, Bool, false)
    .OP_END_FACTORY_REG(LightningIndexer)

/**
* @brief Function QuantLightningIndexer.

* @par Inputs:
* @li query: A matrix tensor. The type support int8, fp8_e4m3fn, hifloat8.
* Query for attention structure.
* @li key: A matrix tensor. The type support int8, fp8_e4m3fn, hifloat8.
* Key for attention structure.
* @li weights: A matrix tensor. The type support float16, bfloat16.
* Weights for attention structure.
* @li query_dequant_scale:  A matrix tensor. The type support float16, float32.
* The dequantization factor of query.
* @li key_dequant_scale:  A matrix tensor. The type support float16, float32.
* The dequantization factor of key.
* @li actual_seq_lengths_query: A matrix tensor. The type support int32.
* Efective sequence length of query in different batches.
* @li actual_seq_lengths_key: A matrix tensor. The type support int32.
* Effective sequence length of key in different batches.
* @li block_table: A matrix tensor. The type support int32.
* The block mapping table used in KV storage of PageAttention.

* @par Attributes:
* @li query_quant_mode: An int. Antiquantization mode of query.
* - 0: per-token+per-head
* @li key_quant_mode: An int. Antiquantization mode of key. The mode number is the same as key_quant_mode.
* @li layout_query: A string. Specifies the layout of `query`, the value must be one of ["BSND", "TND"]. Default: "BSND".
* @li layout_key: A string. Specifies the layout of `key`, the value must be one of ["BSND", "TND", "PA_BSND"]. Default: "BSND".
* @li sparse_count: An int. The sparse count. Default: 2048.
* @li sparse_mode: Sparse mode. Default: 3.
* - 0: default mask
* - 3: rightDownCausal make
* - 4: band mask
* @li pre_tokens: An int. Previous tokens. Default: 9223372036854775807.
* @li next_tokens: An int. Next tokens. Default: 9223372036854775807.

* @par Outputs:
* @li sparse_indices: A matrix tensor. The type support int32.
* The indices of sparse kv cache.
*/
REG_OP(QuantLightningIndexer)
    .INPUT(query, TensorType({DT_INT8, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .INPUT(key, TensorType({DT_INT8, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .INPUT(weights, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(query_dequant_scale, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(key_dequant_scale, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(actual_seq_lengths_query, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(actual_seq_lengths_key, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(block_table, TensorType({DT_INT32}))
    .OUTPUT(sparse_indices, TensorType({DT_INT32}))
    .REQUIRED_ATTR(query_quant_mode, Int)
    .REQUIRED_ATTR(key_quant_mode, Int)
    .ATTR(layout_query, String, "BSND")
    .ATTR(layout_key, String, "BSND")
    .ATTR(sparse_count, Int, 2048)
    .ATTR(sparse_mode, Int, 3)
    .ATTR(pre_tokens, Int, 9223372036854775807)
    .ATTR(next_tokens, Int, 9223372036854775807)
    .OP_END_FACTORY_REG(QuantLightningIndexer)

/**
* @brief Function SparseFlashAttention.
*
* @par Inputs:
* @li query: A matrix tensor. The type support float16, bfloat16.
* Query for attention structure.
* @li key: A matrix tensor. The type support float16, bfloat16.
* Key for attention structure.
* @li value: A matrix tensor. The type support float16, bfloat16.
* Value for attention structure.
* @li sparse_indices: A matrix tensor. The type support int32.
* The indices of sparse kv cache.
* @li block_table: A matrix tensor. The type support int32.
* The block mapping table used in KV storage of PageAttention.
* @li actual_seq_lengths_query: A matrix tensor. The type support int32.
* Efective sequence length of query in different batches.
* @li actual_seq_lengths_kv: A matrix tensor. The type support int32.
* Effective sequence length of key and value in different batches.
* @li query_rope: A tensor. The type support float16, bfloat16.
* @li key_rope: A tensor. The type support float16, bfloat16.
*
* @par Attributes:
* @li scale_value: A float. A required attribute.
* @li sparse_block_size: An int. An optional attribute. Max value: 64. Default: 1.
* @li layout_query: A string. An optional attribute. Specifies the layout of `query`, the value must be one of ["BSND", "TND"]. Default: "BSND".
* @li layout_kv: A string. An optional attribute. Specifies the layout of `key` and 'value', the value must be one of ["BSND", "TND", "PA_BSND"]. Default: "BSND".
* @li sparse_mode: An int. Sparse mode. Default: 3.
* - 0: default mask
* - 3: rightDownCausal make
* - 4: band mask
* @li pre_tokens: An int. Previous tokens. Default: 9223372036854775807.
* @li next_tokens: An int. Next tokens. Default: 9223372036854775807.
* @li attention_mode: An int. An optional attribute. Default: 0.
* @li return_softmax_lse: A bool. An optional attribute. Default: false.
*
* @par Outputs:
* @li attention_out: A matrix tensor. The type support float16, bfloat16.
* @li softmax_max: A matrix tensor. The type support float32.
* @li softmax_sum: A matrix tensor. The type support float32.
*/
REG_OP(SparseFlashAttention)
    .INPUT(query, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(key, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(value, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(sparse_indices, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(block_table, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(actual_seq_lengths_query, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(actual_seq_lengths_kv, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(query_rope, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(key_rope, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(softmax_max, TensorType({DT_FP32}))
    .OUTPUT(softmax_sum, TensorType({DT_FP32}))
    .REQUIRED_ATTR(scale_value, Float)
    .ATTR(sparse_block_size, Int, 1)
    .ATTR(layout_query, String, "BSND")
    .ATTR(layout_kv, String, "BSND")
    .ATTR(sparse_mode, Int, 3)
    .ATTR(pre_tokens, Int, 9223372036854775807)
    .ATTR(next_tokens, Int, 9223372036854775807)
    .ATTR(attention_mode, Int, 0)
    .ATTR(return_softmax_lse, Bool, false)
    .OP_END_FACTORY_REG(SparseFlashAttention)

/**
 * @brief Fusion op of matmul and alltoall.
 * @par Inputs:
 * two inputs, including:
 * @li x1: A matrix Tensor. The type support bfloat16, float16, float8_e4m3fn, float8_e5m2. The format supports ND.
 * @li x2: A matrix Tensor. The type support bfloat16, float16, float8_e4m3fn, float8_e5m2. The format supports ND.
 * @li bias: A matrix Tensor. The type support bfloat16, float16, float. The format supports ND.
 * @li x1_scale: A matrix Tensor. The type support float. The format supports ND.
 * @li x2_scale: A matrix Tensor. The type support float. The format supports ND.
 * @li comm_scale: A matrix Tensor. The type support float. The format supports ND.
 * @li x1_offset: A matrix Tensor. The type support float. The format supports ND.
 * @li x2_offset: A matrix Tensor. The type support float, float16. The format supports ND.
 * @par Outputs:
 * @li y: A matrix Tensor. The type support bfloat16, float16, float. The format supports ND.
 * @par Attributes:
 * @li group: A string. Communication domain identifier.
 * @li world_size: An int. Default: -1.
 * @li all2all_axes: An ListInt. Indicate the data direction for All2All communication. Default: {-1, -2}.
 * @li y_dtype: An int. Declare the output dtype. Default: static_cast<int64_t>(ge::DT_UNDEFINED) 为28.
 * @li x1_quant_mode: An int. Quantization mode of x1. Default: 0.
 * @li x2_quant_mode: An int. Quantization mode of x2. Default: 0.
 * @li comm_quant_mode: An int. Quantitative types for communication. Default: 0.
 * @li comm_quant_dtype: An int. Communication accuracy. Default: static_cast<int64_t>(ge::DT_UNDEFINED) 为28.
 * @li transpose_x1: A bool. Whether x1 is transposed. Default: false.
 * @li transpose_x2: A bool. Whether x2 is transposed. Default: false.
 * @li group_size: An int. Default: 0.
 */
REG_OP(MatmulAlltoAll)
    .INPUT(x1, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2}))
    .INPUT(x2, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2}))
    .OPTIONAL_INPUT(bias, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(x1_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(x2_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(comm_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(x1_offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(x2_offset, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(group, String)
    .REQUIRED_ATTR(world_size, Int)
    .ATTR(all2all_axes, ListInt, {-1, -2})
    .ATTR(y_dtype, Int, 28)
    .ATTR(x1_quant_mode, Int, 0)
    .ATTR(x2_quant_mode, Int, 0)
    .ATTR(comm_quant_mode, Int, 0)
    .ATTR(comm_quant_dtype, Int, 28)
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(group_size, Int, 0)
    .OP_END_FACTORY_REG(MatmulAlltoAll)

/**
 * @brief Fusion op of alltoall and matmul.
 * @par Inputs:
 * two inputs, including:
 * @li x1: A matrix Tensor. The type support bfloat16, float16. The format supports ND.
 * @li x2: A matrix Tensor. The type support bfloat16, float16, float8_e4m3fn, float8_e5m2. The format supports ND.
 * @li bias: A matrix Tensor. The type support bfloat16, float16, float. The format supports ND.
 * @li x1_scale: A matrix Tensor. The type support float. The format supports ND.
 * @li x2_scale: A matrix Tensor. The type support float. The format supports ND.
 * @li comm_scale: A matrix Tensor. The type support float. The format supports ND.
 * @li x1_offset: A matrix Tensor. The type support float. The format supports ND.
 * @li x2_offset: A matrix Tensor. The type support float, float16. The format supports ND.
 *
 * @par Outputs:
 * @li y: A matrix Tensor. The type support bfloat16, float16, float. The format supports ND.
 * @li all2all_out: A matrix Tensor. The type support bfloat16, float16. The format supports ND.
 *
 * @par Attributes:
 * @li group: A string. Communication domain identifier.
 * @li world_size: An int. Default: -1.
 * @li all2all_axes: An ListInt. Indicate the data direction for All2All communication. Default: {-2, -1}.
 * @li y_dtype: An int. Declare the output dtype. Default: static_cast<int64_t>(ge::DT_UNDEFINED) 为28.
 * @li x1_quant_mode: An int. Quantization mode of x1. Default: 0.
 * @li x2_quant_mode: An int. Quantization mode of x2. Default: 0.
 * @li comm_quant_mode: An int. Quantitative types for communication. Default: 0.
 * @li x1_quant_dtype: An int. Quantization type of the left matrix input to Matmul after communication. Default: static_cast<int64_t>(ge::DT_UNDEFINED) 为28.
 * @li comm_quant_dtype: An int. Quantization type of x1 input before communication. Default: static_cast<int64_t>(ge::DT_UNDEFINED) 为28.
 * @li transpose_x1: A bool. Whether x1 is transposed. Default: false.
 * @li transpose_x2: A bool. Whether x2 is transposed. Default: false.
 * @li group_size: An int. Default: 0.
 * @li alltoall_out_flag: A bool. Default: true.
 */
REG_OP(AlltoAllMatmul)
    .INPUT(x1, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(x2, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2}))
    .OPTIONAL_INPUT(bias, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(x1_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(x2_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(comm_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(x1_offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(x2_offset, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(all2all_out, TensorType({DT_BF16, DT_FLOAT16}))
    .REQUIRED_ATTR(group, String)
    .REQUIRED_ATTR(world_size, Int)
    .ATTR(all2all_axes, ListInt, {-2, -1})
    .ATTR(y_dtype, Int, 28)
    .ATTR(x1_quant_mode, Int, 0)
    .ATTR(x2_quant_mode, Int, 0)
    .ATTR(comm_quant_mode, Int, 0)
    .ATTR(x1_quant_dtype, Int, 28)
    .ATTR(comm_quant_dtype, Int, 28)
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(group_size, Int, 0)
    .ATTR(alltoall_out_flag, Bool, true)
    .OP_END_FACTORY_REG(AlltoAllMatmul)
}  // namespace ge


#endif // OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_WARN_SHOWN
#endif  // OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_
