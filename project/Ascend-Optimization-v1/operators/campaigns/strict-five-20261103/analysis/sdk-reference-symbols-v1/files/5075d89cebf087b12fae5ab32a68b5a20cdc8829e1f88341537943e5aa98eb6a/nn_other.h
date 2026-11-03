/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file nn_other.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_OTHER_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_OTHER_H_

#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_OTHER_H_WARN_SHOWN
#define OPS_BUILT_IN_OP_PROTO_INC_NN_OTHER_H_WARN_SHOWN
#warning "nn_other.h is scheduled to be deprecated in December 2026, and will be replaced by the \
ops_proto_math.h, ops_proto_cv.h, ops_proto_nn.h, ops_proto_transformer.h, ops_proto_legacy.h. \
We apologize for any inconvenience caused and appreciate your timely migration to the new interface."

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Apply rotary position embedding for a single tensor.
 * @par Inputs:
 * @li x: A 4D tensor which rotary position embedding is applied, format supports ND, and data type must be float16, float or bfloat16.
 * @li cos: A 4D tensor which is "cos" in rotary position embedding, format supports ND, data type must be the same as "x", and shape must be the same as "sin".
 * @li sin: A 4D tensor which is "sin" in rotary position embedding, format supports ND, data type must be the same as "x", and shape must be the same as "cos".
 * @li rotate: An optional 2D tensor which is the transformation matrix for position transformation of the "x" in rotary position embedding, 
 * format supports ND, data type must be the same as "x", both dimensions are the same and equal to the last dimension of "x", rotate does not support back propagation.
 * @par Outputs:
 * y: A 4D tensor which is the result of rotary position embedding, format supports ND, data type must be the same as "x", and shape must be the same as "x".
 * @par Attributes:
 * mode: An optional attribute of type int, specifying the mode of rotary position embedding, must be 0-"half", 1-"interleave", 2-"quarter" or 3-"interleave-half". Defaults to 0.
 * Atlas A2 Training Series Product/ Atlas 800I A2 Inference Product and Atlas A3 Training Series Product only support 0-"half" and 1-"interleave".
 * @attention Constraints:
 * Let (B, S, N, D) represents the shape of the 4-D input "x". Under this representation, the shape constraints of each parameter can be described as follows:
 * @li The D of "x", "cos", "sin", "rotate" and "y" must be equal. For Ascend 950 AI Processor, D should be less or equal to 1024. 
 * For Atlas A2 Training Series Product/ Atlas 800I A2 Inference Product and Atlas A3 Training Series Product, D should be less or equal to 896.
 * @li In half, interleave and interleave-half mode, D must be a multiple of 2. In quarter mode, D must be a multiple of 4.
 * @li B, S, N of "cos" and "sin" must meet one of the following four conditions:
 *  - B, S, N are 1, means the shape is (1, 1, 1, D).
 *  - B, S, N are the same as that of "x", means the shape is (B, S, N, D).
 *  - One of S and N is 1, the remaining one dimension and B are the same as that of "x", means the shape is (B, 1, N, D) or (B, S, 1, D).
 *  - Two of B, S and N are 1, the remaining one dimension is the same as that of "x", means the shape is (1, 1, N, D), (1, S, 1, D) or (B, 1, 1, D).
 *  - The shape for rotate should be (D, D).
 */
REG_OP(RotaryPositionEmbedding)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(cos, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(sin, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OPTIONAL_INPUT(rotate, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .ATTR(mode, Int, 0)
    .OP_END_FACTORY_REG(RotaryPositionEmbedding)

/**
 * @brief Backwards calculation of RotaryPositionEmbedding.
 * @par Inputs:
 * @li dy: A 4D tensor which represents the gradient of output "y" in RotaryPositionEmbedding, format supports ND, and data type must be float16, float or bfloat16.
 * @li cos: A 4D tensor which is input "cos" in RotaryPositionEmbedding, format supports ND, data type must be the same as "dy", and shape must be the same as "sin".
 * @li sin: A 4D tensor which is input "sin" in RotaryPositionEmbedding, format supports ND, data type must be the same as "dy", and shape must be the same as "cos".
 * @li x: An optional 4D tensor which is input "x" in RotaryPositionEmbedding, format supports ND, data type must be the same as "dy", and shape must be the same as "dy".
 * If "x" is nullptr, the output "dcos" and "dsin" is meaningless.
 * @par Outputs:
 * @li dx: A 4D Tensor which is the grad of input "x" in RotaryPositionEmbedding, format supports ND, data type must be the same as "dy", and shape must be the same as "dy".
 * @li dcos: A 4D Tensor which is the grad of input "cos" in RotaryPositionEmbedding, format supports ND, data type must be the same as "dy", and shape must be the same as "cos".
 * @li dsin: A 4D Tensor which is the grad of input "sin" in RotaryPositionEmbedding, format supports ND, data type must be the same as "dy", and shape must be the same as "sin".
 * @par Attributes:
 * mode: An optional attribute of type int, specifying the mode of rotary position embedding, must be 0-"half", 1-"interleave", 2-"quarter" or 3-"interleave-half". Defaults to 0.
 * Atlas A2 Training Series Product/ Atlas 800I A2 Inference Product and Atlas A3 Training Series Product only support 0-"half" and 1-"interleave".
 * @attention Constraints:
 * Let (B, S, N, D) represents the shape of the 4-D input "dy". Under this representation, the shape constraints of each parameter can be described as follows:
 * @li The D of "dy", "cos", "sin", "x", "dx", "dcos" and "dsin" must be equal. For Ascend 950 AI Processor, D should be less or equal to 1024. 
 * For Atlas A2 Training Series Product/ Atlas 800I A2 Inference Product and Atlas A3 Training Series Product, D should be less or equal to 896.
 * @li In half, interleave and interleave-half mode, D must be a multiple of 2. In quarter mode, D must be a multiple of 4.
 * @li B, S, N of "cos", "sin", "dcos" and "dsin" must meet one of the following four conditions:
 *  - B, S, N are 1, means the shape is (1, 1, 1, D).
 *  - B, S, N are the same as that of "dy", means the shape is (B, S, N, D).
 *  - One of S and N is 1, the remaining one dimension and B are the same as that of "dy", means the shape is (B, 1, N, D) or (B, S, 1, D).
 *  - Two of B, S and N are 1, the remaining one dimension is the same as that of "dy", means the shape is (1, 1, N, D), (1, S, 1, D) or (B, 1, 1, D).
 */
REG_OP(RotaryPositionEmbeddingGrad)
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(cos, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(sin, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OPTIONAL_INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OUTPUT(dx, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OUTPUT(dcos, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OUTPUT(dsin, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .ATTR(mode, Int, 0)
    .OP_END_FACTORY_REG(RotaryPositionEmbeddingGrad)

/**
 * @brief Apply rotary position embedding.
 * @par Inputs:
 * @li x: A 4-dimensions tensor with layout BNSD, BSND or SBND, where B, N < 1000 and D is multiples of 64.
 * Must be one of the following types: float16, float32, bfloat16.
 * @li r1: A 4-dimensions tensor with layout 11SD/B1SD/BNSD, 1S1D/BS1D/BSND or S11D/SB1D/SBND. When r1 broadcasts to x,
 * the product of the broadcast axes needs to be less than 1024. The dtype must be same as "x".
 * @li r2: A 4-dimensions tensor. Has the same shpae and dtype as "r1".

 * @par Outputs:
 * y: A Tensor. Has the same shape and dtype as "x".
 */
REG_OP(RotaryMul)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(r1, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(r2, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OP_END_FACTORY_REG(RotaryMul)

/**
 * @brief Apply rotary position embedding.
 * @par Inputs:
 *@li query: A tensor for transformer query. Must be one of the following types: float16, float32, bfloat16.
 *@li key: A tensor for transformer key. Must be one of the following types: float16, float32, bfloat16.
 *@li cos: A tensor for rotary position embedding cos. Must be one of the following types: float16, float32, bfloat16.
 *@li sin: A tensor for rotary position embedding sin. Must be one of the following types: float16, float32, bfloat16.
 * @par Attributes:
 *@li layout: Optional.Explanation input format. 1-"BSND" 2-"SBND" 3-"BNSD". Defaults to 1.
 *@li rotary_mode: Optional.Explanation rotary mode. Support "half","interleave","quarter". Defaults to "half".
 * @par Outputs:
 *@li query: A tensor for transformer query. Has the same shape as "query".
 *@li key: A tensor for transformer key. Has the same shape as "key".
 * @attention Constraints:
 *@li Inputs  tensor of query, key, cos, sin must be 4D.
 *@li query, key must be the same except for the N dimension, the shape of sin and cos must be the same.
 *@li The B dimension of cos and sin must be consistent with query and key or equal to 1.
 *@li The S dimension of query, key, sin, cos must be the same.
 *@li The N dimension of cos and sin must be equal to 1.
 *@li The D dimension of query, key, sin, cos must be the same and not greater than 1024.
 *@li The dtype of input tensor query, key, sin, cos must be the same.
 *@li When the rotary_mode is "half" or "interleave", the D dimension must be divisible by 2.
 * when the rotary_mode is "quarter", the D dimension must be divisible by 4
 *@li Atlas Inference Series Product and Atlas Trainning Series Product:
    - support types: float16, float32.
    - support layout: not support.
    - support rotary_mode: not support.
 *@li Atlas A2 Training Series Product/ Atlas 800I A2 Inference Product and Atlas A3 Training Series Product:
    - support types: float16, float32.
    - support layout: 1.
    - support rotary_mode: not support.
 *@li Ascend 950 AI Processor
    - support types: float16, float32, bfloat16.
    - support layout: 1, 2, 3.
    - support rotary_mode: "half","interleave","quarter".
 */
REG_OP(ApplyRotaryPosEmb)
    .INPUT(query, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(key, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(cos, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(sin, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .ATTR(layout, Int, 1)
    .ATTR(rotary_mode, String, "half")
    .OUTPUT(query,TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OUTPUT(key,TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OP_END_FACTORY_REG(ApplyRotaryPosEmb)

/**
 * @brief Calculate the inverse gradient of RotaryMul.
 * @par Inputs:
 * @li x: A 4-dimensions tensor with layout BNSD, BSND or SBND, where B, N < 1000 and D is multiples of 64.
 * Must be one of the following types: float16, float, bfloat16. \n
 * @li r1: A 4-dimensions tensor with layout 11SD/BNSD, 1S1D/BSND or S11D/SBND, indicates cos value.
 * When r1 broadcasts to x, the product of the broadcast axes needs to be less than 1024. The dtype must be same as "x". \n
 * @li r2: A 4-dimensions tensor with layout 11SD/BNSD, 1S1D/BSND or S11D/SBND, indicates sin value.
 * Has the same shape and dtype as "r1". \n
 * @li dy: A 4-dimensions tensor. Data of grad increment. Has the same shape and dtype as "x". \n
 * @par Attributes:
 * need_backward: An optional bool. Need to calculate dr1 and dr2 when need_backward is "true". Defaults to "true".
 * @par Outputs:
 * @li dx: A tensor. The grad of input x and has the same shape and dtype as "x". \n
 * @li dr1: A tensor. The grad of input r1 and has the same shape and dtype as "r1". \n
 * @li dr2: A tensor. The grad of input r2 and has the same shape and dtype as "r2". \n
 */
REG_OP(RotaryMulGrad)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BFLOAT16}))
    .INPUT(r1, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BFLOAT16}))
    .INPUT(r2, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BFLOAT16}))
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BFLOAT16}))
    .ATTR(need_backward, Bool, true)
    .OUTPUT(dx, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BFLOAT16}))
    .OUTPUT(dr1, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BFLOAT16}))
    .OUTPUT(dr2, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BFLOAT16}))
    .OP_END_FACTORY_REG(RotaryMulGrad)

/**
* @brief init PartitionMap table. \n

* @par Inputs:
* @li ps_num: A Tensor, dtype is int32. 0-D. indicates ps number.
* @li ps_ids: A Tensor, dtype is int32. 1-D. indicates the id of ps. \n

* @par Attributes:
* @li partition_num: An optional int, indicates the number of partition. Defaults to 65537. \n
*/
REG_OP(InitPartitionMap)
    .INPUT(ps_num, TensorType({DT_INT32}))
    .INPUT(ps_ids, TensorType({DT_INT32}))
    .ATTR(partition_num, Int, 65537)
    .OP_END_FACTORY_REG(InitPartitionMap)

/**
* @brief uninit PartitionMap table. \n
*/
REG_OP(UninitPartitionMap)
    .OP_END_FACTORY_REG(UninitPartitionMap)

/**
* @brief init Embedding hashtable. \n

* @par Inputs:
* @li table_id: A Tensor, dtype is int32. 0-D. indicates the id of hashtable. \n

* @par Attributes:
* @li value_total_len: A Int, indicates the length of hashtable value. \n
* @li embedding_dim: A Int, indicates the length of embedding. \n
* @li bucket_size: An optional int. Defaults to "0". \n
* @li dtype: An optional attribute that type for data. Defaults to "DT_FLOAT". \n
* @li initializer_mode: An optional string of "random_uniform", "truncated_normal" , "constant" or "".
* indicates the algo of init method. Defaults to "".
* @li constant_value: An optional float, used when initializer_mode is "constant". Defaults to "0". \n
* @li min: An optional float used when initializer_mode is "truncated_normal", the minimum value of the random number.
* Defaults to "-2".
* @li max: An optional float used when initializer_mode is "truncated_normal", the maximum value of the random number.
* Defaults to "2".
* @li mu: An optional float used when initializer_mode is "truncated_normal", The mean of the truncated_normal.
* Defaults to "0".
* @li sigma: An optional float used when initializer_mode is "truncated_normal", The variance of the truncated_normal.
* Defaults to "1".
* @li seed: An optional int. Defaults to "0". \n
* @li seed2: An optional int. Defaults to "0". \n
* @li filter_mode: An optional string of "no_filter" or "counter". indicates the type of the hashmap. Defaults to "no_filter". \n
* @li optimizer_mode: An optional string of "adam" or "adamw" or "adagrad" or "sgd" or "rmsprop" or "ftrl". indicates the type of the optimizer_mode.
* Defaults to "".
* @li optimizer_params: An optional float list, when optimizer_mode is "adagrad", the initialize value of the optimizer. \n
*/
REG_OP(InitEmbeddingHashmap)
    .INPUT(table_id, TensorType({DT_INT32}))
    .ATTR(bucket_size, Int, 0)
    .REQUIRED_ATTR(value_total_len, Int)
    .REQUIRED_ATTR(embedding_dim, Int)
    .ATTR(dtype, Type, DT_FLOAT)
    .ATTR(initializer_mode, String, "")
    .ATTR(constant_value, Float, 0)
    .ATTR(min, Float, -2)
    .ATTR(max, Float, 2)
    .ATTR(mu, Float, 0)
    .ATTR(sigma, Float, 1)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .ATTR(filter_mode, String, "no_filter")
    .ATTR(optimizer_mode, String, "")
    .ATTR(optimizer_params, ListFloat, {})
    .OP_END_FACTORY_REG(InitEmbeddingHashmap)

/**
* @brief embedding hashtable data import. \n

* @par Inputs:
* @li file_path: A Tensor, dtype is string. 0-D. indicates embedding filepath.
* @li ps_id: A Tensor, dtype is int32. 0-D. indicates the id of ps.
* @li table_id: A Tensor, dtype is int32. 1-D. indicates the id of hashtable.
* @li global_step: An optional Scalar, dtype is int32/int64. 1-D. indicates the import save step. \n

* @par Attributes:
* @li embedding_dim: A ListInt. indicates the hashtable value number.
* @li value_total_len: A ListInt. indicates the hashtable total length, include m+v or accum.
* @li only_var_flag: An optional bool that only import var. Defaults to "false".
* @li file_type: An optional string that indicates the import file. Defaults to "bin".
* @li table_name: An optional List string that represents table name corresponding to table id. \n
*/
REG_OP(EmbeddingTableImport)
    .INPUT(file_path, TensorType({DT_STRING}))
    .INPUT(ps_id, TensorType({DT_INT32}))
    .INPUT(table_id, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(global_step, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(embedding_dim, ListInt)
    .REQUIRED_ATTR(value_total_len, ListInt)
    .ATTR(only_var_flag, Bool, false)
    .ATTR(file_type, String, "bin")
    .ATTR(table_name, ListString, {})
    .OP_END_FACTORY_REG(EmbeddingTableImport)

/**
* @brief embedding hashtable data import. \n

* @par Inputs:
* @li table_handles: A Tensor, dtype is DT_INT64. 1-D. Indicates addr of hashtable.
* @li embedding_dims: A Tensor, dtype is DT_INT64. 1-D. Indicates the length of values.
* @li bucket_sizes: A Tensor, dtype is DT_INT64. 1-D. Indicates the hash bucket size.
* @li keys: A list of Tensor, dtype is DT_INT64, dynamic input. Indicates the hashtable keys number.
* @li counters: A list of Tensor, dtype is DT_UINT64, dynamic input. Indicates the hashtable counters.
* @li filter_flags: A list of Tensor, dtype is DT_UINT8, dynamic input. Indicates filter_flags.
* @li values: A list of Tensor, dtype is DT_FLOAT32, dynamic input. Indicates the import values .

* @attention Constraints:
* @li table_handles, embedding_dims and bucket_sizes have same shape len.
* @li actual kernel compute filter_flags be 64 bit.
* @li single struct contains keys、counters、filter_flags and values, should be algin to 8 bytes.
*/
REG_OP(EmbeddingHashTableImport)
.INPUT(table_handles, TensorType({DT_INT64}))
.INPUT(embedding_dims, TensorType({DT_INT64}))
.INPUT(bucket_sizes, TensorType({DT_INT64}))
.DYNAMIC_INPUT(keys, TensorType({DT_INT64}))
.DYNAMIC_INPUT(counters, TensorType({DT_UINT64}))
.DYNAMIC_INPUT(filter_flags, TensorType({DT_UINT8}))
.DYNAMIC_INPUT(values, TensorType({DT_FLOAT}))
.OP_END_FACTORY_REG(EmbeddingHashTableImport)

/**
*@brief insert keys in NPU hashtable and return coresponding value.
*
*@par Inputs:
*Inputs include:
* @li table_handle: A tensor. Must be int64. Contains addr of table's infos. shape of [5].
* @li keys: A tensor. Must be int64. Keys to be insert.
*
*@par Outputs:
* values: A tensor. Must be float32. N-D with shape [N, embedding_dim].
*
*@par Attributes:
* @li bucket_size: Required, int, table capacity.
* @li embedding_dim: Required, int, value dims.
* @li filter_mode: Optional, string, set to "no_filter" or "counter", set "counter" to enable the counter-based filter mode, set "no_filter" to disable the filter function. default is "no_filter".
* @li filter_freq: Optional, int, filter threshold. default is 0.
* @li default_key_or_value: Optional, bool, set true will return the value of default_key, set false will return default_value. default is false.
* @li default_key: Optional, int, default key set by customer,  default is 0.
* @li default_value: Optional, float, default value set by customer. default is 0.
* @li filter_key_flag: Optional, bool, set true to enable filter_key, set false to disable the function of filter key. default is false.
* @li filter_key: Optional, int, filter input key and return default_value when flag is true. default is -1.
*/
REG_OP(EmbeddingHashTableLookupOrInsert)
.INPUT(table_handle, TensorType({DT_INT64}))
.INPUT(keys, TensorType({DT_INT64}))
.OUTPUT(values, TensorType({DT_FLOAT32}))
.REQUIRED_ATTR(bucket_size, Int)
.REQUIRED_ATTR(embedding_dim, Int)
.ATTR(filter_mode, String, "no_filter")
.ATTR(filter_freq, Int, 0)
.ATTR(default_key_or_value, Bool, false)
.ATTR(default_key, Int, 0)
.ATTR(default_value, Float, 0)
.ATTR(filter_key_flag, Bool, false)
.ATTR(filter_key, Int, -1)
.OP_END_FACTORY_REG(EmbeddingHashTableLookupOrInsert)

/**
*@brief Evict keys and reinitialize embedding values in NPU hashtable.
*
*@par Inputs:
* @li table_handle: A 1-D tensor of DT_INT64, which contains tables' starting addrs.
* @li keys: A 1-D tensor of DT_INT64, which represents keys to be evicted.
* @li sampled_values: A 1-D tensor of DT_FLOAT, used for reinitializing embeddings of evicted keys. \n
*
*@par Attributes:
* @li table_cap: A required int attr, which indicates the capacity of table.
* @li embedding_dim: A required int attr, which indicates dimension of embedding value.
* @li init_mode: An optional string attr. which indicates reinitializing modes. Mode "random"
      or "constant" are available, defaults to "constant".
* @li const_val: An optional float attr, which indicates the constant value in initialization. \n
*/
REG_OP(EmbeddingHashTableEvict)
    .INPUT(table_handle, TensorType({DT_INT64}))
    .INPUT(keys, TensorType({DT_INT64}))    
    .OPTIONAL_INPUT(sampled_values, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(table_cap, Int)
    .REQUIRED_ATTR(embedding_dim, Int)
    .ATTR(init_mode, String, "constant")
    .ATTR(const_val, Float, 0.0)
    .OP_END_FACTORY_REG(EmbeddingHashTableEvict)

/**
* @brief embedding hashtable export. \n

* @par Inputs:
* @li table_handles: A Tensor, dtype is DT_INT64, 1-D. indicates the table handle address.
* @li table_sizes: A Tensor, dtype is DT_INT64,  1-D. indicates the size of hash map for each table.
* @li embedding_dims: A Tensor, dtype is DT_INT64,  1-D. indicates the dim of embedding value in hashtable.
* @li bucket_sizes: A Tensor, dtype is DT_INT64. 1-D. indicates the hash bucket size. \n

* @par Outputs:
* @li keys: A list of Tensor objects, dtype is DT_INT64. indicates the hashtable keys. It's a dynamic output.
* @li counters: A list of Tensor objects, dtype is DT_UINT64. indicates the hashtable counter. It's a dynamic output.
* @li filter_flags: A list of Tensor objects, dtype is DT_UINT8. indicates the hashtable filter flag. It's a dynamic
*     output.
* @li values: A list of Tensor objects, dtype is DT_FLOAT. indicates the hashtable value. It's a dynamic output. \n

* @par Attributes:
* @li export_mode: An optional String. Represents export mode, can be either "all" or "new". Defaults to "all".
* @li filtered_export_flag: An optional Bool. Represents filter export flag on counter filter scenario.
      Defaults to false. \n

* @attention Constraints:
* @li table_handles, table_sizes, embedding_dims and bucket_sizes have same shape len.
* @li the number of tensors in each dynamic output tensor list (keys/counters/filter_flags/values) shall be equal to
    the shape len of table_handles. \n
*/
REG_OP(EmbeddingHashTableExport)
    .INPUT(table_handles, TensorType({DT_INT64}))
    .INPUT(table_sizes, TensorType({DT_INT64}))
    .INPUT(embedding_dims, TensorType({DT_INT64}))
    .INPUT(bucket_sizes, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(keys, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(counters, TensorType({DT_UINT64}))
    .DYNAMIC_OUTPUT(filter_flags, TensorType({DT_UINT8}))
    .DYNAMIC_OUTPUT(values, TensorType({DT_FLOAT}))
    .ATTR(export_mode, String, "all")
    .ATTR(filtered_export_flag, Bool, false)
    .OP_END_FACTORY_REG(EmbeddingHashTableExport)

/**
*@brief Init EmbeddingHashTable.
*
*@par Inputs:
*Inputs include:
* table_handle: A Tensor. Dtype support: int64
* sampled_values: A Tensor. Dtype support: float32
*
*@par Attributes:
* @li bucket_size: An required attribute indicates hashtable size.
* @li embedding_dim: An required attribute indicates hashtable value size.
* @li initializer_mode: An optional attribute indicates init hashtable mode.
* @li constant_value: An optional attribute indicates hashtable constant mode const value.
*
*@attention Constraints:
* Only support value's dtype is float32.
*/
REG_OP(InitEmbeddingHashTable)
    .INPUT(table_handle, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(sampled_values, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(bucket_size, Int)
    .REQUIRED_ATTR(embedding_dim, Int)
    .ATTR(initializer_mode, String, "random")
    .ATTR(constant_value, Float, 0.0)
    .OP_END_FACTORY_REG(InitEmbeddingHashTable)

/**
* @brief embedding hashtable data lookup. \n

* @par Inputs:
* @li table_id: A Tensor, dtype is int32. 0-D. indicates the id of hashtable.
* @li keys: A Tensor, dtype is uint32. 1-D. indicates the hashtable key. \n

* @par Outputs:
* @li values: indicates the hashtable value. \n

* @par Attributes:
* @li embedding_dim: Int List. indicates the hashtable value number.
* @li default_value: An optional float List, indicate the default value when can not find key. Defaults to "-1". \n
*/
REG_OP(EmbeddingTableFind)
    .INPUT(table_id, TensorType({DT_INT32}))
    .INPUT(keys, TensorType({DT_INT64}))
    .OUTPUT(values, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(embedding_dim, ListInt)
    .ATTR(default_value, ListFloat, {-1})
    .OP_END_FACTORY_REG(EmbeddingTableFind)

/**
* @brief uninit embedding hashtable. \n

* @par Inputs:
* @li table_id: A Tensor, dtype is int32. 0-D. indicates the id of hashtable. \n
*/
REG_OP(UninitEmbeddingHashmap)
    .INPUT(table_id, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(UninitEmbeddingHashmap)

/**
* @brief embedding hashtable lookup or init. \n

* @par Inputs:
* @li table_id: A Tensor, dtype is DT_INT32. 0-D. indicates the id of hashtable.
* @li keys: A Tensor, dtype is DT_INT64. 1-D. indicates the hashtable key. \n

* @par Outputs:
* @li values: indicates the hashtable value. \n

* @par Attributes:
* @li embedding_dim: Int List, indicates the dim of embedding var value in hashtable.
* @li value_total_len: Int List, indicates the dim of embedding var+m+v or var+accum values in hashtable
* @li initializer_mode: An optional string list of "random_uniform", "truncated_normal" or "constant".
* indicates the algo of init method. Defaults to "random_uniform".
* @li constant_value: An optional string list, used when initializer_mode is "constant". Defaults to "0".
* @li min: An optional string list, used when initializer_mode is "truncated_normal", the minimum value of the random number.
* Defaults to "-2".
* @li max: An optional string list, used when initializer_mode is "truncated_normal", the maximum value of the random number.
* Defaults to "2".
* @li mu: An optional string list, used when initializer_mode is "truncated_normal", The mean of the truncated_normal.
* Defaults to "0".
* @li sigma: An optional string list, used when initializer_mode is "truncated_normal", The variance of the truncated_normal.
* Defaults to "1".
* @li seed: An optional int list, used to create a random seed. Defaults to "0".
* @li seed2: An optional int list, used to create a random seed. Defaults to "0".
* @li filter_mode: An optional string list of "no_filter" or "counter". indicates the type of the hashmap. Defaults to "no_filter".
* @li filter_freq: An optional int list, used to set the threshold of the tal. Defaults to "0".
* @li default_key_or_value: An optional int list, indicates the default value get way. Defaults to "0".
* @li default_key: An optional int list, when default_key_or_value is true, use the default_key corresponding value as default value. Defaults to "0".
* @li default_value: An optional int list, when default_key_or_value is false, use the default_value as default value. Defaults to "0".
* @li completion_key: An optional int list, indicates the completion hashtable key. Defaults to "0".
* @li completion_key_mask: An optional int list, whether to perform no-update interception when key==completion_key. Defaults to "1".
* @li optimizer_mode: An optional string list of "adam" or "adamw" or "adagrad" or "sgd" or "rmsprop" or "ftrl". indicates the type of the optimizer_mode.
* Defaults to "".
* @li optimizer_params: An optional float list, when optimizer_mode is "adagrad", the initialize value of the optimizer. \n
*/
REG_OP(EmbeddingTableFindAndInit)
    .INPUT(table_id, TensorType({DT_INT32}))
    .INPUT(keys, TensorType({DT_INT64}))
    .OUTPUT(values, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(embedding_dim, ListInt)
    .REQUIRED_ATTR(value_total_len, ListInt)
    .ATTR(initializer_mode, ListString, {"random_uniform"})
    .ATTR(constant_value, ListFloat, {0})
    .ATTR(min, ListFloat, {-2})
    .ATTR(max, ListFloat, {2})
    .ATTR(mu, ListFloat, {0})
    .ATTR(sigma, ListFloat, {1})
    .ATTR(seed, ListInt, {0})
    .ATTR(seed2, ListInt, {0})
    .ATTR(filter_mode, ListString, {"no_filter"})
    .ATTR(filter_freq, ListInt, {0})
    .ATTR(default_key_or_value, ListInt, {0})
    .ATTR(default_key, ListInt, {0})
    .ATTR(default_value, ListFloat, {0})
    .ATTR(completion_key, ListInt, {0})
    .ATTR(completion_key_mask, ListInt, {1})
    .ATTR(optimizer_mode, ListString, {})
    .ATTR(optimizer_params, ListFloat, {})
    .OP_END_FACTORY_REG(EmbeddingTableFindAndInit)

/**
* @brief embedding hashtable embedding applyadam. \n

* @par Inputs:
* @li var_handle: The handle of embedding hashtable.
* @li beta1_power: A Scalar, dtype is DT_FLOAT16 or DT_FLOAT. 0-D. indicates the beta1's power.
* @li beta2_power: A Scalar, dtype is same as "beta1_power". 0-D. indicates the beta2's power.
* @li lr: A Scalar, dtype is same as "beta1_power". 0-D. indicates the learning rate.
* @li beta1: A Scalar, dtype is same as "beta1_power". 0-D. indicates the beta1 param.
* @li beta2: A Scalar, dtype is same as "beta1_power". 0-D. indicates the beta2 param.
* @li epsilon: A Scalar, dtype is same as "beta1_power". 0-D. indicates the small value param.
* @li grad: A Tensor, dtype is same as "beta1_power". 1-D. indicates the grad.
* @li keys: A Tensor, dtype is DT_INT64. 1-D. indicates the hashtable key.
* @li global_step: A Scalar, dtype is DT_INT32/DT_INT64. 0-D. indicates the train step. \n

* @par Outputs:
* @li var_handle: The handle of embedding hashtable. \n

* @par Attributes:
* @li embedding_dim: Int List, indicates the dim of embedding value in hashtable.
* @li mask_zero: An optional int list, whether to perform no-update interception when key==0. Defaults to "0".
* @li padding_key: An optional int list, indicates the padding hashtable key. Defaults to "0".
* @li padding_key_mask: An optional int list, whether to perform no-update interception when key==padding_key. Defaults to "1".
* @li completion_key: An optional int list, indicates the completion hashtable key. Defaults to "0".
* @li completion_key_mask: An optional int list, whether to perform no-update interception when key==completion_key. Defaults to "1". \n
*/
REG_OP(EmbeddingApplyAdam)
    .INPUT(var_handle, TensorType({DT_RESOURCE, DT_INT32}))
    .INPUT(beta1_power, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(beta2_power, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(lr, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(beta1, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(beta2, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(epsilon, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(grad, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(keys, TensorType({DT_INT64}))
    .INPUT(global_step, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(var_handle, TensorType({DT_RESOURCE, DT_INT32}))
    .REQUIRED_ATTR(embedding_dim, ListInt)
    .ATTR(mask_zero, ListInt, {0})
    .ATTR(padding_key, ListInt, {0})
    .ATTR(padding_key_mask, ListInt, {1})
    .ATTR(completion_key, ListInt, {0})
    .ATTR(completion_key_mask, ListInt, {1})
    .OP_END_FACTORY_REG(EmbeddingApplyAdam)

/**
* @brief embedding hashtable embedding applyadamW. \n

* @par Inputs:
* @li var_handle: The handle of embedding hashtable.
* @li beta1_power: A Tensor, dtype is float16 or float. 0-D. indicates the beta1's power.
* @li beta2_power: A Tensor, dtype is same as "beta1_power". 0-D. indicates the beta2's power.
* @li lr: A Tensor, dtype is same as "beta1_power". 0-D. indicates the learning rate.
* @li weight_decay: A Tensor, dtype is same as "beta1_power". 0-D. indicates the weight decay.
* @li beta1: A Tensor, dtype is same as "beta1_power". 0-D. indicates the beta1 param.
* @li beta2: A Tensor, dtype is same as "beta1_power". 0-D. indicates the beta2 param.
* @li epsilon: A Tensor, dtype is same as "beta1_power". 0-D. indicates the small value param.
* @li grad: A Tensor, dtype is same as "beta1_power". 1-D. indicates the grad.
* @li keys: A Tensor, dtype is int64. 1-D. indicates the hashtable key.
* @li max_grad_norm: A mutable Tensor of the same type as "beta1_power", an optional input.
* @li global_step: A Scalar, dtype is DT_INT32/DT_INT64. 0-D. indicates the train step. \n

* @par Outputs:
* @li var_handle: The handle of embedding hashtable. \n

* @par Attributes:
* @li embedding_dim: Int list, indicates the dim of embedding value in hashtable.
* @li amsgrad: An optional int list, indicates whether to use the AMSGrad variant of hits algorithm from
*     the paper On the Convergence of Adam and Beyond. Defaults to "0".
* @li maximize: An optional int list, maximize the params based on the objective. Defaults to "0".
* @li mask_zero: An optional int list, whether to perform no-update interception when key==0. Defaults to "0".
* @li padding_key: An optional int list, indicates the padding hashtable key. Defaults to "0".
* @li padding_key_mask: An optional int list, whether to perform no-update interception when key==padding_key. Defaults to "1".
* @li completion_key: An optional int list, indicates the completion hashtable key. Defaults to "0".
* @li completion_key_mask: An optional int list, whether to perform no-update interception when key==completion_key. Defaults to "1". \n
*/
REG_OP(EmbeddingApplyAdamW)
    .INPUT(var_handle, TensorType({DT_RESOURCE, DT_INT32}))
    .INPUT(beta1_power, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(beta2_power, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(lr, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(weight_decay, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(beta1, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(beta2, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(epsilon, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(grad, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(keys, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(max_grad_norm, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OPTIONAL_INPUT(global_step, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(var_handle, TensorType({DT_RESOURCE, DT_INT32}))
    .REQUIRED_ATTR(embedding_dim, ListInt)
    .ATTR(amsgrad, ListInt, {0})
    .ATTR(maximize, ListInt, {0})
    .ATTR(mask_zero, ListInt, {0})
    .ATTR(padding_key, ListInt, {0})
    .ATTR(padding_key_mask, ListInt, {1})
    .ATTR(completion_key, ListInt, {0})
    .ATTR(completion_key_mask, ListInt, {1})
    .OP_END_FACTORY_REG(EmbeddingApplyAdamW)

/**
* @brief embedding hashtable resource applyadagrad. \n

* @par Inputs:
* @li var_handle: The handle of embedding hashtable.
* @li lr: A Scalar, dtype is the same as "grad". 0-D. indicates the learning rate.
* @li grad: A Tensor, dtype is the same as "lr". 1-D. indicates the grad.
* @li keys: A Tensor, dtype is DT_INT64. 1-D. indicates the hashtable key.
* @li global_step: A Scalar, dtype is DT_INT32/DT_INT64. 0-D. indicates the train step. \n

* @par Outputs:
* @li var_handle: The handle of embedding hashtable. \n

* @par Attributes:
* @li embedding_dim: Int list, indicates the dim of embedding value in hashtable.
* @li mask_zero: An optional int list, whether to perform no-update interception when key==0. Defaults to "0".
* @li padding_key: An optional int list, indicates the padding hashtable key. Defaults to "0".
* @li padding_key_mask: An optional int list, whether to perform no-update interception when key==padding_key. Defaults to "1".
* @li completion_key: An optional int list, indicates the completion hashtable key. Defaults to "0".
* @li completion_key_mask: An optional int list, whether to perform no-update interception when key==completion_key. Defaults to "1". \n
*/
REG_OP(EmbeddingApplyAdaGrad)
    .INPUT(var_handle, TensorType({DT_RESOURCE, DT_INT32}))
    .INPUT(lr, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(grad, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(keys, TensorType({DT_INT64}))
    .INPUT(global_step, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(var_handle, TensorType({DT_RESOURCE, DT_INT32}))
    .REQUIRED_ATTR(embedding_dim, ListInt)
    .ATTR(mask_zero, ListInt, {0})
    .ATTR(padding_key, ListInt, {0})
    .ATTR(padding_key_mask, ListInt, {1})
    .ATTR(completion_key, ListInt, {0})
    .ATTR(completion_key_mask, ListInt, {1})
    .OP_END_FACTORY_REG(EmbeddingApplyAdaGrad)

/**
* @brief embedding hashtable resource apply sgd. \n

* @par Inputs:
* @li var_handle: The handle of embedding hashtable.
* @li lr: A Scalar, dtype is the same as "grad". 0-D. indicates the learning rate.
* @li grad: A Tensor, dtype is DT_FLOAT/DT_FLOAT16. 1-D. indicates the grad.
* @li keys: A Tensor, dtype is DT_INT64. 1-D. indicates the hashtable key.
* @li global_step: A Scalar, dtype is DT_INT32/DT_INT64. 0-D. indicates the train step. \n

* @par Outputs:
* @li var_handle: The handle of embedding hashtable. \n

* @par Attributes:
* @li embedding_dim: Int list, indicates the dim of embedding value in hashtable.
* @li mask_zero: An optional int list, whether to perform non-update interception when key==0. Defaults to "0".
* @li padding_key: An optional int list, indicates the padding hashtable key. Defaults to "0".
* @li padding_key_mask: An optional int list, whether to perform no-update interception when key==padding_key. Defaults to "1".
* @li completion_key: An optional int list, indicates the completion hashtable key. Defaults to "0".
* @li completion_key_mask: An optional int list, whether to perform no-update interception when key==completion_key. Defaults to "1". \n
*/
REG_OP(EmbeddingApplySgd)
    .INPUT(var_handle, TensorType({DT_RESOURCE, DT_INT32}))
    .INPUT(lr, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(grad, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(keys, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(global_step, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(var_handle, TensorType({DT_RESOURCE, DT_INT32}))
    .REQUIRED_ATTR(embedding_dim, ListInt)
    .ATTR(mask_zero, ListInt, {0})
    .ATTR(padding_key, ListInt, {0})
    .ATTR(padding_key_mask, ListInt, {1})
    .ATTR(completion_key, ListInt, {0})
    .ATTR(completion_key_mask, ListInt, {1})
    .OP_END_FACTORY_REG(EmbeddingApplySgd)

/**
* @brief embedding hashtable resource apply rmsprop. \n

* @par Inputs:
* @li var_handle: The handle of embedding hashtable.
* @li lr: A Scalar, dtype is the same as "grad". indicates the learning rate.
* @li rho: A Scalar, dtype is the same as "grad". indicates the decay rate.
* @li momentum: A Scalar, dtype is the same as "grad". indicates the momentum.
* @li epsilon: A Scalar, dtype is the same as "grad". indicates the small value param.
* @li grad: A Tensor, dtype is NumberType. indicates the grad.
* @li keys: A Tensor, dtype is DT_INT64. 1-D. indicates the hashtable key.
* @li global_step: A Scalar, dtype is DT_INT32/DT_INT64. 0-D. indicates the train step. \n

* @par Outputs:
* @li var_handle: The handle of embedding hashtable. \n

* @par Attributes:
* @li embedding_dim: Int list, indicates the dim of embedding value in hashtable.
* @li mask_zero: An optional int list, whether to perform non-update interception when key==0. Defaults to "0".
* @li padding_key: An optional int list, indicates the padding hashtable key. Defaults to "0".
* @li padding_key_mask: An optional int list, whether to perform no-update interception when key==padding_key. Defaults to "1".
* @li completion_key: An optional int list, indicates the completion hashtable key. Defaults to "0".
* @li completion_key_mask: An optional int list, whether to perform no-update interception when key==completion_key. Defaults to "1". \n
*/
REG_OP(EmbeddingApplyRmsprop)
    .INPUT(var_handle, TensorType({DT_RESOURCE, DT_INT32}))
    .INPUT(lr, TensorType::NumberType())
    .INPUT(rho, TensorType::NumberType())
    .INPUT(momentum, TensorType::NumberType())
    .INPUT(epsilon, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .INPUT(keys, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(global_step, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(var_handle, TensorType({DT_RESOURCE, DT_INT32}))
    .REQUIRED_ATTR(embedding_dim, ListInt)
    .ATTR(mask_zero, ListInt, {0})
    .ATTR(padding_key, ListInt, {0})
    .ATTR(padding_key_mask, ListInt, {1})
    .ATTR(completion_key, ListInt, {0})
    .ATTR(completion_key_mask, ListInt, {1})
    .OP_END_FACTORY_REG(EmbeddingApplyRmsprop)

/**
* @brief embedding hashtable embedding applyftrl. \n

* @par Inputs:
* @li var_handle: The handle of embedding hashtable.
* @li lr: A Scalar, dtype is DT_FLOAT16 or DT_FLOAT. 0-D. indicates the learning rate.
* @li lr_power: A Scalar, dtype is same as "lr". 0-D. indicates the learning rate factor.
* @li lambda1: A Scalar, dtype is same as "lr". 0-D. indicates the lambda1 param.
* @li lambda2: A Scalar, dtype is same as "lr". 0-D. indicates the lambda2 param.
* @li grad: A Tensor, dtype is same as "lr". 1-D. indicates the grad.
* @li keys: A Tensor, dtype is DT_INT64. 1-D. indicates the hashtable key.
* @li global_step: A Scalar, dtype is DT_INT32/DT_INT64. 0-D. indicates the train step. \n

* @par Outputs:
* @li var_handle: The handle of embedding hashtable. \n

* @par Attributes:
* @li embedding_dim: Int list, indicates the dim of embedding value in hashtable.
* @li mask_zero: An optional int list, whether to perform no-update interception when key==0. Defaults to "0".
* @li padding_key: An optional int list, indicates the padding hashtable key. Defaults to "0".
* @li padding_key_mask: An optional int list, whether to perform no-update interception when key==padding_key. Defaults to "1".
* @li completion_key: An optional int list, indicates the completion hashtable key. Defaults to "0".
* @li completion_key_mask: An optional int list, whether to perform no-update interception when key==completion_key. Defaults to "1". \n
*/
REG_OP(EmbeddingApplyFtrl)
    .INPUT(var_handle, TensorType({DT_RESOURCE, DT_INT32}))
    .INPUT(lr, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(lr_power, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(lambda1, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(lambda2, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(grad, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(keys, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(global_step, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(var_handle, TensorType({DT_RESOURCE, DT_INT32}))
    .REQUIRED_ATTR(embedding_dim, ListInt)
    .ATTR(mask_zero, ListInt, {0})
    .ATTR(padding_key, ListInt, {0})
    .ATTR(padding_key_mask, ListInt, {1})
    .ATTR(completion_key, ListInt, {0})
    .ATTR(completion_key_mask, ListInt, {1})
    .OP_END_FACTORY_REG(EmbeddingApplyFtrl)

/**
* @brief Exponential decay algorithm. \n

* @par Inputs:
* @li var_handle: The handle of embedding hashtable.
* @li initial_learning_rate: A Scalar, dtype is DT_FLOAT/DT_FLOAT16. 0-D. indicates the learning rate.
* @li decay_rate: A Scalar, dtype is  the same as initial_learning_rate. 0-D. indicates the decay rate.
* @li decay_steps: A Scalar, dtype is DT_INT32/DT_INT64. 0-D. indicates the decay steps.
* @li global_step: A optional Scalar, dtype is DT_INT32/DT_INT64. 0-D. indicates the train step. \n

* @par Outputs:
* @li decayed_lr: Indicates the learning rate after updating. \n

* @par Attributes:
* @li staircase: An optional bool that dims is 0-D and indicates the strategy for updating lr.
* If true indicates updating lr according to the decay_steps. If false indicates updating lr each step. Defaults to false.\n
*/
REG_OP(ExponentialDecayLR)
    .INPUT(var_handle, TensorType({DT_RESOURCE, DT_INT32}))
    .INPUT(initial_learning_rate, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(decay_rate, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(decay_steps, TensorType({DT_INT32, DT_INT64}))
    .OPTIONAL_INPUT(global_step, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(decayed_lr, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(staircase, Bool, false)
    .OP_END_FACTORY_REG(ExponentialDecayLR)

/**
* @brief embedding hashtable export. \n

* @par Inputs:
* @li file_path: A String, indicates the export file path.
* @li ps_id: A Int, dtype is DT_INT32, indicates the ps server id.
* @li table_id: A Tensor, 1D, dtype is DT_INT32, indicates the hashtable id.
* @li global_step: A Scalar that indicates the export save step. It's dtype is DT_INT32 or DT_INT64. Dim is 0-D.  \n

* @par Attributes:
* @li embedding_dim: A int list. indicates the hashtable value number.
* @li value_total_len: A int list. indicates the hashtable total length, include m+v or accum.
* @li export_mode: An optional string. export mode, the value of export mode has four values:
*     "all" "old" "new" "specifiednew", Defaults to "all".
* @li only_var_flag: An optional bool. only export var. Defaults to "false".
* @li file_type: An optional string that indicates the export file. Defaults to "bin".
* @li table_name: An optional string list that represents table name corresponding to table id.
* @li filter_export_flag: An optional bool that represents filter export flag on counter filter scenario.
* @li steps_to_live_list: An optional int list that dtype is DT_INT64. 0-D. indicates the step threshold. \n
*/
REG_OP(EmbeddingTableExport)
    .INPUT(file_path, TensorType({DT_STRING}))
    .INPUT(ps_id, TensorType({DT_INT32}))
    .INPUT(table_id, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(global_step, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(embedding_dim, ListInt)
    .REQUIRED_ATTR(value_total_len, ListInt)
    .ATTR(export_mode, String, "all")
    .ATTR(only_var_flag, Bool, false)
    .ATTR(file_type, String, "bin")
    .ATTR(table_name, ListString, {})
    .ATTR(filter_export_flag, Bool, false)
    .ATTR(steps_to_live_list, ListInt, {})
    .OP_END_FACTORY_REG(EmbeddingTableExport)

/**
* @brief Embedding table eviction. \n

* @par Inputs:
* @li var_handle: The handle of embedding hashtable.
* @li global_step: A Scalar, dtype is DT_INT32/DT_INT64. 0-D. indicates the train step. \n

* @par Outputs:
* @li var_handle: The handle of embedding hashtable. \n

* @par Attributes:
* @li steps_to_live: An optional int that indicates the step threshold. Defaults to 0. Dtype is DT_INT64. Dim is 0-D. \n
*/
REG_OP(EmbeddingTableEvict)
    .INPUT(var_handle, TensorType({DT_RESOURCE, DT_INT32}))
    .OPTIONAL_INPUT(global_step, TensorType({DT_INT32, DT_INT64}))
    .ATTR(steps_to_live, Int, 0)
    .OP_END_FACTORY_REG(EmbeddingTableEvict)

/**
* @brief embedding tableid trans to resource. \n

* @par Inputs:
* @li table_id: A Tensor, dtype is int32. 0-D. indicates the id of hashtable.

* @par Outputs:
* @li table_handle: indicates the resource_handle of tableid. \n
*/
REG_OP(TableToResource)
    .INPUT(table_id, TensorType({DT_INT32}))
    .OUTPUT(table_handle, TensorType({DT_RESOURCE}))
    .OP_END_FACTORY_REG(TableToResource)

/**
* @brief embedding feature_id trans to offset_id. \n

* @par Inputs:
* @li feature_id: A Tensor, dtype is int64. \n

* @par Outputs:
* @li offset_id: A Tensor with same shape of feature_id, dtype is int32. \n
*/
REG_OP(EmbeddingFeatureMapping)
    .INPUT(feature_id, TensorType({DT_INT64}))
    .OUTPUT(offset_id, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(EmbeddingFeatureMapping)

/**
* @brief embedding feature_id trans to offset_id according to table name. \n

* @par Inputs:
* @li table_name: A Scalar, dtype is string, indicates the hash table name.
* @li feature_id: A Tensor, dtype is int64, indicates the original hash key. \n

* @par Outputs:
* @li offset_id: A Tensor with same shape of feature_id, dtype is int32.
*                indicates the mapping value of hash key. \n

* @par Attributes:
* @li table_total_size: A int list, indicates the table total size of small table combination scenario.
* @li table_actual_size: A int list, indicates the table actual size of small table combination scenario. \n
*/
REG_OP(EmbeddingFeatureMappingV2)
    .INPUT(table_name, TensorType({DT_STRING}))
    .INPUT(feature_id, TensorType({DT_INT64}))
    .OUTPUT(offset_id, TensorType({DT_INT32}))
    .REQUIRED_ATTR(table_total_size, ListInt)
    .REQUIRED_ATTR(table_actual_size, ListInt)
    .OP_END_FACTORY_REG(EmbeddingFeatureMappingV2)

/**
* @brief get export size of the embedding table. \n

* @par Inputs:
* @li table_name: A Tensor, dtype is string, indicates the hash table names. \n

* @par Outputs:
* @li feature_size: A Tensor, dtype is int64, indicates the size of hash map for each table. \n
*/
REG_OP(EmbeddingFeatureMappingTableSize)
    .INPUT(table_name, TensorType({DT_STRING}))
    .OUTPUT(feature_size, TensorType({DT_INT64}))
    .OP_END_FACTORY_REG(EmbeddingFeatureMappingTableSize)

/**
* @brief query data in the embedding table based on table name. \n

* @par Inputs:
* @li table_name: A Tensor, dtype is string, indicates the hash table names.
* @li feature_size: A Tensor, dtype is int64, indicates the size of hash map for each table. \n

* @par Outputs:
* @li feature_id: Tensors which number is consistent with table's number, dtype is int64,
*                 indicates the original hash key for one table.
* @li offset_id: Tensors which number is consistent with table's number, dtype is int32.
*                indicates the mapping value of hash key for one table. \n
*/
REG_OP(EmbeddingFeatureMappingFind)
    .INPUT(table_name, TensorType({DT_STRING}))
    .INPUT(feature_size, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(feature_id, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(offset_id, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(EmbeddingFeatureMappingFind)

/**
* @brief export table data from the embedding table to file. \n

* @par Inputs:
* @li file_path: A Scalar, dtype is string, indicates the file path to export.
* @li table_name: A Tensor, dtype is string, indicates the hash table names.
* @li global_step: A Scalar, dtype is DT_INT32/DT_INT64. 0-D. indicates the export save step.
* @li values: An optional Tensor whose shape is sum of feature_id's shape, dtype is float32,
*             indicates the values of hash key.
* @li feature_id: Tensors which number is consistent with table's number, dtype is int64,
*                 indicates the original hash key for one table.
* @li offset_id: Tensors which number is consistent with table's number, dtype is int32.
*                indicates the mapping value of hash key for one table. \n

* @par Attributes:
* @li embedding_dim: An optional int list that indicates the length of embedding for each table.
*/
REG_OP(EmbeddingFeatureMappingExport)
    .INPUT(file_path, TensorType({DT_STRING}))
    .INPUT(table_name, TensorType({DT_STRING}))
    .OPTIONAL_INPUT(global_step, TensorType({DT_INT32, DT_INT64}))
    .OPTIONAL_INPUT(values, TensorType({DT_FLOAT}))
    .DYNAMIC_INPUT(feature_id, TensorType({DT_INT64}))
    .DYNAMIC_INPUT(offset_id, TensorType({DT_INT32}))
    .ATTR(embedding_dim, ListInt, {})
    .OP_END_FACTORY_REG(EmbeddingFeatureMappingExport)

/**
* @brief get import size of the embedding table file. \n

* @par Inputs:
* @li file_path: A Scalar, dtype is string, indicates the path of import file.
* @li table_name: A Tensor, dtype is string, indicates the hash table names.
* @li global_step: A Scalar, dtype is DT_INT32/DT_INT64. 0-D. It indicates the save step. \n

* @par Outputs:
* @li feature_size: A Tensor, dtype is int64, indicates the size of hash map for each table. \n

* @par Attributes:
* @li embedding_dim: List of int, indicates the length of embedding for each table.
* @li only_offset_flag: An optional bool that only export feature id and offset id. Defaults to "true".
*/
REG_OP(EmbeddingFeatureMappingFileSize)
    .INPUT(file_path, TensorType({DT_STRING}))
    .INPUT(table_name, TensorType({DT_STRING}))
    .OPTIONAL_INPUT(global_step, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(feature_size, TensorType({DT_INT64}))
    .REQUIRED_ATTR(embedding_dim, ListInt)
    .ATTR(only_offset_flag, Bool, true)
    .OP_END_FACTORY_REG(EmbeddingFeatureMappingFileSize)

/**
* @brief import table data from file to embedding table. \n

* @par Inputs:
* @li file_path: A Scalar, dtype is string, indicates the path of import file.
* @li table_name: A Tensor, dtype is string, indicates the hash table names.
* @li feature_size: A Tensor, dtype is int64, indicates the size of hash map for each table.
* @li global_step: A Scalar, dtype is DT_INT32/DT_INT64. 0-D. indicates the import save step. \n

* @par Outputs:
* @li feature_id: Tensors which number is consistent with table's number, dtype is int64,
*                 indicates the original hash key for each table.
* @li offset_id: Tensors which number is consistent with table's number,
*                with same shape of feature_id for each table, dtype is int32.
*                indicates the mapping value of hash key for each table.
* @li values: Tensors which number is consistent with table's number,
*             with same shape of feature_id for each table, dtype is float32.
*             indicates the values of hash key for each table. \n

* @par Attributes:
* @li embedding_dim: List of int, indicates the length of embedding for each table.
* @li only_offset_flag: An optional bool that only export feature id and offset id, Defaults to "true".
*/
REG_OP(EmbeddingFeatureMappingImport)
    .INPUT(file_path, TensorType({DT_STRING}))
    .INPUT(table_name, TensorType({DT_STRING}))
    .INPUT(feature_size, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(global_step, TensorType({DT_INT32, DT_INT64}))
    .DYNAMIC_OUTPUT(feature_id, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(offset_id, TensorType({DT_INT32}))
    .DYNAMIC_OUTPUT(values, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(embedding_dim, ListInt)
    .ATTR(only_offset_flag, Bool, true)
    .OP_END_FACTORY_REG(EmbeddingFeatureMappingImport)

/**
* @brief insert table data in the embedding table. \n

* @par Inputs:
* @li table_name: A Tensor, dtype is string, indicates the hash table names.
* @li feature_id: Tensors which number is consistent with table's number, dtype is int64,
*                 indicates the original hash key for each table.
* @li offset_id: Tensors which number is consistent with table's number, with same shape of feature_id for each table,
*                dtype is int32. indicates the mapping value of hash key for each table. \n
*/
REG_OP(EmbeddingFeatureMappingInsert)
    .INPUT(table_name, TensorType({DT_STRING}))
    .DYNAMIC_INPUT(feature_id, TensorType({DT_INT64}))
    .DYNAMIC_INPUT(offset_id, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(EmbeddingFeatureMappingInsert)

/**
* @brief embedding compute var export. \n

* @par Inputs:
* @li file_path: A String, indicates the export file path.
* @li ps_id: A Int, dtype is int32, indicates the ps server id.
* @li table_id: A Int, dtype is int32, indicates the hashtable id.
* @li global_step: An optional Scalar, dtype is int32int64. 1-D. It indicates the ckpt export save step. \n

* @par Attributes:
* @li table_name: An optional string list that represents table name corresponding to table id. \n
*/
REG_OP(EmbeddingComputeVarExport)
    .INPUT(file_path, TensorType({DT_STRING}))
    .INPUT(ps_id, TensorType({DT_INT32}))
    .INPUT(table_id, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(global_step, TensorType({DT_INT32, DT_INT64}))
    .ATTR(table_name, ListString, {})
    .OP_END_FACTORY_REG(EmbeddingComputeVarExport)

/**
* @brief embedding compute var import. \n

* @par Inputs:
* @li file_path: A String, indicates the import file path.
* @li ps_id: A Int, dtype is int32, indicates the ps server id.
* @li table_id: A Int, dtype is int32, indicates the hashtable id.
* @li global_step: An optional Scalar, dtype is int32/int64. 1-D. indicates the ckpt import save step. \n

* @par Attributes:
* @li table_name: An optional string list that represents table name corresponding to table id. \n
*/
REG_OP(EmbeddingComputeVarImport)
    .INPUT(file_path, TensorType({DT_STRING}))
    .INPUT(ps_id, TensorType({DT_INT32}))
    .INPUT(table_id, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(global_step, TensorType({DT_INT32, DT_INT64}))
    .ATTR(table_name, ListString, {})
    .OP_END_FACTORY_REG(EmbeddingComputeVarImport)


/**
* @brief init embedding hash map table. \n

* @par Inputs:
* @li table_id: A scalar, dtype is int32, indicates the hash table id. \n

* @par Outputs:
* @li table_handle: A scalar, dtype is int64, indicates the hashmap info address. \n

* @par Attributes:
* @li bucket_size: A scalar, dtype is int64, indicates the hash bucket size.
* @li load_factor: A scalar, dtype is int64, indicates the hash load factor.
* @li embedding_dim: A scalar, dtype is int64, indicates the dim of embedding value in hash table.
* @li dtype: An optional attribute that indicates the type of value in hash table. Must be one of the following types:float32,double,int32,int64. Defaults to float32. \n
*/
REG_OP(InitEmbeddingHashmapV2)
    .INPUT(table_id, TensorType({DT_INT32}))
    .OUTPUT(table_handle, TensorType({DT_INT64}))
    .REQUIRED_ATTR(bucket_size, Int)
    .REQUIRED_ATTR(load_factor, Int)
    .REQUIRED_ATTR(embedding_dim, Int)
    .ATTR(dtype, Type, DT_FLOAT)
    .OP_END_FACTORY_REG(InitEmbeddingHashmapV2)

/**
* @brief uninit embedding hash map table. \n

* @par Inputs:
* @li table_id: A scalar, dtype is int32, indicates the hash table id. \n
*/
REG_OP(DeinitEmbeddingHashmapV2)
    .INPUT(table_id, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(DeinitEmbeddingHashmapV2)

/**
* @brief convert embedding hashmap table id to handle. \n

* @par Inputs:
* @li table_id: A scalar, dtype is int32, indicates the hash table id. \n

* @par Outputs:
* @li table_handle: A scalar, dtype is int64, indicates the hashmap info address. \n
*/
REG_OP(TableToResourceV2)
    .INPUT(table_id, TensorType({DT_INT32}))
    .OUTPUT(table_handle, TensorType({DT_INT64}))
    .OP_END_FACTORY_REG(TableToResourceV2)

/**
* @brief get export size of the embedding hashmap. \n

* @par Inputs:
* @li table_ids: A Tensor, dtype is int32, indicates the hash table id. \n

* @par Outputs:
* @li table_sizes: A Tensor, dtype is int64, indicates the size of hash map for each table. \n

* @par Attributes:
* @li filter_export_flag: An optional bool that represents filter export flag on counter filter scenario. Defaults to "false".
* @li export_mode: An optional string that is export mode. The value of export mode has two values: 
*     "all" and "new". Defaults to "all". \n
*/
REG_OP(EmbeddingHashmapSize)
    .INPUT(table_ids, TensorType({DT_INT32}))
    .OUTPUT(table_sizes, TensorType({DT_INT64}))
    .ATTR(filter_export_flag, Bool, false)
    .ATTR(export_mode, String, "all")
    .OP_END_FACTORY_REG(EmbeddingHashmapSize)

/**
* @brief Export table data from the embedding table to file. \n

* @par Inputs:
* @li file_path: A Scalar, dtype is string, indicates the file path to export.
                 There are two types of paths: folder path with a file name prefix such as /home/path/ckpt
                 and folder path without a file name prefix such as /home/path/.
* @li table_ids: A Tensor, dtype is int32, indicates the hash table id.
* @li table_names: A Tensor, dtype is string, indicates the hash table names.
* @li global_step: A Scalar, dtype is DT_INT32/DT_INT64. 0-D. indicates the export save step.
* @li keys: Tensors which number is consistent with table's number, dtype is int64,
*                 indicates the original hash key for one table.
* @li counters: Tensors which number is consistent with table's number, dtype is uint64,
*                 indicates the counters of each hashmap.
* @li filter_flags: Tensors which number is consistent with table's number, dtype is uint8,
*                 indicates the filter flag of each hashmap.
* @li values: Tensors which number is consistent with table's number, dtype is float32.
*                indicates the value of each hashmap. \n
*/
REG_OP(EmbeddingHashmapExport)
    .INPUT(file_path, TensorType({DT_STRING}))
    .INPUT(table_ids, TensorType({DT_INT32}))
    .INPUT(table_names, TensorType({DT_STRING}))
    .INPUT(global_step, TensorType({DT_INT32, DT_INT64}))
    .DYNAMIC_INPUT(keys, TensorType({DT_INT64}))
    .DYNAMIC_INPUT(counters, TensorType({DT_UINT64}))
    .DYNAMIC_INPUT(filter_flags, TensorType({DT_UINT8}))
    .DYNAMIC_INPUT(values, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(EmbeddingHashmapExport)

/**
* @brief get import size of the embedding hashmap file. \n

* @par Inputs:
* @li file_path: A Scalar, dtype is string, indicates the path of import file.
* @li table_ids: A Tensor, dtype is int32, indicates the hash table id.
* @li table_names: A Tensor, dtype is string, indicates the hash table names.
* @li global_step: A Scalar, dtype is DT_INT32/DT_INT64. 0-D. indicates the save step. \n

* @par Outputs:
* @li table_sizes: A Tensor, dtype is int64, indicates the size of hash map for each table. \n

* @par Attributes:
* @li embedding_dims: List of Int, indicates the length of embedding for each table. \n
*/
REG_OP(EmbeddingHashmapFileSize)
    .INPUT(file_path, TensorType({DT_STRING}))
    .INPUT(table_ids, TensorType({DT_INT32}))
    .INPUT(table_names, TensorType({DT_STRING}))
    .INPUT(global_step, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(table_sizes, TensorType({DT_INT64}))
    .REQUIRED_ATTR(embedding_dims, ListInt)
    .OP_END_FACTORY_REG(EmbeddingHashmapFileSize)

/**
* @brief Import table data from file to embedding hashmap. \n

* @par Inputs:
* @li file_path: A Scalar, dtype is string, indicates the path folder of import file.
* @li table_ids: A Tensor, dtype is int32, indicates the hash table id.
* @li table_sizes: A Tensor, dtype is int64, indicates the size of hash map for each table.
* @li table_names: A Tensor, dtype is string, indicates the hash table names.
* @li global_step: A Scalar, dtype is DT_INT32/DT_INT64. 0-D. indicates the import save step. \n

* @par Outputs:
* @li keys: Tensors which number is consistent with table's number, dtype is int64,
*                 indicates the original hash key for one table.
* @li counters: Tensors which number is consistent with table's number, dtype is uint64,
*                 indicates the counters of each hashmap.
* @li filter_flags: Tensors which number is consistent with table's number, dtype is uint8,
*                 indicates the filter flag of each hashmap.
* @li values: Tensors which number is consistent with table's number, dtype is float32.
*                indicates the value of each hashmap. \n

* @par Attributes:
* @li embedding_dims: List of Int, indicates the length of embedding for each table. \n
*/
REG_OP(EmbeddingHashmapImport)
    .INPUT(file_path, TensorType({DT_STRING}))
    .INPUT(table_ids, TensorType({DT_INT32}))
    .INPUT(table_sizes, TensorType({DT_INT64}))
    .INPUT(table_names, TensorType({DT_STRING}))
    .INPUT(global_step, TensorType({DT_INT32, DT_INT64}))
    .DYNAMIC_OUTPUT(keys, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(counters, TensorType({DT_UINT64}))
    .DYNAMIC_OUTPUT(filter_flags, TensorType({DT_UINT8}))
    .DYNAMIC_OUTPUT(values, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(embedding_dims, ListInt)
    .OP_END_FACTORY_REG(EmbeddingHashmapImport)

/**
* @brief Updates embedding hash table according to the AdamW algorithm.

* @par Inputs:
* @li table_handle: A Tensor, dtype is int64. 1-D. Stores the address of the hashtable.
* @li keys: A Tensor, dtype is int64. 1-D. Indicates the hashtable key.
*           Should be deduplicated, otherwise can not guarantee the correctness of the results.
* @li m: A Tensor, dtype is float16 or float, 2-D. Indicates the first moment estimate.
*        Shape is (bucket_size, embedding_dim).
* @li v: A Tensor, dtype is float16 or float, 2-D. Indicates the second moment estimate.
*        Shape is (bucket_size, embedding_dim).
* @li beta1_power: A Tensor, dtype is float16 or float. 1-D. Used to cache intermediate calculated values.
*                  Value is beta1*(step-1). Range is [0.0, 1.0).
* @li beta2_power: A Tensor, dtype is the same as "beta1_power". 1-D. Used to cache intermediate calculated values.
*                  Value is beta2*(step-1). Range is [0.0, 1.0).
* @li lr: A Tensor, dtype is the same as "beta1_power". 1-D. Indicates the learning rate. Range is [0.0, INF).
* @li weight_decay: A Tensor, dtype is the same as "beta1_power". 1-D. Indicates the weight decay. Range is [0.0, INF).
* @li beta1: A Tensor, dtype is the same as "beta1_power". 1-D. Indicates the first order momentum. Range is [0.0, 1.0).
* @li beta2: A Tensor, dtype is the same as "beta1_power". 1-D. Indicates the second order momentum. Range is [0.0, 1.0).
* @li epsilon: A Tensor, dtype is the same as "beta1_power". 1-D. Indicates the small value param. Range is [0.0, 1.0).
* @li grad: A Tensor, dtype is the same as "beta1_power". 2-D. Indicates the grad. Shape is (keys number, embedding_dim).
* @li max_grad_norm: A mutable Tensor of the same type as "m". Indicates the gradient parameter.
*                    Shape is (bucket_size, embedding_dim).

* @par Outputs:
* @li m: A Tensor, dtype is float16 or float, 2-D. Update of the input m value.
* @li v: A Tensor, dtype is float16 or float, 2-D. Update of the input v value.
* @li beta1_power: Update the value of input beta1_power.
*                  Used to cache intermediate calculated values. Dtype is float16 or float. 1-D.
* @li beta2_power: Update the value of input beta2_power.
*                  Used to cache intermediate calculated values. Dtype is the same as "beta1_power". 1-D.
* @li max_grad_norm: A mutable Tensor of the same type as "m". Update of the input max_grad_norm value.\n

* @par Attributes:
* @li embedding_dim: An Int, indicates the dim of embedding value in hashtable.
* @li bucket_size: An Int, indicates the size of hash map.
* @li amsgrad: An optional bool, indicates whether to use the AMSGrad variant of htis algorithm from
*     the paper On the Convergence of Adam and Beyond.
*     If "True", max_grad_norm input and output must be entered.
* @li maximize: An optional bool, maximize the params based on the objective. \n
*/
REG_OP(EmbeddingHashTableApplyAdamW)
    .INPUT(table_handle, TensorType({DT_INT64}))
    .INPUT(keys, TensorType({DT_INT64}))
    .INPUT(m, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(v, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(beta1_power, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(beta2_power, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(lr, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(weight_decay, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(beta1, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(beta2, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(epsilon, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(grad, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(max_grad_norm, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(m, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(v, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(beta1_power, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(beta2_power, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(max_grad_norm, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(embedding_dim, Int)
    .REQUIRED_ATTR(bucket_size, Int)
    .ATTR(amsgrad, Bool, false)
    .ATTR(maximize, Bool, false)
    .OP_END_FACTORY_REG(EmbeddingHashTableApplyAdamW)

/**
* @brief fake remote lookup host unique. \n

* @par Inputs:
* @li table_id: A Tensor, dtype is DT_INT32. 0-D. indicates the id of hashtable.
* @li keys: A Tensor, dtype is DT_INT64. 1-D. indicates the hashtable key.
* @li actual_keys_num: dtype is DT_INT64. 1-D. indicates the actual hashtable key to host.
* @li unique_indices: A Tensor, dtype is DT_INT32. indicates the unique indices.
* @li key_count: An optional input Tensor, dtype is DT_INT64. 1-D. indicates the count of each key. \n

* @par Outputs:
* @li values: indicates the hashtable value. \n

* @par Attributes:
* @li embedding_dim: Int list, indicates the dim of embedding var value in hashtable.
* @li value_total_len: Int list, indicates the dim of embedding var+m+v or var+accum values in hashtable.
* @li initializer_mode: An optional string list of "random_uniform", "truncated_normal" or "constant".
* indicates the algo of init method. Defaults to "random_uniform".
* @li constant_value: An optional float List, used when initializer_mode is "constant". Defaults to "0".
* @li min: An optional float list, used when initializer_mode is "truncated_normal", the minimum value of the random number.
* Defaults to "-2".
* @li max: An optional float list, used when initializer_mode is "truncated_normal", the maximum value of the random number.
* Defaults to "2".
* @li mu: An optional float List, used when initializer_mode is "truncated_normal", The mean of the truncated_normal.
* Defaults to "0".
* @li sigma: An optional float list, used when initializer_mode is "truncated_normal", The variance of the truncated_normal.
* Defaults to "1".
* @li seed: An optional int list, used to create a random seed. Defaults to "0".
* @li seed2: An optional int list, used to create a random seed. Defaults to "0".
* @li filter_mode: An optional string list of "no_filter" or "counter". indicates the type of the hashmap, Defaults to "no_filter".
* @li filter_freq: An optional int list, used to set the threshold of the tal. Defaults to "0".
* @li default_key_or_value: An optional int list, indicates the default value get way.
* @li default_key: An optional int list, when default_key_or_value is true, use the default_key corresponding value as default value.
* @li default_value: An optional int list, when default_key_or_value is false, use the default_value as default value.
* @li completion_key: An optional int list, indicates the completion hashtable key.
* @li completion_key_mask: An optional int list, whether to perform no-update interception when key==completion_key.
* @li optimizer_mode: An optional string list of "adam" or "adamw" or "adagrad". indicates the type of the optimizer_mode,
* Defaults to "".
* @li optimizer_params: An optional float list, when optimizer_mode is "adagrad", the initialize value of the optimizer. \n
*/
REG_OP(FakeRemoteLookupUniqued)
    .INPUT(table_id, TensorType({DT_INT32}))
    .INPUT(keys, TensorType({DT_INT64}))
    .INPUT(actual_keys_num, TensorType({DT_INT64}))
    .INPUT(unique_indices, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(key_count, TensorType({DT_INT64}))
    .OUTPUT(values, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(embedding_dim, ListInt)
    .REQUIRED_ATTR(value_total_len, ListInt)
    .ATTR(initializer_mode, ListString, {"random_uniform"})
    .ATTR(constant_value, ListFloat, {0})
    .ATTR(min, ListFloat, {-2})
    .ATTR(max, ListFloat, {2})
    .ATTR(mu, ListFloat, {0})
    .ATTR(sigma, ListFloat, {1})
    .ATTR(seed, ListInt, {0})
    .ATTR(seed2, ListInt, {0})
    .ATTR(filter_mode, ListString, {"no_filter"})
    .ATTR(filter_freq, ListInt, {0})
    .ATTR(default_key_or_value, ListInt, {0})
    .ATTR(default_key, ListInt, {0})
    .ATTR(default_value, ListFloat, {0})
    .ATTR(completion_key, ListInt, {0})
    .ATTR(completion_key_mask, ListInt, {1})
    .ATTR(optimizer_mode, ListString, {})
    .ATTR(optimizer_params, ListFloat, {})
    .OP_END_FACTORY_REG(FakeRemoteLookupUniqued)

/**
* @brief Step the original data block of y forward one by one, 
* discard the first data block, and then update x to the last data block of y. \n

* @par Inputs:
* @li x: A Tensor, A tensor that stores updated data.
*        Must be one of the following types:
*        int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float, bfloat16
*        double, complex64, complex128, qint8, qint16, qint32, quint8, quint16.
* @li clean_cache: A Bool, Indicates whether to reset y to zero first. \n

* @par Attributes:
* @li axis: A int, dtype is int64. Specify which dimension to start updating elements from. \n
* @li cache_depth: A int, dtype is int64. Specify the depth of data caching. \n

* @par Outputs:
* @li y: The updated tensor. Dtype is same as x. \n
*/
REG_OP(FillWindowCache)
    .INPUT(x, TensorType::BasicType())
    .INPUT(clean_cache, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(axis, Int)
    .REQUIRED_ATTR(cache_depth, Int)
    .OP_END_FACTORY_REG(FillWindowCache)

/**
* @brief Generate rgb and frame images into a out image with alpha transparency. \n

* @par Inputs:
* @li rgb: A Int, dtype is uint8, rgb images data.
* @li alpha: A Int, dtype is uint8, alpha transparency images data.
* @li frame: A Int, dtype is uint8, frame images data.  \n

* @par Outputs:
* @li out: The out tensor. Dtype is same as rgb. \n
*/
REG_OP(BlendImagesCustom)
    .INPUT(rgb, TensorType({DT_UINT8}))
    .INPUT(alpha, TensorType({DT_UINT8}))
    .INPUT(frame, TensorType({DT_UINT8}))
    .OUTPUT(out, TensorType({DT_UINT8}))
    .OP_END_FACTORY_REG(BlendImagesCustom)
} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_NN_OTHER_H_WARN_SHOWN
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_OTHER_H_
