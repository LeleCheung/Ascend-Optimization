/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file nn_batch_norm_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_BATCH_NORM_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_BATCH_NORM_OPS_H_

#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_BATCH_NORM_OPS_H_WARN_SHOWN
#define OPS_BUILT_IN_OP_PROTO_INC_NN_BATCH_NORM_OPS_H_WARN_SHOWN
#warning "nn_batch_norm_ops.h is scheduled to be deprecated in December 2026, and will be replaced by the \
ops_proto_math.h, ops_proto_cv.h, ops_proto_nn.h, ops_proto_transformer.h, ops_proto_legacy.h. \
We apologize for any inconvenience caused and appreciate your timely migration to the new interface."

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Normalizes elements of a specific dimension of eigenvalues (L2) .

*@par Inputs:
*x: A ND Tensor(1D-8D) of type float16 or float32, specifying the eigenvalue . \n

*@par Attributes:
*@li axis: A optional required attribute of type list, specifying the axis for normalization Defaults to {} .
*@li eps: An optional attribute of type float, specifying the lower limit of normalization. Defaults to "1e-4" . \n

*@par Outputs:
*y: A ND Tensor(1D-8D) of type float16 or float32, specifying the eigenvalue for normalization. \n

*@par Third-party framework compatibility
* Compatible with the L2 scenario of PyTorch operator Normalize.
*/
REG_OP(L2Normalize)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(axis, ListInt, {})
    .ATTR(eps, Float, 1e-4f)
    .OP_END_FACTORY_REG(L2Normalize)

/**
*@brief Performs the backpropagation of L2Normalize for training scenarios .

*@par Inputs:
* Three inputs, including:
*@li x: A ND Tensor(1D-8D) of type float16 or float32, specifying
* the eigenvalue of forward inputs.
*@li y: A ND Tensor(1D-8D) of type float16 or float32, specifying
* the normalization result of the forward output. the same shape with x.
*@li dy: A ND Tensor(1D-8D) of type float16 or float32, specifying
* the reverse input gradient. the same shape with x . \n

*@par Attributes:
*@li dim: A required attribute of type int, specifying the axis to be
* normalized.  Defaults to {}.
*@li eps: An optional attribute of type float, specifying the lower limit of
* normalization. Defaults to "1e-4" . \n

*@par Outputs:
*dx: A ND Tensor(1D-8D), Reverse gradient of eigenvalue "x". Has the same shape as "x" . \n

*@par Third-party framework compatibility
* Compatible with the L2 scenario of PyTorch operator NormalizeGrad.
*/
REG_OP(L2NormalizeGrad)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(dx, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(dim, ListInt, {})
    .ATTR(eps, Float, 0.0001f)
    .OP_END_FACTORY_REG(L2NormalizeGrad)

/**
* @brief Performs batch normalization with support for 4D/5D tensors and training/inference modes.
* 
* @par Inputs:
* Five inputs, with format constraints as follows:
* @li x: A 4D or 5D tensor of type float16, bfloat16, or float32. 
*        Supported data formats:
*        - 4D: NHWC (batch, height, width, channels) or NCHW (batch, channels, height, width).
*        - 5D: NDHWC (batch, depth, height, width, channels) or NCDHW (batch, channels, depth, height, width).
* @li scale: A 1D tensor of type float32, with length equal to the number of channels in "x".
*        Specifies the scaling factor (gamma) applied after normalization.
* @li offset: A 1D tensor of type float32, with length equal to the number of channels in "x".
*        Specifies the offset (beta) applied after scaling.
* @li mean: A 1D tensor of type float32, with length equal to the number of channels in "x".
*        - Inference mode (is_training=false): Must be provided as input, representing the 
*          moving mean computed during training.
*        - Training mode (is_training=true): Optional input. When provided, will be used to 
*          initialize the moving mean for updates; when None, moving mean starts from zeros.
* @li variance: A 1D tensor of type float32, with length equal to the number of channels in "x".
*        - Inference mode (is_training=false): Must be provided as input, representing the
*          moving variance computed during training.
*        - Training mode (is_training=true): Optional input. When provided, will be used to
*          initialize the moving variance for updates; when None, moving variance starts from ones.
* 
* @par Attributes:
* @li epsilon: Optional float32. Small value added to variance to avoid division by zero.
*        Defaults to 0.0001f.
* @li data_format: Optional string. Specifies the data format of "x". 
*        Allowed values: "NHWC" (4D default), "NCHW" (4D), "NDHWC" (5D), "NCDHW" (5D).
* @li is_training: Optional bool. Specifies operation mode:
*        - true: Training mode (computes batch mean/variance and updates moving stats).
*        - false: Inference mode (uses provided mean/variance for normalization).
*        Defaults to true.
* @li exponential_avg_factor: Optional float32. Factor for updating moving averages during training.
*        Formula: new_mean = (1 - factor) * old_mean + factor * batch_mean.
*        Defaults to 1.0f.
* 
* @par Outputs:
* Up to six outputs, with shape and format matching "x" unless specified:
* @li y: A tensor with the same rank (4D/5D), type, and format as "x", containing normalized values.
*        (Required output)
* @li batch_mean: A 1D tensor of type float32 (channel dimension).
*        - Training mode: Mean of the current batch (computed over spatial dimensions).
*        - Inference mode: Equal to input "mean" (for compatibility).
*        (Required output)
* @li batch_variance: A 1D tensor of type float32 (channel dimension).
*        - Training mode: Variance of the current batch (computed over spatial dimensions, with Bessel's correction).
*        - Inference mode: Equal to input "variance" (for compatibility).
*        (Required output)
* @li reserve_space_1: Optional 1D tensor of type float32 (channel dimension).
*        Reserved for gradient computation.
*        - Training mode: Same as batch_mean.
*        - Inference mode: Same as input "mean".
* @li reserve_space_2: Optional 1D tensor of type float32 (channel dimension).
*        Reserved for gradient computation.
*        - Training mode: saved inv_var (1/sqrt(epsilon + variance), to be reused in the backward gradient computation.
*        - Inference mode: Same as input "variance".
* @li reserve_space_3: A 1D tensor of type float32 with exactly one element.
*        Exists solely for TensorFlow compatibility and contains no meaningful data.
*/
REG_OP(BatchNorm)
    .INPUT(x, "T1")
    .INPUT(scale, "T2")
    .INPUT(offset, "T2")
    .OPTIONAL_INPUT(mean, "T2")
    .OPTIONAL_INPUT(variance, "T2")
    .OUTPUT(y, "T1")
    .OUTPUT(batch_mean, "T2")
    .OUTPUT(batch_variance, "T2")
    .OUTPUT(reserve_space_1, "T2")
    .OUTPUT(reserve_space_2, "T2")
    .OUTPUT(reserve_space_3, "T2")
    .ATTR(epsilon, Float, 1e-4f)
    .ATTR(data_format, String, "NHWC")
    .ATTR(is_training, Bool, true)
    .ATTR(exponential_avg_factor, Float, 1.0f)
    .DATATYPE(T1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T2, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BatchNorm)

/**
* @brief After the mean and reciprocal of standard deviation(invert_std) are separately calculated on each device,
* the mena and reciprocal of standard deviation(invert_std) data on each device are normlized,
* a total mean and reciprocal of standard deviation(invert_std) are returned, and running_var are updated.

* @par Inputs:
* include:
* @li mean_all: A Tensor. The mean of each device. Must be one of the following types: float16, float32, bfloat16.
* @li invert_std_all: A Tensor. Reciprocal of the variances of each device. Must be one of the following types: float16, float32, bfloat16.
* @li count_all: A Tensor. Number of data for each device. Must be one of the following types: float16, float32, bfloat16.
* @li mean_broadcast: A Tensor. The overall average and broadcast. Must be one of the following types: float16, float32, bfloat16.
* @li count_sum: A Tensor. General statistics. Must be one of the following types: float16, float32, bfloat16.
* @li running_var: A Tensor. Runtime variance. Must be one of the following types: float16, float32, bfloat16. \n

* @par Attributes:
* Two Attributes, including:
* @li momentum: An optional float. Defaults to 0.01. \n
* @li epsilon: An optional float. Defaults to 0.00001. \n

* @par Outputs:
* include:
* @li invert_std: A Tensor. It's inverse of total variance. Must be one of the following types: float16, float32, bfloat16.
* @li running_var_update: A Tensor. It's moving variance of each device after the update. Must be one of the following types: float16, float32, bfloat16. \n

* @par Third-party framework compatibility
* ReduceMeanWithCount and SyncBatchNormGatherStatsWithCounts and SyncBNTrainingUpdate
* compatible with the Pytorch operator BatchNormGatherStatsWithCounts.
*/
REG_OP(SyncBatchNormGatherStatsWithCounts)
    .INPUT(mean_all, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(invert_std_all, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(count_all, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(mean_broadcast, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(count_sum, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(running_var, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(invert_std, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(running_var_update, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(momentum, Float, 0.1f)
    .ATTR(epsilon, Float, 0.001f)
    .OP_END_FACTORY_REG(SyncBatchNormGatherStatsWithCounts)

/**
* @brief update running_mean.

* @par Inputs:
* include:
* @li mean: A Tensor. The mean of each device. Must be one of the following types: float16, float32, bfloat16.
* @li running_mean: A Tensor. Runtime Mean. Must be one of the following types: float16, float32, bfloat16. \n

* @par Attributes:
* One Attribute, including:
* @li momentum: A optional float. Defaults to 0.01. \n

* @par Outputs:
* include:
* @li running_mean_update: A Tensor. It's moving mean of each device after the update. Must be one of the following types: float16, float32, bfloat16.\n

* @par Third-party framework compatibility
* ReduceMeanWithCount and SyncBatchNormGatherStatsWithCounts and SyncBNTrainingUpdate
* compatible with the Pytorch operator BatchNormGatherStatsWithCounts.
*/
REG_OP(SyncBNTrainingUpdate)
    .INPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(running_mean, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(running_mean_update, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(momentum, Float, 0.1f)
    .OP_END_FACTORY_REG(SyncBNTrainingUpdate)

/**
*@brief part of SyncBatchNormBackward .

*@par Inputs:
* Four inputs, including:
*@li sum_dy: A ND tensor. Represents the sum of the gradients of the loss function output to the batch normalization layer.
* Must be one of the following types: float16, float32, bfloat16.
*@li sum_dy_dx_pad: A ND tensor. Represents the sum of the gradients of the input of the loss function to the batch normalization layer.
* Must be one of the following types: float16, float32, bfloat16. Has the same type, shape and format as "sum_dy".
*@li mean: A ND tensor. The mean value of the input data calculated during forward propagation.
* Must be one of the following types: float16, float32, bfloat16. Has the same type, shape and format as "sum_dy".
*@li invert_std: A ND tensor. The reciprocal of the input data standard deviation calculated during forward propagation.
* Must be one of the following types: float16, float32, bfloat16. Has the same type, shape and format as "sum_dy". \n

*@par Outputs:
*@li sum_dy_xmu: A ND tensor. Represents the sum of the gradients of the loss function against the mean.
* Has the same type, shape and format as "sum_dy".
*@li y: A ND tensor. Indicates the adjusted gradient, which is used for backpropagation to the previous layer.
* Has the same type, shape and format as "sum_dy". \n
*/
REG_OP(SyncBatchNormBackwardReduce)
    .INPUT(sum_dy, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(sum_dy_dx_pad, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mean, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(invert_std, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(sum_dy_xmu, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(SyncBatchNormBackwardReduce)

/**
*@brief part of SyncBatchNormBackward .

*@par Inputs:
* Seven inputs, including:
*@li grad_output: A ND tensor(2D-8D). Forward output differential. Must be one of the following types: float16, bfloat16, float32.
*@li save_input: A ND tensor(2D-8D). The input of forward. Has the same shape, format and type as input "grad_output".
*@li mean: A ND tensor(2D-8D). Mean of saved forward input. Has the same shape, format and type as input "grad_output".
*@li invstd: A ND tensor(2D-8D). Reciprocal of the standard deviation of the saved forward input. Has the same shape, format and type as input "grad_output".
*@li weight: A ND tensor(2D-8D). The weight parameter. Has the same shape, format and type as input "grad_output".
*@li mean_dy: A ND tensor(2D-8D). A part of sum_dy. Has the same shape, format and type as input "grad_output".
*@li mean_dy_xmu: A ND tensor(2D-8D). A part of sum_dy_xmu. Has the same shape, format and type as input "grad_output". \n

*@par Outputs:
* grad_input: A ND tensor(2D-8D). The reverse gradient corresponding to the input in the forward calculation. Has the same shape, format and type as input "grad_output". \n
*/
REG_OP(SyncBatchNormBackwardElemt)
    .INPUT(grad_output, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(save_input, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(mean, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(invstd, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(weight, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(mean_dy, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(mean_dy_xmu, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(grad_input, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OP_END_FACTORY_REG(SyncBatchNormBackwardElemt)

/**
* @brief After the sum(total_sum) and the square sum(total_square_sum) are separately calculated on each device,
* a total mean(batch_mean) and reciprocal of standard deviation(batch_invstd) are returned,
* running_mean and running_var are updated.

* @par Inputs:
* include:
* @li total_sum: A 2-D tensor, that is, [N, C]. The sum of each device. The format must be ND.
* Must be one of the following types: bfloat16, float16, float32.
* @li total_square_sum: A 2-D tensor, that is, [N, C]. The format must be ND. The square sum of each device.
* Must be one of the following types: bfloat16, float16, float32. Has the same type, shape and format as "total_sum".
* @li sample_count: A 1-D tensor. Number of data for each device. The format must be ND.
* Must be one of the following types: int32. The value of "sample_count" needs to be consistent with the N-axis value of "total_sum".
* @li mean: A 1-D tensor. Runtime mean. The format must be ND.
* Must be one of the following types: bfloat16, float16, float32. The value of "mean" needs to be consistent with the C-axis value of "total_sum".
* @li variance: A 1-D tensor. Runtime variance. The format must be ND.
* Must be one of the following types: bfloat16, float16, float32. Has the same type, shape and format as input "mean".
* The value of "variance" needs to be consistent with the C-axis value of "total_sum". \n

* @par Attributes:
* Two Attributes, including:
* @li momentum: An optional float. Control the update speed of the moving average. Defaults to 0.1. \n
* @li eps: An optional float. A very small value to prevent division by zero. Defaults to 0.00001. \n

* @par Outputs:
* include:
* @li batch_mean: A 1-D tensor. Total mean of this batch.
* Must be one of the following types: bfloat16, float16, float32. Has the same type, shape and format as input "mean".
* @li batch_invstd: A 1-D tensor. General statistics.
* Must be one of the following types: bfloat16, float16, float32. Has the same type, shape and format as input "mean".
* @li mean: A 1-D tensor. Updated Runtime mean.
* Must be one of the following types: bfloat16, float16, float32. Has the same type, shape and format as input "mean".
* @li variance: A 1-D tensor. Updated Runtime variance.
* Must be one of the following types: bfloat16, float16, float32. Has the same type, shape and format as input "mean". \n
*/
REG_OP(SyncBatchNormGatherStats)
    .INPUT(total_sum, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(total_square_sum, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(sample_count, TensorType({DT_INT32}))
    .INPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(variance, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(batch_invstd, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(variance, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(momentum, Float, 0.1f)
    .ATTR(eps, Float, 0.00001f)
    .OP_END_FACTORY_REG(SyncBatchNormGatherStats)

/**
*@brief Performs batch normalization .

*@par Inputs:
* Five inputs, including: (NDHWC, NCDHW)
*@li x: A 5D Tensor of type float16 or float32, with format NDHWC or NCDHW.
*@li scale: A Tensor of type float32. Must be 1D if input "x" is with format NDHWC or NCDHW. 
Specifies the scaling factor.
*@li offset: A Tensor of type float32. Must be 3D if input "x" is with format NDHWC or NCDHW.
Specifies the offset.
*@li mean: A Tensor of type float32. Must be 3D if input "x" is with format NDHWC or NCDHW.
Specifies the mean used for inference. Must be "None" if the
operation is used for training.
*@li variance: A Tensor of type float32. Must be 3D if input "x" is with format NHWC or NCHW.
Specifies the variance used for inference. Must be "None"
if the operation is used for training . \n

*@par Attributes:
*@li epsilon: An optional float32, specifying the small value added to variance to avoid dividing by zero. Defaults to "0.0001".
*@li data_format: An optional string, specifying the format of "x". Defaults to "NCDHW".
*@li is_training: An optional bool, specifying if the operation is used for training or inference. Defaults to "True" . \n

*@par Outputs:
* Five outputs, including: (NDHWC, NCDHW)
*@li y: A 5D Tensor of type float16 or float32 for the normalized "x", with format NDHWC or NCDHW.
*@li batch_mean: A Tensor of type float32. Must be 3D if input "x" is with format NDHWC or NCDHW.
Specifies the mean of "x".
*@li batch_variance: A Tensor of type float32. Must be 1D if input "x" is with format NDHWC or NCDHW.
Specifies the variance of "x".
*@li reserve_space_1: An optional Tensor of type float32. Must be 1D if input "x" is with format NDHWC or NCDHW.
Specifies the mean of "x" for gradient computation. Pass "None" to skip this output.
*@li reserve_space_2: An optional Tensor of type float32. Must be 1D if input "x" is with format NHWC or NCHW.
Specifies the variance of "x" for gradient computation. Pass "None" to skip this output . \n

*@attention Constraints:
*@li If the operation is used for inference and outputs "reserve_space_1" and "reserve_space_2" are available,
then "reserve_space_1" has the same value as "mean" and "reserve_space_2" has the same value as "variance".
*@li For Atlas 200/300/500 Inference Product, the result accuracy fails to reach 1‰ due to the square root instruction . \n

*@par Third-party framework compatibility
*@li Compatible with the TensorFlow operator fused_batch_norm.
*@li Compatible with the TensorFlow operator fused_batch_norm_v2.
*/
REG_OP(BatchNorm3D)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(mean, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(variance, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_1, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_2, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001f)
    .ATTR(data_format, String, "NCDHW")
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(BatchNorm3D)
/**
*@brief Performs batch normalization .

*@par Inputs:
* Five inputs, including: (NHWC or NCHW supported)
*@li input_x: A 4D Tensor of type float16 or float32.
*@li input_scale: A 1D Tensor of type float32, for the scaling factor.
*@li input_offset: A 1D Tensor of type float32, for the scaling offset.
*@li input_mean: A 1D Tensor of type float32, for the mean used for inference.
* This cannot be used if the operation is used for training.
*@li input_variance: A 1D Tensor of type float32, for the variance used for inference.
* This cannot be used if the operation is used for training . \n

*@par Attributes:
*@li epsilon: An optional float32, specifying the small value
added to variance to avoid dividing by zero. Defaults to "0.0001".
*@li data_format: An optional string, specifying the format of "x". Defaults to "NHWC".
*@li is_training: An optional bool, specifying if the operation
is used for training or inference. Defaults to "True" . \n

*@par Outputs:
* Five outputs, including: (NHWC or NCHW supported)
*@li output_y: A 4D Tensor of type float16 or float32, for the normalized "x".
*@li output_mean: A 1D Tensor of type float32, for the mean of "x".
*@li output_variance: A 1D Tensor of type float32, for the variance of "x".
*@li output_reserve_space_1: A 1D Tensor of type float32, for the mean of "x" for gradient computation.
*@li output_reserve_space_2: A 1D Tensor of type float32, for the variance of "x" for gradient computation . \n

*@attention Constraints:
*@li If the operation is used for inference, then output "reserve_space_1"
has the same value as "mean" and output "reserve_space_2" has the same value as "variance".
*@li For Atlas 200/300/500 Inference Product, the result accuracy fails to reach 1‰ due to the square root instruction . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator fused_batch_norm_v2.
*/
REG_OP(BatchNormExt2)
    .INPUT(input_x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input_scale, TensorType({DT_FLOAT}))
    .INPUT(input_offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(input_mean, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(input_variance, TensorType({DT_FLOAT}))
    .OUTPUT(output_y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(output_mean, TensorType({DT_FLOAT}))
    .OUTPUT(output_variance, TensorType({DT_FLOAT}))
    .OUTPUT(output_reserve_space_1, TensorType({DT_FLOAT}))
    .OUTPUT(output_reserve_space_2, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001f)
    .ATTR(data_format, String, "NHWC")
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(BatchNormExt2)

/**
*@brief Performs the backpropagation of BatchNorm.

*@par Inputs:
* Six inputs, including:
*@li y_backprop: A 4D or 5D Tensor of type bfloat16, float16 or float32, with format NCHW, NHWC, NDHWC or NCDHW, for the gradient.
*@li x: A 4D or 5D Tensor of type bfloat16, float16 or float32, with format NCHW, NHWC, NDHWC or NCDHW, the same shape with "y_backprop".
*@li scale: A 1D Tensor of type float32, with format ND, shape must be C channel.
*@li reserve_space_1: A 1D Tensor of type float32, with format ND, shape must be C channel. It is an output of BatchNorm.
* When in training mode, it represents the saved mean of "x". And in inference mode, it represents the running mean of "x".
*@li reserve_space_2: A 1D Tensor of type float32, with format ND, shape must be C channel. It is an output of BatchNorm.
*@li reserve_space_3: A 1D optional Tensor of type float32, with format ND. Not used and not involved in calculations. When in training mode,
* it represents the saved inverse standard deviation of "x" And in inference mode, it represents the running variance of "x".  \n

*@par Attributes:
*@li epsilon: An optional float32. Defaults to "1e-4". A small float number used to add with running variance of "x" in inference mode.
*@li data_format: An optional string. Defaults to "NHWC". Should be same as y_backprop/x/x_backprop's dtype.
*@li is_training: An optional bool. Defaults to "true". Specifies the operation is for training (default) or inference.
*@li output_mask: An optional ListBool. Defaults to [true, false, false]. Valid only in inference mode, it determines whether
* the outputs "x_backprop", "scale_backprop" and "offset_backprop" contain actual reseluts. \n

*@par Outputs:
*@li x_backprop: A 4D or 5D Tensor of type bfloat16, float16 or float32, with format NCHW, NHWC, NDHWC or NCDHW. For the offset of "x".
* the same shape with "x".
*@li scale_backprop: A Tensor of type float32, with format ND, for the offset of "scale", the same shape format with scale.
*@li offset_backprop: A Tensor of type float32, with format ND, for the offset of "offset", the same shape format with scale.
*@li reserve_space_4: A Tensor of type float32, with shape ND. The same shape format with scale.
*@li reserve_space_5: A Tensor of type float32, with shape ND. The same shape format with scale. \n

*@attention Constraints:
* The preceding layer of this operator must be operator BatchNorm . \n

*@see BatchNorm
*@par Third-party framework compatibility
* Compatible with the TensorFlow operators FusedBatchNormGrad, FusedBatchNormGradV2 and FusedBatchNormGradV3.
*/
REG_OP(BatchNormGrad)
    .INPUT(y_backprop, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(reserve_space_1, TensorType({DT_FLOAT}))
    .INPUT(reserve_space_2, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(reserve_space_3, TensorType({DT_FLOAT}))
    .OUTPUT(x_backprop, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(scale_backprop, TensorType({DT_FLOAT}))
    .OUTPUT(offset_backprop, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_4, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_5, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 1e-4)
    .ATTR(data_format, String, "NHWC")
    .ATTR(is_training, Bool, true)
    .ATTR(output_mask, ListBool, {true, false, false})
    .OP_END_FACTORY_REG(BatchNormGrad)


/**
*@brief Performs the backpropagation of BatchNorm .

*@par Inputs:
* Five inputs, including:
*@li y_backprop: A 5D Tensor of type float16 or float32, with format NDHWC, NCDHW, for the gradient.
*@li x: A 5D Tensor of type float16 or float32, with format NDHWC, NCDHW.
*@li scale: A 5D Tensor of type float32, with format NDHWC, NCDHW.
*@li reserve_space_1: A 5D Tensor of type float32, with format NDHWC, NCDHW. It is an output of BatchNorm.
*@li reserve_space_2: A 5D Tensor of type float32, with format NDHWC, NCDHW. It is an output of BatchNorm . \n

*@par Attributes:
*@li epsilon: An optional float32. Defaults to "0.0001". A small float number added to the variance of "x".
*@li data_format: An optional string. Defaults to "NCDHW".
*@li is_training: An optional bool. Defaults to "true". Specifies the operation is for training (default) or inference . \n

*@par Outputs:
*@li x_backprop: A Tensor of type float16 or float32, with format NDHWC, NCDHW, for the offset of "x".
*@li scale_backprop: A Tensor of type float32, with format NDHWC, NCDHW, for the offset of "scale".
*@li *offset_backprop: A Tensor of type float32, with format NDHWC, NCDHW, for the offset of "offset".
*@li *reserve_space_4: A Tensor of type float32, with shape NDHWC, NCDHW. Pass "None" to skip this output.
*@li *reserve_space_5: A Tensor of type float32, with shape NDHWC, NCDHW. Pass "None" to skip this output . \n

*@attention Constraints:
* The preceding layer of this operator must be operator BatchNorm . \n

*@see BatchNorm
*@par Third-party framework compatibility
* Compatible with the TensorFlow operators FusedBatchNormGradV2 and FusedBatchNorm3DGrad.
*/
REG_OP(BatchNorm3DGrad)
    .INPUT(y_backprop, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(reserve_space_1, TensorType({DT_FLOAT}))
    .INPUT(reserve_space_2, TensorType({DT_FLOAT}))
    .OUTPUT(x_backprop, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(scale_backprop, TensorType({DT_FLOAT}))
    .OUTPUT(offset_backprop, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_4, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_5, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001f)
    .ATTR(data_format, String, "NCDHW")
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(BatchNorm3DGrad)

/**
*@brief Performs the backpropagation of BatchNorm .

*@par Inputs:
* Five inputs, including:
*@li y_backprop: A 4D Tensor of type float16 or float32, with format NHWC or NCHW, for the gradient.
*@li x: A 4D Tensor of type float16 or float32, with format NHWC or NCHW, the shape is same as input y_backprop.
*@li scale: A 4D Tensor of type float32, with format NHWC or NCHW, the shape is same as input y_backprop.
*@li reserve_space_1: A 4D Tensor of type float32, with format NHWC or NCHW, the shape is same as input y_backprop,
* it is an output of BatchNormExt2.
*@li reserve_space_2: A 4D Tensor of type float32, with format NHWC or NCHW, the shape is same as input y_backprop,
* it is an output of BatchNormExt2 . \n

*@par Attributes:
*@li epsilon: A required float32. A small float number added to the variance of "x".
*@li data_format: A required string for the format.
*@li is_training: A required bool for specifying the operation is for training (true) or inference (false) . \n

*@par Outputs:
*@li x_backprop: A Tensor of type float16 or float32, with format NHWC or NCHW, for the offset of "x".
*@li scale_backprop: A Tensor of type float32, with format NHWC or NCHW, for the offset of "scale".
*@li offset_backprop: A Tensor of type float32, with format NHWC or NCHW, for the offset of "offset".
*@li reserve_space_3: A Tensor of type float32, with format NHWC or NCHW.
*@li reserve_space_4: A Tensor of type float32, with format NHWC or NCHW . \n

*@attention Constraints:
* The preceding layer of this operator must be BatchNormExt2 . \n

*@see BatchNormExt2
*@par Third-party framework compatibility
* Compatible with the TensorFlow operator FusedBatchNormGradV2.
*/
REG_OP(BatchNormGradExt2)
    .INPUT(y_backprop, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(reserve_space_1, TensorType({DT_FLOAT}))
    .INPUT(reserve_space_2, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001f)
    .ATTR(data_format, String, "NHWC")
    .ATTR(is_training, Bool, true)
    .OUTPUT(x_backprop, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(scale_backprop, TensorType({DT_FLOAT}))
    .OUTPUT(offset_backprop, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_3, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_4, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BatchNormGradExt2)


/**
* @brief Performs batch normalization .

* @par Inputs:
* @li x: A 4D or 5D Tensor of type float16 or float32 or bfloat16, with format NHWC or NCHW.
* @li mean: A 1D Tensor of type float32 or float16 or bfloat16, the shape is same as dim C of input x.
* Specifies the mean used for inference.
* @li variance: A 1D Tensor of type float32 or float16 or bfloat16, the shape is same as dim C of input x.
* Specifies the variance used for inference.
* @li momentum: A 1D Tensor of type float32 or float16 or bfloat16, the shape is same as dim C of input x.
* represents the mean and the variance's scale factor
* @li scale: An optional 1D tensor of type float16 or float32 or bfloat16, the shape is same as dim C of input x.
* @li offset: An optional 1D tensor of type float16 or float32 or bfloat16, the shape is same as dim C of input x.
* @par Attributes:
* @li epsilon: An optional float32, specifying the small value added to variance to avoid dividing by zero.
      Defaults to "0.00001".
* @li use_global_stats: An optional bool, mean inference mode, only can be "True".
* @li mode: An optional int, defaults to "1".
* @par Outputs:
* @li y: A 4D or 5D Tensor of type float16 or float32 or bfloat16 for the normalized "x"
*/
REG_OP(BNInference)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mean, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(variance, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(momentum, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(epsilon, Float,1e-5f)
    .ATTR(use_global_stats, Bool,true)
    .ATTR(mode, Int,1)
    .OP_END_FACTORY_REG(BNInference)

/**
* @brief Performs batch normalization .

* @par Inputs:
* @li x: A 4D or 5D Tensor of type float16 or float32 or bfloat16, with format NHWC or NCHW.
* @li mean: A Tensor of type float32 or float16 or bfloat16. Must be 1D if input "x"
* Specifies the mean used for inference.
* @li variance: A Tensor of type float32 or float16 or bfloat16 . Must be 1D if input "x"
* Specifies the variance used for inference.
* @li scale: An optional tensor of type float16 or float32 or bfloat16, no use.
* @li b: An optional tensor of type float16 or float32 or bfloat16, no use. \n

* @par Attributes:
* @li momentum: An optional float32 num, represents the mean and
* the variance's scale factor Defaults to "0.9".
* @li epsilon: An optional float32, specifying the small value
* added to variance to avoid dividing by zero. Defaults to "0.00001".
* @li use_global_stats: mean inference mode , only can be "True".
* @li mode: An optional attr, not use. \n

* @par Outputs:
* @li y: A 4D or 5D Tensor of type float16 or float32 or bfloat16 for the normalized "x". \n

* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use BNInference instead.
*/
REG_OP(BNInferenceD)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mean, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(variance, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(b, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(momentum, Float,0.9f)
    .ATTR(epsilon, Float,1e-5f)
    .ATTR(use_global_stats, Bool,true)
    .ATTR(mode, Int,1)
    .OP_END_FACTORY_REG(BNInferenceD)

}  // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_NN_BATCH_NORM_OPS_H_WARN_SHOWN
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_BATCH_NORM_OPS_H_
