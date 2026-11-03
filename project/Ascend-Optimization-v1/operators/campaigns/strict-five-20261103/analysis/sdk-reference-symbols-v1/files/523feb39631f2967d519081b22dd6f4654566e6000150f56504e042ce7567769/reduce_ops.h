/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file reduce_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_REDUCE_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_REDUCE_OPS_H_

#ifndef OPS_BUILT_IN_OP_PROTO_INC_REDUCE_OPS_H_WARN_SHOWN
#define OPS_BUILT_IN_OP_PROTO_INC_REDUCE_OPS_H_WARN_SHOWN
#warning "reduce_ops.h is scheduled to be deprecated in December 2026, and will be replaced by the \
ops_proto_math.h, ops_proto_cv.h, ops_proto_nn.h, ops_proto_transformer.h, ops_proto_legacy.h. \
We apologize for any inconvenience caused and appreciate your timely migration to the new interface."

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Performs reduced batch normalization .

* @par Inputs:
* x: A 4D tensor of type float16 or float32 or bfloat16, with format NHWC or NCHW.
* Indicates the input tensor, that is, the original data to be normalized.

* @par Outputs:
* @li sum: A 1D tensor of type float32 for SUM reduced "x". It represents the sum of the input tensor "x" on the C axis.
* The shape of sum is consistent with the C axis of "x". Has the same format as "x".
* @li square_sum: A 1D tensor of type float32 for SUMSQ reduced "x". It represents the sum of squares of the input tensor "x" on the C axis.
* The shape of sum is consistent with the C axis of "x". Has the same format as "x". \n

* @attention Constraints:
* This operator is a BatchNorm fusion operator for updating the moving
* averages for training.
* This operator is used in conjunction with BNTrainingReduce.
*/
REG_OP(BNTrainingReduce)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(sum, TensorType({DT_FLOAT}))
    .OUTPUT(square_sum, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BNTrainingReduce)

/**
* @brief Performs reduced batch normalization .

* @par Inputs:
* x: A 5D tensor of type float16 or float32 or bfloat16, with format NDHWC or NCDHW.
* Represents the input tensor in batch normalization training.
* When the C axis is 0, other dimensions support empty tensors; when the C axis is not 0, other dimensions do not support empty tensors.

* @par Outputs:
* @li sum: A 1D tensor of type float32 for SUM reduced "x". It represents the sum of the input tensor "x" on the C axis.
* The shape of sum is consistent with the C axis of "x". Has the same format as "x".
* @li square_sum: A 1D tensor of type float32 for SUMSQ reduced "x". It represents the sum of squares of the input tensor "x" on the C axis.
* The shape of sum is consistent with the C axis of "x". Has the same format as "x". \n

* @attention Constraints:
* This operator is a BatchNorm fusion operator for updating the moving
* averages for training.
* This operator is used in conjunction with BN3DTrainingReduce.
*/
REG_OP(BN3DTrainingReduce)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(sum, TensorType({DT_FLOAT}))
    .OUTPUT(square_sum, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BN3DTrainingReduce)

/**
* @brief Performs the backpropagation of BatchNorm .

* @par Inputs:
* Seven inputs, including:
* @li grads: A 4D tensor of type float16 or float32 or bfloat16, for the gradient, with format NHWC or NCHW.
* The gradient of the loss function with respect to the output of the batch normalization layer.
* @li x: A 4D tensor of type float16 or float32 or bfloat16, with format NHWC or NCHW.
* It represents the data input to the batch normalization layer during the forward pass.
* Has the same type, format and shape as "grads".
* @li diff_scale: A 1D tensor of type float32, the shape is same as dim C of input grads.
* Indicates the gradient of the loss function to the scaling parameter "scale".
* Has the same format as "grads".
* @li diff_offset: A 1D tensor of type float32, the shape is same as dim C of input grads.
* Represents the gradient of the loss function to the offset parameter.
* Has the same format as "grads".
* @li scale: A 1D tensor of type float32, the shape is same as dim C of input grads.
* The scaling parameter in batch normalization, used to adjust the normalized output.
* Has the same format as "grads".
* @li batch_mean: A 1D tensor of type float32, the shape is same as dim C of input grads, for the mean of "x".
* Has the same format as "grads".
* @li batch_variance: A 1D tensor of type float32, the shape is same as dim C of input grads, for the variance of "x".
* Has the same format as "grads". \n

* @par Attributes:
* epsilon: An optional float32. Defaults to "0.0001".
* Represents a small positive number added to the variance of "x" to prevent division by zero. \n

* @par Outputs:
* y: A Tensor of type float16, float32 or bfloat16, with format NHWC or NCHW.
* It represents the gradient of the loss function with respect to the input data x.
* Has the same type, format and shape as "grads". \n

* @attention Constraints:
* The preceding layer of this operator must be BNTrainingUpdateGrad . \n

* @see BNTrainingUpdateGrad
*/
REG_OP(BNTrainingReduceGrad)
    .INPUT(grads, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(diff_scale, TensorType({DT_FLOAT}))
    .INPUT(diff_offset, TensorType({DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(batch_mean, TensorType({DT_FLOAT}))
    .INPUT(batch_variance, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(epsilon, Float, 0.0001)
    .OP_END_FACTORY_REG(BNTrainingReduceGrad)

/**
* @brief Performs the backpropagation of BatchNorm .

* @par Inputs:
* Seven inputs, including:
* @li grads: A 5Dtensor of type float16 or float32 or bfloat16, for the gradient, with format NDHWC or NCDHW.
* @li x: A 5D tensor of type float16 or float32 or bfloat16, with format NDHWC or NCDHW.
* @li diff_scale: A 1D tensor of type float32,
* for the mean of "x". shape must be C channel.
* @li diff_offset: A 1D tensor of type float32,
* for the variance of "x". shape must be C channel.
* @li scale: A 1D tensor of type float32.
* @li batch_mean: A 1D tensor of type float32,
* for the mean of "x". shape must be C channel.
* @li batch_variance: A 1D tensor of type float32,
* for the variance of "x" . shape must be C channel. \n

* @par Attributes:
* epsilon: An optional float32. Defaults to "0.0001". A small float number
* added to the variance of "x" . \n

* @par Outputs:
* y: A 5D Tensor of type float16 or float32 or bfloat16, with format NDHWC or NCDHW. \n

* @attention Constraints:
* The preceding layer of this operator must be BN3DTrainingReduceGrad . \n

* @see BN3DTrainingReduceGrad
*/
REG_OP(BN3DTrainingReduceGrad)
    .INPUT(grads, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(diff_scale, TensorType({DT_FLOAT}))
    .INPUT(diff_offset, TensorType({DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(batch_mean, TensorType({DT_FLOAT}))
    .INPUT(batch_variance, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(epsilon, Float, 0.0001)
    .OP_END_FACTORY_REG(BN3DTrainingReduceGrad)

/**
* @brief Performs reduced batch normalization .

* @par Inputs:
* Seven inputs, including:
* @li x: A 4D tensor of type float16 or float32 or bfloat16, with format NHWC or NCHW. Empty tensors are not supported.
* Input tensor, that is, the original data that needs to be normalized.
* @li sum: A 1D tensor of type float32, the shape is same as dim C of input "x", for the output of operator BNTrainingReduce.
* It represents the sum of the input tensor "x" on the C axis. Has the same format as "x".
* @li square_sum: A 1D tensor of type float32, the shape is same as dim C of input "x", for the output of operator BNTrainingReduce.
* It represents the sum of squares of the input tensor "x" on the C axis. Has the same format as "x".
* @li scale: A 1D tensor of type float32, the shape is same as dim C of input "x", for the scaling factor. Has the same format as "x".
* @li offset: A 1D tensor of type float32, the shape is same as dim C of input "x", for the scaling offset. Has the same format as "x".
* @li mean: A 1D tensor of type float32, the shape is same as dim C of input "x", for the updated mean. Has the same format as "x".
* @li variance: A 1D tensor of type float32, the shape is same as dim C of input "x", for the updated variance. Has the same format as "x". \n

* @par Attributes:
* @li epsilon: A required float32, specifying the small value added to variance
* to avoid dividing by zero.
* @li factor: A required float32, specifying the weight for updating the mean
* and variance . \n

* @par Outputs:
* Five outputs, including:
* @li y: A 4D tensor of type float16 or float32 or bfloat16, for normalized "x". Empty tensors are not supported.
* Has the same dype, format and shape as "x".
* @li mean: A 1D tensor of type float32, for the updated mean. shape must be C channel. Has the same format as "x".
* @li variance: A 1D tensor of type float32, for the updated variance. shape must be C channel. Has the same format as "x".
* @li batch_mean: A 1D tensor of type float32, for the mean of "x". shape must be C channel. Has the same format as "x".
* @li batch_variance: A 1D tensor of type float32, for the variance of "x" . shape must be C channel. Has the same format as "x". \n

* @attention Constraints:
* @li This operator is a BatchNorm fusion operator for updating the moving
* averages for training. This operator is used in conjunction with
* BNTrainingUpdate.
* @li For Atlas 200/300/500 Inference Product, the result accuracy fails to reach 1/1000 due to the
* square root instruction.
*/
REG_OP(BNTrainingUpdate)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(sum, TensorType({DT_FLOAT}))
    .INPUT(square_sum, TensorType({DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(offset, TensorType({DT_FLOAT}))
    .INPUT(mean, TensorType({DT_FLOAT}))
    .INPUT(variance, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(factor, Float)
    .REQUIRED_ATTR(epsilon, Float)
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(mean, TensorType({DT_FLOAT}))
    .OUTPUT(variance, TensorType({DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BNTrainingUpdate)

/**
* @brief Performs reduced batch normalization .

* @par Inputs:
* Seven inputs, including:
* @li x: A 5D tensor of type float16 or float32 or bfloat16, with format NDHWC or NCDHW.
* @li sum: A 1D tensor of type float32 for the output of operator BN3DTrainingUpdate. shape must be C channel.
* @li square_sum: A 1D tensor of type float32 for the output of operator BN3DTrainingUpdate. shape must be C channel.
* @li scale: A 1D tensor of type float32, for the scaling factor. shape must be C channel.
* @li offset: A 1D tensor of type float32, for the scaling offset. shape must be C channel.
* @li mean: A 1D tensor of type float32, for the updated mean. shape must be C channel.
* @li variance: A 1D tensor of type float32, for the updated variance . shape must be C channel. \n

* @par Attributes:
* @li epsilon: A required float32, specifying the small value added to variance
* to avoid dividing by zero.
* @li factor: A required float32, specifying the weight for updating the mean
* and variance . \n

* @par Outputs:
* Five outputs, including:
* @li y: A 5D tensor of type float16 or float32 or bfloat16, for normalized "x", with format NDHWC or NCDHW.
* @li mean: A 1D tensor of type float32, for the updated mean. shape must be C channel.
* @li variance: A 1D tensor of type float32, for the updated variance. shape must be C channel.
* @li batch_mean: A 1D tensor of type float32, for the mean of "x". shape must be C channel.
* @li batch_variance: A 1D tensor of type float32, for the variance of "x" . shape must be C channel. \n

* @attention Constraints:
* @li This operator is a BatchNorm fusion operator for updating the moving
  averages for training.
* This operator is used in conjunction with BN3DTrainingUpdate.
* @li For Atlas 200/300/500 Inference Product, the result accuracy fails to reach 1/1000 due to the square
* root instruction.
*/
REG_OP(BN3DTrainingUpdate)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(sum, TensorType({DT_FLOAT}))
    .INPUT(square_sum, TensorType({DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(offset, TensorType({DT_FLOAT}))
    .INPUT(mean, TensorType({DT_FLOAT}))
    .INPUT(variance, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(factor, Float)
    .REQUIRED_ATTR(epsilon, Float)
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(mean, TensorType({DT_FLOAT}))
    .OUTPUT(variance, TensorType({DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BN3DTrainingUpdate)

/**
* @brief Performs batch normalization for inference .

* @par Inputs:
* Five inputs, including:
* @li x: A 4D tensor of type float16 or float32 or bfloat16, with format NHWC or NCHW.
* @li scale: A 1D tensor of type float32, for the scaling factor, the shape is same as dim C of input x.
* @li offset: A 1D tensor of type float32, for the scaling offset, the shape is same as dim C of input x.
* @li mean: A 1D tensor of type float32, for the mean, the shape is same as dim C of input x.
* @li variance: A 1D tensor of type float32, for the variance, the shape is same as dim C of input x. \n

* @par Attributes:
* epsilon: An optional float32, specifying the small value added to variance to
* avoid dividing by zero. Defaults to "0.0001" . \n

* @par Outputs:
* y: A 4D tensor of type float16 or float32 or bfloat16 for the normalized "x", with format NHWC or NCHW. \n

* @attention Constraints:
* For Atlas 200/300/500 Inference Product, the result accuracy fails to reach 1/1000 due to the
* square root instruction.
*/
REG_OP(BNInfer)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(offset, TensorType({DT_FLOAT}))
    .INPUT(mean, TensorType({DT_FLOAT}))
    .INPUT(variance, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(epsilon, Float)
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(BNInfer)

/**
* @brief Performs reduced batch normalization. For some scenes which don't
* contain assign moving average .

* @par Inputs:
* Five inputs, including:
* @li x: A 4D tensor of type float16/float32/bfloat16, with format NHWC or NCHW. Empty tensors are not supported.
* Input tensor, that is, the original data that needs to be normalized.
* @li sum: A 1D tensor of type float32, the shape is same as dim C of input "x", for the output of operator BNTrainingReduce.
* It represents the sum of the input tensor "x" on the C axis. Has the same format as "x".
* @li square_sum: A 1D tensor of type float32, the shape is same as dim C of input "x", for the output of operator BNTrainingReduce.
* It represents the sum of squares of the input tensor "x" on the C axis. Has the same format as "x".
* @li scale: A 1D tensor of type float32, the shape is same as dim C of input "x", for the scaling factor. Has the same format as "x".
* @li offset: A 1D tensor of type float32, the shape is same as dim C of input "x", for the scaling offset. Has the same format as "x". \n

* @par Attributes:
* epsilon: A required float32, specifying the small value added to variance
* to avoid dividing by zero. \n

* @par Outputs:
* Three outputs, including:
* @li y: A 4D tensor of type float16 or float32 or bfloat16, for normalized "x". Empty tensors are not supported.
* Has the same dype, format and shape as "x".
* @li batch_mean: A 1D tensor of type float32, for the mean of "x". shape must be C channel. Has the same format as "x".
* @li batch_variance: A 1D tensor of type float32, for the variance of "x" . shape must be C channel. Has the same format as "x". \n

* @attention Constraints:
* @li This operator is used in conjunction with BNTrainingReduce.
* @li For Atlas 200/300/500 Inference Product, the result accuracy fails to reach 1/1000 due to
* the square root instruction.
*/
REG_OP(BNTrainingUpdateV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(sum, TensorType({DT_FLOAT}))
    .INPUT(square_sum, TensorType({DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(offset, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(epsilon, Float)
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BNTrainingUpdateV2)

/**
* @brief Performs reduced batch normalization v3. For some scenes which
* don't contain assign moving average .

* @par Inputs:
* Five inputs, including:
* @li x: A 4D tensor of type float16 or float32 or bfloat16, with format NHWC or NCHW. Empty tensors are not supported.
* Input tensor, that is, the original data that needs to be normalized.
* @li sum: A 1D tensor of type float32, the shape is same as dim C of input "x", for the output of operator BNTrainingReduce.
* It represents the sum of the input tensor "x" on the C axis. Has the same format as "x".
* @li square_sum: A 1D tensor of type float32, the shape is same as dim C of input "x", for the output of operator BNTrainingReduce.
* It represents the sum of squares of the input tensor "x" on the C axis. Has the same format as "x".
* @li scale: A 1D tensor of type float32, the shape is same as dim C of input "x", for the scaling factor. Has the same format as "x".
* @li offset: A 1D tensor of type float32, the shape is same as dim C of input "x", for the scaling offset. Has the same format as "x". \n

* @par Attributes:
* epsilon: A required float32, specifying the small value added to variance
* to avoid dividing by zero. \n

* @par Outputs:
* @li y: A 4D tensor of type float16 or float32 or bfloat16, for normalized "x". Empty tensors are not supported.
* Has the same dype, format and shape as "x".
* @li batch_mean: A 1D tensor of type float32, for the mean of "x". shape must be C channel. Has the same format as "x".
* @li batch_variance: A 1D tensor of type float32, for the variance of "x" . shape must be C channel. Has the same format as "x". 
* @li reserve_1: A 1D tensor of type float32, for the mean of batch "x".
* Has the same type, shape and format as input "sum".
* @li reserve_2: A 1D tensor of type float32, for the variance of batch "x".
* Has the same type, shape and format as input "sum". \n

* @attention Constraints:
* @li This operator is used in conjunction with BNTrainingReduce.
* @li For Atlas 200/300/500 Inference Product, the result accuracy fails to reach 1/1000 due to
* the square root instruction.
*/
REG_OP(BNTrainingUpdateV3)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(sum, TensorType({DT_FLOAT}))
    .INPUT(square_sum, TensorType({DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(offset, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(epsilon, Float)
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_1, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_2, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BNTrainingUpdateV3)

/**
* @brief Performs the backpropagation of BatchNorm .

* @par Inputs:
* Four inputs, including:
* @li grads: A 4D tensor of type float16 or float32 or bfloat16,
* for the gradient, with format NHWC or NCHW.
* Indicates the gradient of the loss function with respect to the output of the batch normalization layer.  
* @li x: A 4D tensor of type float16 or float32 or bfloat16, with format NHWC or NCHW.
* Indicates the data input to the batch normalization layer during the forward propagation process.  
* Has the same type, format and shape as "grads".
* @li batch_mean: A 1D tensor of type float32,
* for the mean of "x". Shape must be C channel.
* Has the same format as "grads".
* @li batch_variance: A 1D tensor of type float32,
* for the variance of "x" . Shape must be C channel.
* Has the same format as "grads". \n

* @par Attributes:
* epsilon: An optional float32. Defaults to "0.0001". 
* Represents a very small positive number that is added to the variance of "x" to prevent division by zero. \n

* @par Outputs:
* @li diff_scale: A 1D Tensor of type float32,
* for the offset of "scale". Shape must be C channel.
* Has the same format as "grads".

* @li diff_offset: A 1D Tensor of type float32,
* for the offset of "offset". Shape must be C channel.
* Has the same format as "grads". \n

*/
REG_OP(BNTrainingUpdateGrad)
    .INPUT(grads, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(batch_mean, TensorType({DT_FLOAT}))
    .INPUT(batch_variance, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001)
    .OUTPUT(diff_scale, TensorType({DT_FLOAT}))
    .OUTPUT(diff_offset, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BNTrainingUpdateGrad)

/**
* @brief Performs the backpropagation of BatchNorm .

* @par Inputs:
* Four inputs, including:
* @li grads: A 5D tensor of type float16 or float32 or bfloat16,
* for the gradient, with format NDHWC or NCDHW.
* @li x: A 5D tensor of type float16 or float32 or bfloat16, with format NDHWC or NCDHW,
* the shape is same as input grads.
* @li batch_mean: A 1D tensor of type float32,
* for the mean of "x", the shape is same as dim C of input grads.
* @li batch_variance: A 1D tensor of type float32,
* for the variance of "x", the shape is same as dim C of input grads. \n

* @par Attributes:
* epsilon: An optional float32. Defaults to "0.0001". A small float number
* added to the variance of "x" . \n

* @par Outputs:
* @li diff_scale: A 1D Tensor of type float32,
* for the offset of "scale", the shape is same as dim C of input grads.
* @li diff_offset: A 1D Tensor of type float32,
* for the offset of "offset", the shape is same as dim C of input grads. \n
*/
REG_OP(BN3DTrainingUpdateGrad)
    .INPUT(grads, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(batch_mean, TensorType({DT_FLOAT}))
    .INPUT(batch_variance, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001)
    .OUTPUT(diff_scale, TensorType({DT_FLOAT}))
    .OUTPUT(diff_offset, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BN3DTrainingUpdateGrad)

/**
* @brief Performs the backpropagation of BatchNorm for inference .

* @par Inputs:
* Three inputs, including:
* @li grads: A tensor of type float16 or float32 or bfloat16. Indicates the gradient of the BathNorm output parameter "y".
* Shape support 4D and 5D. Format support NHWC, NCHW or NC1HWC0.
* @li scale: A 1D tensor of type float32. Shape must be C channel.
* Specifies the scaling factor. Has the same format as "grads".
* @li batch_variance: A 1D tensor of type float32. Calculated variance, that is, the value of BatchNorm output parameter "batch_variance".
* Shape must be C channel. Has the same format as "grads". \n

* @par Attributes:
* epsilon: An optional float32. Defaults to "0.0001". A small float number
* added to the variance of "batch_variance". \n

* @par Outputs:
* x_backprop: A tensor of type float16 or float32 or bfloat16.
* Indicates the gradient of BathNorm input data "x".
* Has the same type, shape and format as "grads". \n

* @attention Constraints:
* The preceding layer of this operator must be operator BatchNorm.
*/
REG_OP(BNInferGrad)
    .INPUT(grads, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(batch_variance, TensorType({DT_FLOAT}))
    .OUTPUT(x_backprop, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(epsilon, Float, 0.0001)
    .OP_END_FACTORY_REG(BNInferGrad)

/**
* @brief Computes the sum of elements across dimensions of a tensor.

* @par Inputs:
* Two inputs, including:
* @li x: A tensor. Must be one of the following types:
* complex128, complex64, double, float32, float16, int16, int32, int64,
* int8, qint32, qint8, quint8, uint16, uint32, uint64, uint8, bfloat16,
* complex32.
* @li axes: A 1D list or tuple of IndexNumberType(int32 or int64).
* Specifies the dimensions to reduce.

* @par Attributes:
* keep_dims: An optional bool. If "true", retains reduced dimensions with
* length 1. Defaults to "false".
* noop_with_empty_axes: An optional bool. Defaults to "true" .
* - If true, when axes = [], not reduce.
* - If false, when axes = [], reduce all.
* This attribute is valid only for Ascend950 AI Processors and later products.

* @par Outputs:
* y: The reduced tensor. Has the same type and format as input "x".

* @attention Constraints:
* @li The value range of "axes" is [-dims, dims - 1]. "dims"
  indicates the dimension length of "x".
* @li When converting ONNX to OM, if the axes of the ReduceSum operator is empty,
   it is recommended to use the sum function with dim explicitly set to all axes
   (e.g., dim=[0, 1, 2]) to prevent shape inference errors.

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Sum.
*/
REG_OP(ReduceSum)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .ATTR(noop_with_empty_axes, Bool, true)
    .OP_END_FACTORY_REG(ReduceSum)

/**
* @brief Computes the sum of elements across dimensions of a tensor .

* @par Inputs:
* One input:
* x: A Tensor. Up to 8D. Must be one of the following types: float16, float32, bfloat16. \n

* @par Attributes:
* @li axes: A required 1D list or tuple of int32 or int64.
* Specifies the dimensions to reduce.
* @li keep_dims: An optional bool. If "true", retains reduced dimensions
* with length 1. Defaults to "false" . \n

* @par Outputs:
* y: The reduced tensor. Has the same type and format as input "x" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Sum.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ReduceSum instead.
*/
REG_OP(ReduceSumD)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(axes, ListInt)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceSumD)

/**
*@brief Calculate the total mean based on the mean of each device . \n

*@par Inputs:
* Three inputs, including:
*@li x: A Tensor. Must be one of the following types: float16, float32 bfloat16 .
*@li count: A Tensor. Must be one of the following types: float16, float32 bfloat16 .
*@li count_sum: A Tensor. Must be one of the following types: float16, float32 bfloat16 . \n

*@par Attributes:
*@li axes: A required 1D list or tuple of int32 or int64. Specifies the dimensions to reduce.
*@li keepdims: An optional bool. If "true", retains reduced dimensions with length 1. Defaults to "false" . \n

*@par Outputs:
*y: The reduced tensor. Has the same type and format as input "x" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Sum.
*/
REG_OP(ReduceMeanWithCount)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(count, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(count_sum, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(axes, ListInt)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceMeanWithCount)

/**
*@brief Calculates the "logical sum" of elements of a tensor in a dimension .

*@par Inputs:
*One input:
*x: The boolean tensor to reduce . \n

*@par Attributes:
*@li keep_dims: A bool, default false.
*If true, retains reduced dimensions with length 1.
*@li axis: The dimensions to reduce. If None, reduces all dimensions.
*Must be in the range [- rank (input_sensor), rank (input_sensor)) . \n

*@par Outputs:
*y: The reduced tensor . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ReduceAll.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ReduceAll instead.
*/
REG_OP(ReduceAllD)
    .INPUT(x, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .REQUIRED_ATTR(axes, ListInt)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceAllD)

/**
*@brief Calculates the "logical sum" of elements of a tensor in a dimension .

*@par Inputs:
*Two inputs, including:
*@li x: The boolean tensor to reduce.
*@li axis: A mutable Tensor with int dtype, The dimensions to reduce.
*If None, reduces all dimensions.
*Must be in the range [- rank (input_sensor), rank (input_sensor)) .

*@par Attributes:
*keep_dims: A bool, default false.
*If true, retains reduced dimensions with length 1 .

*@par Outputs:
*y: The reduced tensor .

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ReduceAll.
*/
REG_OP(ReduceAll)
    .INPUT(x, TensorType({DT_BOOL}))
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_BOOL}))
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceAll)

/**
*@brief  Reduce a tensor on a certain axis based on product.

*@par Inputs:
*Two inputs, including:
*@li x: A Tensor. Must be the type of NumberType.(NumberType
*includes: complex128, complex64, double, float32, float16, int16,
*int32, int64,int8, qint32, qint8, quint8, uint16, uint32, uint64,
*uint8, bfloat16, complex32).Supported format list ["ND"].
*@li axes: A Tensor. Must be the type of IndexNumberType(
*includes: int32, int64). The dimensions to reduce.Supported format list ["ND"]. \n

*@par Attributes:
*keep_dims: A bool. If true, retains reduced dimensions with length 1.
*Optional and defaults to "False" . \n
* noop_with_empty_axes: An optional bool. Defaults to "true" .
* - If true, when axes = [], not reduce.
* - If false, when axes = [], reduce all.
* This attribute is valid only for Ascend950 AI Processors and later products.

*@par Outputs:
*y: A Tensor. Has the same type and format as input "x" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ReduceProd.
*/
REG_OP(ReduceProd)
    .INPUT(x,TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y,TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .ATTR(noop_with_empty_axes, Bool, true)
    .OP_END_FACTORY_REG(ReduceProd)

/**
* @brief Computes the product of elements across dimensions of a tensor .

* @par Inputs:
* One input:
* x: A Tensor. Must be one of the following types: float16, float, int8, uint8,
* bfloat16, int32. \n

* @par Attributes:
* @li axes: A required int8, int16, int32, or int64. Specifies the dimensions
* to reduce.
* @li keep_dims: An optional bool. If "True", retains reduced dimensions
* with length 1. Defaults to "False" . \n

* @par Outputs:
* y: A Tensor. Has the same type and format as input "x" . \n

* @attention Constraints:
* "keep_dims" is in the range [-rank(input_tensor), rank(input_tensor)] . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ReduceProd.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ReduceProd instead.
*/
REG_OP(ReduceProdD)
    .INPUT(x, TensorType({DT_FLOAT, DT_UINT8, DT_INT8, DT_INT32, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_UINT8, DT_INT8, DT_INT32, DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(axes, ListInt)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceProdD)

/**
* @brief Reduces "x" along the dimensions according to "axis".

* @par Inputs:
* Two inputs, including:
* @li x: A tensor. Must be one of the following types:
*complex128, complex64, double, float32, float16, int64, int32, int16, int8,
*uint64, uint32, uint16, uint8, bfloat16. The data format supports ND.
* @li axes: The dimensions to reduce. Must be one of the following types:
* int, list, tuple, NoneType. Data type must be int32 or int64.
* If None (the default), reduces all dimensions.
* Must be in the range [-rank(x), rank(x)).

* @par Attributes:
* @li keep_dims: An optional bool. Defaults to false.
* If true, retains reduced dimensions with length 1.
* If false, the rank of the tensor is reduced by 1 for each entry in axis.
* @li noop_with_empty_axes: An optional bool. Defaults to true.
* If true, when axes = [], not reduce.
* If false, when axes = [], reduce all.
* @par Outputs:
* y: A tensor. Has the same type and format as "x".

* @attention Constraints:
* @li When converting ONNX to OM, if the axes of the ReduceMean operator is empty,
   and noop_with_empty_axes is true, it is recommended to use the mean function with dim explicitly
   set to all axes(e.g., dim=[0, 1, 2]) to prevent shape inference errors.

* @par Third-party framework compatibility:
* Compatible with the TensorFlow operator ReduceMean.
*/
REG_OP(ReduceMean)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .ATTR(noop_with_empty_axes, Bool, true)
    .OP_END_FACTORY_REG(ReduceMean)

/**
* @brief Reduces "x" along the dimensions according to "axis".

* @par Inputs:
* One input:
* x: A Tensor. Must be one of the following types: float16, float32 ,bfloat16. \n

* @par Attributes:
* @li axes: A required list of int.
* The dimensions to reduce. Must be one of the following types:
* int, list, tuple, NoneType.
* If None (the default), reduces all dimensions.
* Must be in the range [-rank(x), rank(x)).
* @li keep_dims: An optional bool or NoneType. Defaults to false.
* If true, retains reduced dimensions with length 1.
* If false, the rank of the tensor is reduced by 1 for each entry in axis.
* @li noop_with_empty_axes: An optional bool default False.
* If true, same as tf.
* If false, when x's shape is [], reduce all dims, for onnx.
* @par Outputs:
* y: A Tensor. Has the same type as "x" . \n

* @par Third-party framework compatibility:
* Compatible with the TensorFlow operator ReduceMean.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ReduceMean instead.
*/
REG_OP(ReduceMeanD)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(axes, ListInt)
    .ATTR(keep_dims, Bool, false)
    .ATTR(noop_with_empty_axes, Bool, false)
    .OP_END_FACTORY_REG(ReduceMeanD)

/**
* @brief Returns the maximum of elements across dimensions of a Tensor .

* @par Inputs:
* Two inputs, including:
* @li x: A multi-dimensional Tensor of Must be the type of NumberType. Supported format list ["ND"]
* @li axes: A Scalar of type in IndexNumberType(IndexNumberType includes the
  following types: int32, int64.), specifying the axes information
  of the index with the maximum value. Supported format list ["ND"] \n

* @par Attributes:
* keep_dims: A bool, specifying whether to keep dimensions for the output Tensor.
* Optional and defaults to "false". \n
* noop_with_empty_axes: An optional bool. Defaults to "true" .
* - If true, when axes = [], not reduce.
* - If false, when axes = [], reduce all.
* This attribute is valid only for Ascend950 AI Processors and later products.

* @par Outputs:
* y: A multi-dimensional Tensor, specifying the maximum value of the
  corresponding axis in the tensor.
  Has the same type as "x". (If "keep_dims" is set to "false",
  the output dimensions are reduced by "dimension" compared with that of "x".
  Otherwise, the output has one fewer dimension than "x").Supported format list ["ND"]

* @attention Constraints:
* @li The value range of "axes" is [-dims, dims - 1]. "dims"
  indicates the dimension length of "x".
* @li When converting ONNX to OM, if the axes of the ReduceMax operator is empty,
   it is recommended to use the amax function with dim explicitly set to all axes
   (e.g., dim=[0, 1, 2]) to prevent shape inference errors.

* @par Third-party framework compatibility
* Compatible with TensorFlow operator Max.
*/
REG_OP(ReduceMax)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .ATTR(noop_with_empty_axes, Bool, true)
    .OP_END_FACTORY_REG(ReduceMax)

/**
* @brief Returns the maximum of elements across dimensions of a Tensor .

* @par Inputs:
* x: A multi-dimensional Tensor of type float16, float32, bfloat16, int8,
  uint8 or int32 . \n

* @par Attributes:
* Two attributes, including:
* @li axes: A required listint, specifying the axes information of the index
  with the maximum value.
* @li keep_dims: A bool, specifying whether to keep dimensions for the output
  Tensor. Defaults to "false" . \n

* @par Outputs:
* y: A multi-dimensional Tensor, specifying the maximum value of the
  corresponding axis in the tensor. Has the same type as "x".
  (If "keep_dims" is set to "false", the output dimensions are reduced
  by "dimension" compared with that of "x". Otherwise, the output has
  one fewer dimension than "x".)

* @attention Constraints:
* The value range of "axis" is [-dims, dims - 1]. "dims" indicates the
  dimension length of "x" . \n

* @par Third-party framework compatibility
* Compatible with TensorFlow operator Max.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ReduceMax instead.
*/
REG_OP(ReduceMaxD)
    .INPUT(x, TensorType({DT_FLOAT, DT_UINT8, DT_INT8,
                          DT_FLOAT16, DT_INT32, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_UINT8, DT_INT8,
                           DT_FLOAT16, DT_INT32, DT_BF16}))
    .REQUIRED_ATTR(axes, ListInt)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceMaxD)

/**
* @brief Computes the minimum of elements across dimensions of a tensor .

* @par Inputs:
* @li x: A tensor. Must be the type of NumberType.
* @li axes: A tensor of type of IndexNumberType.(IndexNumberType
* includes: int32, int64.) Specifies the dimensions to reduce.
* Defaults to "None".

* @par Attributes:
* keep_dims: An optional bool. If "True", reduced dimensions will be retained.
* Defaults to "False".
* noop_with_empty_axes: An optional bool. Defaults to "true" .
* - If true, when axes = [], not reduce.
* - If false, when axes = [], reduce all.
* This attribute is valid only for Ascend950 AI Processors and later products.

* @par Outputs:
* y: A tensor. Must be the type of NumberType.

* @attention Constraints:
* @li If "axes = None", all dimensions will be reduced. "axes" must be in the
  range [-rank(input_shape), rank(input_shape)).
* @li When converting ONNX to OM, if the axes of the ReduceMin operator is empty,
   it is recommended to use the amin function with dim explicitly set to all axes
   (e.g., dim=[0, 1, 2]) to prevent shape inference errors.

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator reduce_min.
*/
REG_OP(ReduceMin)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .ATTR(noop_with_empty_axes, Bool, true)
    .OP_END_FACTORY_REG(ReduceMin)

/**
* @brief Computes the minimum of elements across dimensions of a tensor .

* @par Inputs:
* x: A tensor. Must be one of the following types: float16, float32,
* int8, uint8 ,bfloat16, int32. \n

* @par Attributes:
* @li axes: A required list int. Specifies the dimensions to reduce.
* @li keep_dims: An optional bool or NoneType value. If "True", reduced
* dimensions will be retained. Defaults to "None" (equivalent to "False").

* @par Outputs:
* y: A tensor. Must be one of the following types: float16,
* float32, int8, uint8, bfloat16, int32 . \n

* @attention Constraints:
* If "axes = None", all dimensions will be reduced. "axes" must be
* in the range [-rank(input_shape), rank(input_shape)) . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator reduce_min.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ReduceMin instead.
*/
REG_OP(ReduceMinD)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_UINT8, DT_INT32, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_UINT8, DT_INT32, DT_BF16}))
    .REQUIRED_ATTR(axes, ListInt)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceMinD)
/**
*@brief Computes the "logical or" of elements across dimensions of a tensor.
* Reduces "x" along the dimensions given in "axes".
* Unless "keep_dims" is true, the rank of the tensor is reduced by 1 for each
* entry in "axes". If "keep_dims" is true, the reduced dimensions
* are retained with length 1.
*
* If "axes" is None, all dimensions are reduced, and a
* tensor with a single element is returned.
*
*@attention Constraints:
* Only support bool
*
*@par Inputs:
*@li x : The boolean tensor to reduce.
*@li axes: The int tensor, The dimensions to reduce.
*          If "None" (default), reduces all dimensions.
*          Must be in the range "[-rank(x), rank(x))".
*@par Attributes:
* keep_dims: bool, default false.
*If true, retains reduced dimensions with length 1.
*
*@par Outputs:
* y: The reduced tensor
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator reduce_any.
*
*/
REG_OP(ReduceAny)
    .INPUT(x, TensorType({DT_BOOL}))
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_BOOL}))
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceAny)
/**
*@brief Computes the "logical or" of elements across dimensions of a tensor.
* Reduces "x" along the dimensions given in "axes".
* Unless "keep_dims" is true, the rank of the tensor is reduced by 1 for each
* entry in "axes". If "keep_dims" is true, the reduced dimensions
* are retained with length 1.
*
* If "axis" is None, all dimensions are reduced, and a
* tensor with a single element is returned.
*
*@attention Constraints:
*  Only support bool
*
*@par Inputs:
* x: The boolean tensor to reduce.
*
*@par Attributes:
*@li axes: The dimensions to reduce. Must be in the range "[-rank(x), rank(x))".
*@li keep_dims: An optional bool. If true, retains reduced dimensions with length 1. Defaults to false.
*
*@par Outputs:
* y: The reduced tensor. Must have the same type as "x".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator reduce_any.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ReduceAny instead.
*/
REG_OP(ReduceAnyD)
    .INPUT(x, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .REQUIRED_ATTR(axes, ListInt)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceAnyD)

/**
*@brief Compute reduction on dimensions specified by "axis".
*Four reduction operations are provided:
*SUM     Computes the sum of elements across specified dimensions of a tensor.
*ASUM    Computes the sum of absolute values of elements across specified
*dimensions of a tensor.
*SUMSQ   Computes the sum of squares of elements across specified
*dimensions of a tensor.
*SUMSQ   Computes the mean values of elements across specified
*dimensions of a tensor .

*@par Inputs:
*x: A Tensor of type float16 or float32. Support 1D ~ 8D, Support format:["ND", "NC1HWC0"].

*@par Attributes:
*@li operation: An optional int32 from 1(SUM), 2(ASUM), 3(SUMSQ), and 4(MEAN),
*specifying the reduction algorithm. Defaults to "1".
*@li axis: An optional int32, specifying the first axis to reduce.
*Defaults to "0".
*The value range is [-N, N-1], where N is the input tensor rank.
*@li coeff: An optional float32, specifying the scale coefficient.
*Defaults to "1.0" . \n

*@par Outputs:
*y: A Tensor. Has the same type as "x". Support 1D ~ 3D, Support format:["ND", "NC1HWC0"].

*@attention Constraints: The Reduction operator supports type float16
*only on the device chip.
*@par Third-party framework compatibility
* Compatible with the Caffe operator Reduction.
*/
REG_OP(Reduction)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(operation, Int, 1)
    .ATTR(axis, Int, 0)
    .ATTR(coeff, Float, 1.0)
    .OP_END_FACTORY_REG(Reduction);

/**
*@brief Computes the euclidean norm of elements across dimensions of a tensor . \n

*@par Inputs:
*@li x: A Tensor. Must be one of the following types: float16, float32, int32.
*@li axes: A Tensor of type int32 or int64. Specifies the dimensions to reduce. Defaults to "None" . \n

*@par Attributes:
*keep_dims: An optional bool. If "True", reduced dimensions will be retained. Defaults to "False" . \n

*@par Outputs:
*y: A Tensor. Must be one of the following types: float16, float32, int32 . \n

*@attention Constraints:
* If "axes = None", all dimensions will be reduced. "axes" must be in the range [-rank(input_shape), rank(input_shape)) . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator EuclideanNorm.
*/
REG_OP(EuclideanNorm)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(EuclideanNorm)

/**
*@brief Computes the euclidean norm of elements across dimensions of a tensor . \n

*@par Inputs:
*input_min: A Tensor. Must be one of the following types: float16, float32, int32 . \n

*@par Attributes:
*@li axes: An optional int32, list, tuple, or NoneType value. Specifies the dimensions to reduce. Defaults to "None".
*@li keep_dims: An optional bool or NoneType value. If "True", reduced dimensions will be retained. Defaults to "None" (equivalent to "False") . \n

*@par Outputs:
*output_min: A Tensor. Must be one of the following types: float16, float32, int32 . \n

*@attention Constraints:
* If "axes = None", all dimensions will be reduced. "axes" must be in the range [-rank(input_shape), rank(input_shape)) . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator EuclideanNorm.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use EuclideanNorm instead.
*/
REG_OP(EuclideanNormD)
    .INPUT(x, TensorType({DT_FLOAT, DT_INT32, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_FLOAT16}))
    .ATTR(axes, ListInt, {})
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(EuclideanNormD)



/**
*@brief Performs instance normalization for inference .

*@par Inputs:
* Five inputs, including:
*@li x: A Tensor of type float16 or float32.
*@li gamma: A optional Tensor of type float32, for the scaling gamma, with shape [N, C1, 1, 1, C0].
*@li beta: A optional Tensor of type float32, for the scaling beta, with the same shape of gamma.
*@li mean: A optional ensor of type float32, for the mean, with the same shape of gamma.
*@li variance: A optional Tensor of type float32, for the variance, with the same shape of gamma. \n

*@par Attributes:
*epsilon: An optional float32, specifying the small value added to variance to avoid dividing by zero.
Defaults to "0.00001" . \n

*@par Outputs:
*@li y: A Tensor of type float16 or float32 for the normalized "x".
*@li batch_mean: A Tensor of type float32 for the result mean.
*@li batch_variance: A Tensor of type float32 for the result variance . \n

*@attention Constraints:
*For Atlas 200/300/500 Inference Product, the result accuracy fails to reach 0.001 due to the square root instruction.
*/
REG_OP(INInferV2)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OPTIONAL_INPUT(gamma, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(beta, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(mean, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(variance, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.00001)
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(INInferV2)

/**
*@brief Performs reduce instance normalization.

*@par Inputs:
*x: A 4D tensor of type float16 or float32, format [NCHW, NHWC]\n

*@par Outputs:
*@li sum: A 4D tensor of type float32 for SUM reduced "x", format [NCHW, NHWC], and HW=1.
*@li square_sum: A 4D tensor of type float32 for SUMSQ reduced "x", format [NCHW, NHWC], and HW=1. \n

*@attention Constraints:
* This operator is a InstanceNorm fusion operator for updating the moving averages for training.
* This operator is used in conjunction with INTrainingUpdateV2.
*/
REG_OP(INTrainingReduceV2)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(sum, TensorType({DT_FLOAT}))
    .OUTPUT(square_sum, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(INTrainingReduceV2)


/**
*@brief Performs update instance normalization. \n

*@par Inputs:
* Seven inputs, including:
*@li x: A 4D tensor of type float16 or float32, format [NCHW, NHWC].
*@li sum: A 4D tensor of type float32 for the output of operator INTrainingReduceV2, format [NCHW, NHWC], and HW=1.
*@li square_sum: A 4D tensor of type float32 for the output of operator INTrainingReduceV2, format [NCHW, NHWC], and HW=1.
*@li gamma: A 4D optional tensor of type float32, for the scaling gamma, format [NCHW, NHWC], and HW=1.
*@li beta: A 4D optional tensor of type float32, for the scaling beta, format [NCHW, NHWC], and HW=1.
*@li mean: A 4D optional tensor of type float32, for the updated mean, format [NCHW, NHWC], and HW=1.
*@li variance: A 4D optional tensor of type float32, for the updated variance, format [NCHW, NHWC], and HW=1.\n

*@par Attributes:
*@li momentum: A optional float32, specifying the momentum to update mean and var. default to 0.1.
*@li epsilon: A optional float32, specifying the small value added to variance to avoid dividing by zero. default to 0.00001. \n

*@par Outputs:
* Three outputs
*@li y: A 4D tensor of type float16 or float32, for normalized "x", format [NCHW, NHWC].
*@li batch_mean: A 4D tensor of type float32, for the updated mean, format [NCHW, NHWC], and HW=1.
*@li batch_variance: A 4D tensor of type float32, for the updated variance, format [NCHW, NHWC], and HW=1. \n

*@attention Constraints:
* This operator is a InstanceNorm fusion operator for updating the moving averages for training.
* This operator is used in conjunction with INTrainingReduceV2.
*/
REG_OP(INTrainingUpdateV2)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(sum, TensorType({DT_FLOAT}))
    .INPUT(square_sum, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(gamma, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(beta, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(mean, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(variance, TensorType({DT_FLOAT}))
    .ATTR(momentum, Float, 0.1)
    .ATTR(epsilon, Float, 0.00001)
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(INTrainingUpdateV2)


/**
*@brief Performs the backpropagation of InstanceNorm. \n

*@par Inputs:
* Seven inputs, including:
*@li dy: A 4D tensor of type float16 or float32, format [NCHW, NHWC].
*@li x: A 4D tensor of type float16 or float32, format [NCHW, NHWC].
*@li variance: A 4D tensor of type float32, for the variance of "x", format [NCHW, NHWC] and HW=1.
*@li mean: A 4D tensor of type float32, for the mean of "x", format [NCHW, NHWC] and HW=1.
*@li res_gamma: A 4D tensor of type float32, format [NCHW, NHWC] and HW=1.
*@li res_beta: A 4D tensor of type float32, format [NCHW, NHWC] and HW=1.
*@li gamma: A 4D tensor of type float32, format [NCHW, NHWC] and HW=1. \n

*@par Outputs:
*pd_x: A 4D tensor of type float16 or float32, for the offset of "x", format [NCHW, NHWC]. \n

*@attention Constraints:
* The preceding layer of this operator must be INTrainingUpdateGrad. \n
*/
REG_OP(INTrainingReduceGrad)
    .INPUT(dy, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(variance, TensorType({DT_FLOAT}))
    .INPUT(mean, TensorType({DT_FLOAT}))
    .INPUT(res_gamma, TensorType({DT_FLOAT}))
    .INPUT(res_beta, TensorType({DT_FLOAT}))
    .INPUT(gamma, TensorType({DT_FLOAT}))
    .OUTPUT(pd_x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OP_END_FACTORY_REG(INTrainingReduceGrad)

/**
*@brief Performs the backpropagation of InstanceNorm. \n

*@par Inputs:
* Four inputs, including:
*@li dy: A 4D tensor of type float16 or float32, for the gradient, format [NCHW, NHWC].
*@li x: A 4Dtensor of type float16 or float32, format [NCHW, NHWC].
*@li variance: A 4D tensor of type float32, for the variance of "x", format [NCHW, NHWC] and HW=1.
*@li mean: A 4D tensor of type float32, for the mean of "x", format [NCHW, NHWC] and HW=1. \n

*@par Outputs:
*@li res_gamma: A 4D tensor of type float32, format [NCHW, NHWC] and HW=1.
*@li res_beta: A 4D tensor of type float32, format [NCHW, NHWC] and HW=1. \n

*/
REG_OP(INTrainingUpdateGrad)
    .INPUT(dy, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(variance, TensorType({DT_FLOAT}))
    .INPUT(mean, TensorType({DT_FLOAT}))
    .OUTPUT(res_gamma, TensorType({DT_FLOAT}))
    .OUTPUT(res_beta, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(INTrainingUpdateGrad)

/**
*@brief Performs the backpropagation of InstanceNorm. \n

*@par Inputs:
* Two inputs, including:
*@li res_gamma: A 4D tensor of type float32,  format [NCHW, NHWC].
*@li res_beta: A 4D tensor of type float32, format [NCHW, NHWC]. \n

*@par Outputs:
*@li pd_gamma: A 4D tensor of type float32, format [NCHW, NHWC].
*@li pd_beta: A 4D tensor of type float32, format [NCHW, NHWC]. \n

*/
REG_OP(INTrainingUpdateGradGammaBeta)
    .INPUT(res_gamma, TensorType({DT_FLOAT}))
    .INPUT(res_beta, TensorType({DT_FLOAT}))
    .OUTPUT(pd_gamma, TensorType({DT_FLOAT}))
    .OUTPUT(pd_beta, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(INTrainingUpdateGradGammaBeta)

/**
* @brief Backwards calculation of GroupedBiasAdd.
* @par Inputs:
* @li grad_y: A Tensor. Type is:BFloat16, Float16 or Float32. When group_idx is inputted, the shape only supports 2 dimensions.
*             When group_idx is not inputted, the shape only supports 3 dimensions, supports non-continuous tensors, and the data format supports ND.
* @li group_idx: An optional Tensor. Type is:Int32 or Int64.
*             Optional parameter, the end position of each group, shape only supports 1 dimension, supports non-continuous tensors, and the data format supports ND.
* @par Outputs:
* grad_bias: A Tensor. Type is:BFloat16, Float16 or Float32.
*             The data type must be the same as that of grad_y, and the shape only supports 2 dimensions, supports non-continuous tensors, and the data format supports ND.
* @par Attributes:
* @li group_idx_type: An optional Int, specifying the significance of group_idx, default to 0.
* @attention Constraints:
* group_idx: A maximum of 2048 groups are supported. \n
* When group_idx is inputted, it is required to ensure that the values of the tensor do not exceed the maxium value of INT32 and are non-negative. \n
* When group_idx is inputted and group_idx_type is 0, it is necessary to ensure that the tensor data is in ascending order, and the last numerical value is equal to the
* size of the 0th dimension of grad_y. \n
* When group_idx is inputted and group_idx_type is 1, it is necessary to ensure that the sum of the tensor values must equal the size of grad_y in the 0th dimension.
*/
REG_OP(GroupedBiasAddGrad)
    .INPUT(grad_y, "T")
    .OPTIONAL_INPUT(group_idx, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(grad_bias, "T")
    .ATTR(group_idx_type, Int, 0)
    .DATATYPE(T, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OP_END_FACTORY_REG(GroupedBiasAddGrad)

/**
*@brief Performs reduced group normalization.

*@par Inputs:
*x: A Tensor of type float16 or float32, with format NCHW NHWC . \n

*@par Outputs:
*@li sum: A Tensor of type float32 for SUM reduced "x". shape is [N, G, 1, 1, 1] for NCHW, [N, 1, 1, G, 1] for NHWC.
*@li square_sum: A Tensor of type float32 for SUMSQ reduced "x".shape is [N, G, 1, 1, 1] for NCHW, [N, 1, 1, G, 1] for NHWC.

*@par Attributes:
*num_groups: A optional Int, specifying the num of groups. required, same to GNTrainingUpdate, default to 2 . \n

*@attention Constraints:
* This operator is a GroupNorm fusion operator for updating the moving averages for training.
* This operator is used in conjunction with GNTrainingUpdate.
*/
REG_OP(GNTrainingReduce)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(sum, TensorType({DT_FLOAT}))
    .OUTPUT(square_sum, TensorType({DT_FLOAT}))
    .ATTR(num_groups, Int, 2)
    .OP_END_FACTORY_REG(GNTrainingReduce)


/**
*@brief Performs update group normalization .

*@par Inputs:
* Seven inputs, including: (NCHW NHWC supported)
*@li x: A Tensor of type float16 or float32.
*@li sum: A tensor of type float32,
shape is [N, G, 1, 1, 1] for NCHW, [N, 1, 1, G, 1] for NHWC
for the output of operator GNTrainingReduce.
*@li square_sum: A tensor of type float32,
shape is [N, G, 1, 1, 1] for NCHW, [N, 1, 1, G, 1] for NHWC
for the output of operator GNTrainingReduce.
*@li scale: A optional tensor of type float32,
shape is [1, G, 1, 1, 1] for NCHW, [1, 1, 1, G, 1] for NHWC
is for the scaling gamma.
*@li offset: A optional tensor of type float32,
shape is [1, G, 1, 1, 1] for NCHW, [1, 1, 1, G, 1] for NHWC
for the scaling beta.
*@li mean: A optional tensor of type float32,
shape is [N, G, 1, 1, 1] for NCHW, [N, 1, 1, G, 1] for NHWC
for the updated mean.
*@li variance: A optional tensor of type float32,
shape is [N, G, 1, 1, 1] for NCHW, [N, 1, 1, G, 1] for NHWC
for the updated variance.

*@par Attributes:
*@li epsilon: A optional float32, specifying the small value added to variance to avoid dividing by zero, default to 0.0001.
*@li num_groups: a optional int, specifying the num of groups. required, same to GNTrainingReduce, default to 2. \n

*@par Outputs:
* Three outputs, including:
*@li y: A Tensor of type float16 or float32, for normalized "x".
*@li batch_mean: A Tensor of type float32, for the updated mean.
*@li batch_variance: A Tensor of type float32, for the updated variance . \n

*@attention Constraints:
*@li This operator is a InstanceNorm fusion operator for updating the moving averages for training.
* This operator is used in conjunction with GNTrainingUpdate.
*@li For Atlas 200/300/500 Inference Product, the result accuracy fails to reach 1/1000 due to the square root instruction.
*/
REG_OP(GNTrainingUpdate)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(sum, TensorType({DT_FLOAT}))
    .INPUT(square_sum, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(mean, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(variance, TensorType({DT_FLOAT}))
    .ATTR(num_groups, Int, 2)
    .ATTR(epsilon, Float, 0.0001)
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(GNTrainingUpdate)

/**
*@brief Joins a string Tensor across the given dimensions. \n

*@par Inputs:
include:
*@li input:A Tensor of type string. The text to be processed.
*@li reduction_indices:A Tensor of type int. The text to be processed.

*@par Attributes:
*@li keep_dims: An optional bool. Defaults to False. If True, retain reduced dimensions with length 1.
*@li separator: An optional string. Defaults to "". \n

*@par Outputs:
*output:A Tensor of type string.
*/
REG_OP(ReduceJoin)
    .INPUT(input, TensorType({DT_STRING}))
    .INPUT(reduction_indices, TensorType({DT_INT32}))
    .OUTPUT(output, TensorType({DT_STRING}))
    .ATTR(keep_dims, Bool, true)
    .ATTR(separator, String, "")
    .OP_END_FACTORY_REG(ReduceJoin)

/**
* @brief Calculates the standard deviation and average value of Tensors.

* @par Inputs:
* x: A Tensor. Must be one of the following types:
*     float16, float32. \n

* @par Attributes:
* Three Attributes, including:
* @li dim: An optional listint, Defaults to "[]". \n

* @li unbiased: An optional bool. Defaults to "True".
*     If "True", Use Bessel Correction.
*     If "False", Do not use Bessel Correction. \n

* @li keepdim: An optional bool. Defaults to "False".
*     If "True", Keep the original tensor dimension.
*     If "False", Do not keep the original tensor dimension. \n

* @par Outputs:
* Two Outputs, including:
* @li y1: A Tensor. Has the same type as "x".
* @li y2: A Tensor. Has the same type as "x". \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator ReduceStd.
*/
REG_OP(ReduceStd)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y1, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y2, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(dim, ListInt, {})
    .ATTR(unbiased, Bool, true)
    .ATTR(keepdim, Bool, false)
    .OP_END_FACTORY_REG(ReduceStd)

/**
* @brief Calculates the standard deviation of Tensors.

* @par Inputs:
* include:
* @li x: A tensor. Must be one of the following types: float16, float32, bfloat16.
      The format must be NCHW, NHWC, or ND. \n
* @li mean: A tensor. It's the mean of X. Has the same shape, format and type as "x". \n


* @par Attributes:
* Six Attributes, including:
* @li dim: The dimensions to reduce. An optional listint, Defaults to "None".
*     If None (the default), reduces all dimensions.
*     Must be in the range [-rank(x), rank(x)). \n
* @li unbiased: An optional bool. Defaults to "True".
*     If "True", Use Bessel Correction.
*     If "False", Do not use Bessel Correction. \n
* @li keepdim: An optional bool. Defaults to "False".
*     If "True", Keep the original tensor dimension.
*     If "False", Do not keep the original tensor dimension. \n
* @li invert: An optional bool, Defaults to "False".
*     If "True", the output is inverse of variance.
*     If "False", the output is variance.
* @li epsilon: An optional float, Defaults to 0.001.
*     Prevent division by 0.
* @li correction: An optional int. Defaults to 1.
*     If unbiased is "True", use Bessel Correction. \n

* @par Outputs:
* y: A tensor. It's the variance of X or reciprocal of vaiance of X. Has the same type and format as "x".

* @par Third-party framework compatibility
* Compatible with the Pytorch operator ReduceStdWithMean.
*/
REG_OP(ReduceStdWithMean)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(dim, ListInt, {})
    .ATTR(unbiased, Bool, true)
    .ATTR(keepdim, Bool, false)
    .ATTR(invert, Bool, false)
    .ATTR(epsilon, Float, 0.001)
    .ATTR(correction, Int, 1)
    .OP_END_FACTORY_REG(ReduceStdWithMean)

/**
*@brief Performs reduced batch normalization . \n

*@par Inputs:
*x: A tensor of type float16 or float32 . \n

*@par Outputs:
*@li mean: A Tensor of type float32 for SUM reduced "x".
*@li variance: A Tensor of type float32 for square sum reduced "x" . \n

*@par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(ReduceMeanVariance)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(mean, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(variance, TensorType({DT_FLOAT16,DT_FLOAT}))
    .ATTR(axes, ListInt, {})
    .ATTR(keep_dims, Bool, true)
    .OP_END_FACTORY_REG(ReduceMeanVariance)

/**
* @brief Calculates the standard deviation or the variance of Tensors with the average value.

* @par Inputs:
* Two inputs, including:
* @li x: A Tensor. Must be one of the following types: float16, float32, bfloat16. \n
* @li mean: A Tensor. It's the mean of X. Has the same shape and type as "x" \n

* @par Attributes:
* Five Attributes, including:
* @li dim: The dimensions to reduce. A required listint.
*     Must be in the range [-rank(x), rank(x)).
* @li if_std: An optional bool. Defaults to "False".
*     If "True", Calculate the standard deviation
*     If "False", Calculate the variance
* @li unbiased: An optional bool. Defaults to "True".
*     If "True", Use Bessel Correction.
*     If "False", Do not use Bessel Correction. \n
* @li keepdim: An optional bool. Defaults to "False".
*     If "True", Keep the original tensor dimension.
*     If "False", Do not keep the original tensor dimension. \n
* @li correction: An optional int. Defaults to 1.
*     If unbiased is "True", use Bessel Correction. \n

* @par Outputs:
* @li output_var: A Tensor. It's the standard deviation or the variance of X. Has the same type as "x".

* @par Third-party framework compatibility
* Compatible with the Pytorch operator Var_mean.
*/
REG_OP(ReduceStdV2Update)
    .INPUT(x, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(mean, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(output_var, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(dim, ListInt)
    .ATTR(if_std, Bool, false)
    .ATTR(unbiased, Bool, true)
    .ATTR(keepdim, Bool, false)
    .ATTR(correction, Int, 1)
    .OP_END_FACTORY_REG(ReduceStdV2Update)

/**
* @brief Calculates the variance and average value of tensors.

* @par Inputs:
* x: A tensor. Format supports ND. Must be one of the following types: float32, float16, bfloat16. \n

* @par Attributes:
* Four Attributes, including:
* @li dim: The dimensions to reduce. An optional listint, Defaults to "None".
*     If None (the default), reduces all dimensions.
*     Must be in the range [-rank(x), rank(x)).

* @li correction: An optional int. Used for Bessel's correction. Defaults to 1.

* @li keepdim: An optional bool. Defaults to "False".
*     If "True", Keep the original tensor dimension.
*     If "False", Do not keep the original tensor dimension.

* @li is_mean_out: An optional bool. Defaults to "True".
*     If "True", Output the mean.
*     If "False", Do not output the mean. \n

* @par Outputs:
* Two Outputs, including:
* @li var: A tensor, the variance of x. Has the same type and format as "x".
* @li mean: A tensor, the mean of x. Has the same type and format as "x". \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator var and var_mean.
*/
REG_OP(ReduceVar)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(var, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(dim, ListInt, {})
    .ATTR(correction, Int, 1)
    .ATTR(keepdim, Bool, false)
    .ATTR(is_mean_out, Bool, true)
    .OP_END_FACTORY_REG(ReduceVar)

/**
* @brief Calculates the standard deviation and average value of tensors.

* @par Inputs:
* x: A tensor. Format supports ND. Must be one of the following types: float32, float16, bfloat16. \n

* @par Attributes:
* Four Attributes, including:
* @li dim: The dimensions to reduce. An optional listint, Defaults to "None".
*     If None (the default), reduces all dimensions.
*     Must be in the range [-rank(x), rank(x)).

* @li correction: An optional int. Used for Bessel's correction. Defaults to 1.

* @li keepdim: An optional bool. Defaults to "False".
*     If "True", Keep the original tensor dimension.
*     If "False", Do not keep the original tensor dimension.

* @li is_mean_out: An optional bool. Defaults to "True".
*     If "True", Output the mean.
*     If "False", Do not output the mean. \n

* @par Outputs:
* Two Outputs, including:
* @li std: A tensor, the standard deviation of x. Has the same type and format as "x".
* @li mean: A tensor, the mean of x. Has the same type and format as "x". \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator std and std_mean.
*/
REG_OP(ReduceStdV2)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(std, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(dim, ListInt, {})
    .ATTR(correction, Int, 1)
    .ATTR(keepdim, Bool, false)
    .ATTR(is_mean_out, Bool, true)
    .OP_END_FACTORY_REG(ReduceStdV2)

/**
* @brief Computes the log and sum and exp of elements across dimensions of a tensor.
* Reduces "x" along the dimensions given in "axes".
* Unless "keep_dims" is true, the rank of the tensor is reduced by 1 for each
* entry in "axes". If "keep_dims" is true, the reduced dimensions
* are retained with length 1.
*
* @par Inputs:
* Two inputs, including:
* @li x: A Tensor. Must be one of the following types: float32, float16, bfloat16.
* @li axes: A 1D list or tuple of int32 or int64. Specifies the dimensions to reduce. \n
*
* @par Attributes:
* keep_dims: An optional bool. If "true", retains reduced dimensions with length 1. Defaults to "false" . \n
*
* @par Outputs:
* y: The reduced tensor. Has the same type and format as input "x" . \n
*
* @par Third-party framework compatibility
* Compatible with the Onnx operator ReduceLogSumExp.
*/
REG_OP(ReduceLogSumExp)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(axes, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceLogSumExp)

/**
* @brief Computes the log and sum of elements across dimensions of a tensor.
* Reduces "x" along the dimensions given in "axes".
* Unless "keep_dims" is true, the rank of the tensor is reduced by 1 for each
* entry in "axes". If "keep_dims" is true, the reduced dimensions
* are retained with length 1.
*
* @par Inputs:
* Two inputs, including:
* @li x: A Tensor. Must be one of the following types: float32, float16.
* @li axes: A 1D list or tuple of int32 or int64. Specifies the dimensions to reduce . \n
*
* @par Attributes:
* keep_dims: An optional bool. If "true", retains reduced dimensions with length 1. Defaults to "false" . \n
*
* @par Outputs:
* y: The reduced tensor. Has the same type and format as input "x" . \n
*
* @par Third-party framework compatibility
* Compatible with the Onnx operator ReduceLogSum.
*/
REG_OP(ReduceLogSum)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceLogSum)

/**
* @brief Computes the sum of elements across dimensions of a tensor,
* treating Not a Numbers(NaNs) as zero .

* @par Inputs:
* Two inputs, including:
* @li x: A Tensor. Must be one of the following types of "T1":
*     float32, float16, bfloat16
* @li axis: A 1D list or tuple of "T2": int32 or int64. Specifies the dimensions to reduce . \n

* @par Attributes:
* keepdims: An optional bool. If "true", retains reduced dimensions with length 1. Defaults to "false" . \n

* @par Outputs:
* y: The reduced tensor. Has the same type and format as input "x" . \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator ReduceNansum.
*/
REG_OP(ReduceNansum)
    .INPUT(x, "T1")
    .INPUT(axes, "T2")
    .OUTPUT(y, "T1")
    .ATTR(keep_dims, Bool, false)
    .DATATYPE(T1, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .DATATYPE(T2, TensorType::IndexNumberType())
    .OP_END_FACTORY_REG(ReduceNansum)
} //namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_REDUCE_OPS_H_WARN_SHOWN
#endif  // OPS_BUILT_IN_OP_PROTO_INC_REDUCE_OPS_H_
