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
 * \file nn_norm_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_NORM_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_NORM_OPS_H_

#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_NORM_OPS_H_WARN_SHOWN
#define OPS_BUILT_IN_OP_PROTO_INC_NN_NORM_OPS_H_WARN_SHOWN
#warning "nn_norm_ops.h is scheduled to be deprecated in December 2026, and will be replaced by the \
ops_proto_math.h, ops_proto_cv.h, ops_proto_nn.h, ops_proto_transformer.h, ops_proto_legacy.h. \
We apologize for any inconvenience caused and appreciate your timely migration to the new interface."

#include "graph/operator_reg.h"
#include "nn_norm.h"
namespace ge {

/**
*@brief Computes the gradient for log softmax activations.

*@par Inputs:
* @li grad: A ND tensor. Must be one of the following data types: float16, bfloat16, float32.
* @li x: A ND tensor. Has the same data type and shape as "grad". \n

*@par Attributes:
* axis: An optional list of ints. Multi-axis reduction is supported. Defaults to "{-1}".
* In Ascend 950 AI Processor, only single-axis reduction is supported. \n

*@par Outputs:
* y: A ND tensor. Has the same data type and shape as "grad". \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator LogSoftmaxGrad.
*/

REG_OP(LogSoftmaxGrad)
    .INPUT(grad, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(axis, ListInt, {-1})
    .OP_END_FACTORY_REG(LogSoftmaxGrad)

/**
*@brief Computes sparse softmax cross entropy cost and gradients to backpropagate.

*@par Inputs:
*Two inputs, including:
* @li features: A Tensor. Must be one of the following types: float16, float32, double, bfloat16.
*A "batch_size * num_classes" matrix.
* @li labels: A Tensor. Must be one of the following types: 'int32', 'int64'.
*batch_size vector with values in [0, num_classes).
*This is the label for the given minibatch entry. \n


*@par Outputs:
*@li loss: A Tensor for per example loss (a "batch_size" vector). Has the same type as "features".
*@li backprop: A Tensor for the backpropagated gradients (a batch_size * num_classes matrix).
Has the same type as "features" . \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator SparseSoftmaxCrossEntropyWithLogits.
*/
REG_OP(SparseSoftmaxCrossEntropyWithLogits)
    .INPUT(features, TensorType({DT_DOUBLE,DT_FLOAT16,DT_FLOAT,DT_BFLOAT16}))
    .INPUT(labels, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(loss, TensorType({DT_DOUBLE,DT_FLOAT16,DT_FLOAT,DT_BFLOAT16}))
    .OUTPUT(backprop, TensorType({DT_DOUBLE,DT_FLOAT16,DT_FLOAT,DT_BFLOAT16}))
    .OP_END_FACTORY_REG(SparseSoftmaxCrossEntropyWithLogits)

/**
*@brief Computes softmax cross entropy cost and gradients to backpropagate. Broadcasting is supported.

*@par Inputs:
* Two inputs, including:
* @li features: A Tensor. Unnormalized scores from model output. Must be one of the following types: float16, bfloat16, float32, double.
* A "batch_size * num_classes" matrix. The format must be ND or NHWC. Support 2D, 4D.
* When the shape of features is 4D, the data type supports bfloat16 in dynamic shape scenarios, and float32 in static shape scenarios.
* @li labels: A Tensor. of the same type and format as "features". The true labels of the samples. A "batch_size * num_classes" matrix.
* Has the type, format and shape as "features". \n

*@par Outputs:
* @li loss: A Tensor for per example loss (a "batch_size" vector). Has the same type and format as "features". Support 1D, 3D.
* If the shape of features and labels is not 4D, the shape of the loss is 1D.
* If the shape of features and labels is 4D, the shape of the loss is 3D.
* @li backprop: A Tensor for the backpropagated gradients (a batch_size * num_classes matrix). 
* Has the same type, format and shape as "features". \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator SoftmaxCrossEntropyWithLogits.
*/
REG_OP(SoftmaxCrossEntropyWithLogits)
    .INPUT(features, TensorType({DT_DOUBLE,DT_FLOAT16,DT_BF16,DT_FLOAT}))
    .INPUT(labels, TensorType({DT_DOUBLE,DT_FLOAT16,DT_BF16,DT_FLOAT}))
    .OUTPUT(loss, TensorType({DT_DOUBLE,DT_FLOAT16,DT_BF16,DT_FLOAT}))
    .OUTPUT(backprop, TensorType({DT_DOUBLE,DT_FLOAT16,DT_BF16,DT_FLOAT}))
    .OP_END_FACTORY_REG(SoftmaxCrossEntropyWithLogits)

/**
*@brief Computes gradients for a softmax operation.

*@par Inputs:
* Two inputs, including:
* @li softmax: A ND tensor. Output of the softmax operator. Must be one of the following
* data types: float16, bfloat16, float32.
* @li grad_softmax: A ND tensor. Has the same shape and data type as "softmax". \n

*@par Attributes:
* axes: An optional list of ints. Multi-axis reduction is supported. Defaults to "{-1}".
* In Ascend 950 AI Processor, only single-axis reduction is supported. \n

*@par Outputs:
*grad_x: A ND tensor. Has the same shape and data type as "softmax" . \n

*@par Third-party framework compatibility
* Compatible with TensorFlow operator SoftmaxGrad.
*/
REG_OP(SoftmaxGrad)
    .INPUT(softmax, TensorType({ DT_FLOAT16, DT_BF16, DT_FLOAT }))
    .INPUT(grad_softmax, TensorType({ DT_FLOAT16, DT_BF16, DT_FLOAT }))
    .OUTPUT(grad_x, TensorType({ DT_FLOAT16, DT_BF16, DT_FLOAT }))
    .ATTR(axes, ListInt, {-1})
    .OP_END_FACTORY_REG(SoftmaxGrad)

/**
* @brief Computes the sigmoid cross entropy loss of "predict" and "target" .

* @par Inputs:
* Three inputs, including:
* @li predict: A multi-dimensional Tensor of type float16 or float32 or bfloat16, specifying the predictive value.
* @li target: A multi-dimensional Tensor of type float16 or float32 or bfloat16, specifying the target value .
* @li dout:A multi-dimensional Tensor of float16 or float32 or bfloat16,specifying the gradient transferred from the upper layer. \n

* @par Outputs:
* gradient: Sigmoid cross entropy between the predictive value and target value. Has the same dimensions as "predict" . \n

* @par Third-party framework compatibility
* Compatible with the scenario where "reduction" is set to "none"of PyTorch operator SigmoidCrossEntropyWithLogitsGrad.
*/
REG_OP(SigmoidCrossEntropyWithLogitsGrad)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(target, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(dout, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(gradient, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(SigmoidCrossEntropyWithLogitsGrad)

/**
* @brief Performs the backpropagation of SigmoidCrossEntropyWithLogits for training scenarios .

* @par Inputs:
* Two inputs, including:
* @li predict: A multi-dimensional Tensor of type float16 or float32 or bfloat16, specifying the predictive value.
* @li target: A multi-dimensional Tensor of type float16 or float32 or bfloat16, specifying the target value. \n

* @par Outputs:
* loss: Return loss. Has the same dimensions and type as "predict" . \n

* @par Third-party framework compatibility
* Compatible with the scenario where "reduction" is set to "none"of PyTorch operator SigmoidCrossEntropyWithLogits.
*/
REG_OP(SigmoidCrossEntropyWithLogits)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(target, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(loss, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(SigmoidCrossEntropyWithLogits)

/**
* @brief Computes the sigmoid cross entropy loss of "predict" and "target".  Broadcasting is supported.

* @par Inputs:
* four inputs, including:
* @li predict: A multi-dimensional Tensor of type float16 or float32 or bfloat16, specifying the predictive value.
* @li target: A multi-dimensional Tensor, has the same dtype as "predict", specifying the target value.
* @li weight: An optional multi-dimensional Tensor, has the same dtype as "predict", specifying the weight value.
* @li pos_weight: An optional multi-dimensional Tensor, has the same dtype as "predict", specifying the pos weight value. \n

* @par Attributes:
* reduction: A optional string attr from ["none", "mean", "sum"],
*  specifying the reduction type to be applied to the output. Defaults to "mean". \n

* @par Outputs:
* loss: An ND Tensor, Sigmoid cross entropy between the predictive value and target value. Has the same dimensions as "predict". \n

* @par Third-party framework compatibility
* Compatible with PyTorch operator BCEWithLogitsLoss.
*/
REG_OP(SigmoidCrossEntropyWithLogitsV2)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(target, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(pos_weight, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(loss, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(SigmoidCrossEntropyWithLogitsV2)

/**
* @brief Computes the sigmoid focal loss of "pred" and "target".

* @par Inputs:
* Three inputs, including:
* @li pred: A 2-dimensional Tensor of type float16 or float32, specifying the predicted value.
* @li target: A 1-dimensional Tensor of type int32, specifying the target value.
* @li weight: A 1-dimensional Tensor of type float16 or float32, specifying the weight value. \n

* @par Attributes:
* @li gamma: An optional float, specifying the exponent of the modulating factor (1 - pt)
* to balance easy/hard examples. Defaults to 2.0.
* @li alpha: An optional float, specifying the weighting factor in range (1, 0) to balance
* the importance of positive/negative examples or less than 0 for ignore. Defaults to 0.25.
* @li reduction: A optional character string from "none", "mean", and "sum", specifying the
* reduction type to be applied to the output. Defaults to "mean".  \n

* @par Outputs:
* y: Sigmoid focal loss between the predicted value and target value. Has the same dimensions as "pred". \n

* @par Third-party framework compatibility
* Compatible with mmcv operator SigmoidFocalLoss.
*/
REG_OP(SigmoidFocalLoss)
    .INPUT(pred, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(target, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .ATTR(gamma, Float, 2.0)
    .ATTR(alpha, Float, 0.25)
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(SigmoidFocalLoss)

/**
* @brief Computes the softmax focal loss of "pred" and "target".

* @par Inputs:
* Three inputs, including:
* @li pred: A 2-dimensional Tensor of type float16 or float32, specifying the predicted value.
* @li target: A 1-dimensional Tensor of type int32, specifying the target value.
* @li weight: A 1-dimensional Tensor, specifying the weight value on class_wise. \n

* @par Attributes:
* @li gamma: An optional float, specifying the exponent of the modulating factor (1 - pt)
* to balance easy/hard examples. Defaults to 2.0.
* @li alpha: An optional float, specifying the weighting factor in range (1, 0) to balance
* the importance of positive/negative examples or less than 0 for ignore. Defaults to 0.25.
* @li reduction: A optional character string from "none", "mean", and "sum", specifying the
* reduction type to be applied to the output. Defaults to "mean".  reduction only support
* "none" currently for matching mmcv.\n

* @par Outputs:
* y: Softmax focal loss between the predicted value and target value. Has the same dimensions as "pred". \n

* @par Third-party framework compatibility
* Compatible with mmcv operator SoftmaxFocalLoss.
*/
REG_OP(SoftmaxFocalLoss)
    .INPUT(pred, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(target, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .ATTR(gamma, Float, 2.0)
    .ATTR(alpha, Float, 0.25)
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(SoftmaxFocalLoss)

/**
* @brief Computes the regression box of the RPN. It is a FasterRCNN operator.

* @par Inputs:
* Two inputs, including:
* @li predict: A multi-dimensional Tensor of type float16 or float32 or bfloat16, specifying the predictive value.
* The maximum dimension is 8.
* @li label: A multi-dimensional Tensor of type float16 or float32 or bfloat16, specifying the target value.
* The maximum dimension is 8, predict and label can be broadcast.

* @par Attributes:
* sigma: Must be a floating point number. Defaults to "1.0".

* @par Outputs:
* loss: Indicates the loss between the predictive value and target value.
* Has the same dtype and dimensions as "predict".

* @attention Constraints:
* This operator does not perform the "reduce" operation on the loss value.
* Call other reduce operators to perform "reduce" operation on the loss if required.

* @par Third-party framework compatibility
* Compatible with the scenario where "reduction" is set to "none"of PyTorch operator SmoothL1Loss.
*/
REG_OP(SmoothL1Loss)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(label, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(loss, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(sigma, Float, 1.0)
    .OP_END_FACTORY_REG(SmoothL1Loss)

/**
* @brief Performs the backpropagation of SmoothL1Loss for training scenarios .

* @par Inputs:
* Three inputs, including:
* @li predict: A multi-dimensional Tensor of type float16 or float32 or bfloat16, specifying the predictive value.
* @li label: A multi-dimensional Tensor of float16 or float32 or bfloat16, specifying the target value.
* @li dout: A multi-dimensional Tensor of float16 or float32 or bfloat16,
    specifying the gradient transferred from the upper layer . \n

* @par Attributes:
* sigma: Must be a floating point number. Defaults to "1.0" . \n

* @par Outputs:
* gradient: Return gradient. Has the same dimensions and type as "predict" . \n

* @par Third-party framework compatibility
* Compatible with the scenario where "reduction" is set to "none"of PyTorch operator SmoothL1LossGrad.
*/
REG_OP(SmoothL1LossGrad)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(label, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(dout, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(gradient, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(sigma, Float, 1.0)
    .OP_END_FACTORY_REG(SmoothL1LossGrad)

/**
* @brief Creates a criterion that measures the Binary Cross Entropy between the target and the output. \n

* @par Inputs:
* Three inputs, including:
* @li x: A multi-dimensional tensor of type bfloat16, float16 or float32, specifying a predictive value. The value of "x" must range from 0 to 1.
* @li y: A multi-dimensional tensor of type bfloat16, float16 or float32, indicating a tag. The value of "y" must range from 0 to 1. Shape, dtype and format are the same as "x".
* @li weight: An optional multi-dimensional tensor, specifying the weight. If not null, shape, dtype and format are the same as "x" \n

* @par Attributes:
* reduction: A string specifying the reduction type to apply to the output, which must be one of: "none", "sum", or "mean". Defaults to "mean". \n

* @par Outputs:
* output: Output loss. Has the same dimension with the inputs. When "reduction" is set to "none", a tensor with the same size as "x" is output. Otherwise, a scalar is output. \n

* @par Third-party framework compatibility
* Compatible with PyTorch operator BCELoss.
*/
REG_OP(BinaryCrossEntropy)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(output, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(BinaryCrossEntropy)

/**
* @brief Performs the backpropagation of BinaryCrossEntropy for training scenarios .

* @par Inputs:
* Four inputs, including:
* @li x: A 1D or 2D Tensor of type bfloat16, float16 or float32, specifying a predictive value.
* @li y: A 1D or 2D Tensor of type bfloat16, float16 or float32, indicating a tag.
* @li grad_output: A 1D or 2D Tensor of type bfloat16, float16 or float32, specifying the backpropagation gradient.
* @li weight: An optional 1D or 2D Tensor of type bfloat16, float16 or float32, specifying the weight . \n

* @par Attributes:
* reduction: A character string from "none", "mean", and "sum", specifying the gradient output mode. Defaults to "mean" . \n

* @par Outputs:
* output: A 1D or 2D Tensor. When "reduction" is set to "none", a Tensor with the same size as "x" is output. Otherwise, a Scalar is output . \n

* @attention Constraints:
* @li The value of "x" must range from 0 to 1.
* @li The value of "y" must be "0" or "1" . \n

* @par Third-party framework compatibility
* Compatible with PyTorch operator BCELossGrad.
*/
REG_OP(BinaryCrossEntropyGrad)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(grad_output, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(output, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(BinaryCrossEntropyGrad)

/**
* @brief Performs the backpropagation of CrossEntropyLoss for training scenarios .

* @par Inputs:
* Six inputs, including:
* @li grad_loss: A 1D tensor or scalar of type float16 or float32 or bfloat16, specifying the backpropagation gradient. When reduction is "none", the shape is [N], when reduction is "sum" or "mean", it is scalar. N is batch size.
* @li log_prob: A 2D tensor of type float16 or float32 or bfloat16. Has the same dtype as grad_loss, specifying the logarithmized probabilities of the outputs. Shape only support [N, C], where N is batch size, C is class.
* @li target: A 1D tensor of type int32 or int64, specifying the target value. It represent the target squences. Shape is [N]. The range of values is [0, C).
* @li weight: An optional 1D tensor of type float32, specifying the weight value. Shape is [C].
* @li grad_zloss: An optional ND tensor of type float16 or float32 or bfloat16. Has the same dtype as grad_loss. Reserved.
* @li lse_gor_zloss: An optional ND tensor of type float16 or float32 or bfloat16. Has the same dtype as grad_loss. Reserved.

* @par Attributes:
* @li reduction: A character string from "none", "mean", and "sum", specifying the gradient output mode. Defaults to "mean" .
* @li ignore_index: An optional int. Specifies a target value that is ignored and does not contribute to the input gradient. Defaults to -100.
* @li label_smoothing: An optional float attr in [0.0, 1.0]. Specifies the amount of smoothing when computing the loss. Defaults to 0.0.
* @li lse_square_scale_for_zloss: An optional float attr, Default: 0.0. Reserved.

* @par Outputs:
* x_grad: A 2D tensor. Has the same dtype and shape as log_prob.
*/
REG_OP(CrossEntropyLossGrad)
    .INPUT(grad_loss, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(log_prob, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(target, TensorType({DT_INT64, DT_INT32}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(grad_zloss, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(lse_gor_zloss, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(x_grad, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .ATTR(reduction, String, "mean")
    .ATTR(ignore_index, Int, -100)
    .ATTR(label_smoothing, Float, 0.0)
    .ATTR(lse_square_scale_for_zloss, Float, 0.0)
    .OP_END_FACTORY_REG(CrossEntropyLossGrad)

/**
* @brief Applies the Softmax function to an n-dimensional input tensor
*  rescaling them. so that the elements of the n-dimensional output tensor lie
*  in the range [0,1] and sum to 1.

* @par Inputs:
* One input:
* x: A mutable input tensor, which can be floating point tensors with different precisions. Must be one of the following data types: float16, float32, bfloat16,
* double. Should be a variable tensor. The format must be ND. Shape support 1D ~ 8D. \n

* @par Attributes:
* @li axes: An optional list of int. Specifies on which dimensions of input x the Softmax operation is performed.
* Multi-axis reduction is supported. Defaults to "{-1}".
* In Ascend 950 AI Processor, only single-axis reduction is supported. \n
* @li half_to_float: An optional bool. 
* This parameter determines whether to convert the output data type to float32 when the input data type is float16.
* Defaults to "false".
* - If true and the input data type is float16, the output data type should be float32.
* - Otherwise, the output data type should be the same as the input data type. \n

* @par Outputs:
* y: A ND tensor. The output tensor represents the probability distribution of the input tensor after being processed by the Softmax function.
* Has the same dimensionality and shape as the "x" with values in the range [0, 1].
* Must be one of the following types: float16, float32, bfloat16, double. \n

* @par Third-party framework compatibility
*  Compatible with the TensorFlow operator Softmax.
*/
REG_OP(SoftmaxV2)
    .INPUT(x, TensorType({ DT_DOUBLE, DT_FLOAT16, DT_BF16, DT_FLOAT }))
    .OUTPUT(y, TensorType({ DT_DOUBLE, DT_FLOAT16, DT_BF16, DT_FLOAT }))
    .ATTR(axes, ListInt, {-1})
    .ATTR(half_to_float, Bool, false)
    .OP_END_FACTORY_REG(SoftmaxV2)

/**
*@brief Function softmax with dropoutDoMaskV3D

*@par Inputs:
*Two inputs, including:
* @li x: A mutable Tensor. The type only support float16.
* @li mask: A mutable Tensor. Must met all of the following rules:
*     shape of mask should be 1D.
*     dtype of mask should be uint8.
*     value of shape should met the following algorithm:
*     value = (size(x) + 128 - 1) // 128 * 128

*@par Attributes:
* @li keep_prob: A mutable Tensor. Must met all of the following rules:
*     shape of "keep_prob" should be (1,) or [1,].
*     Must be one of the following types: float16. \n
* @li axes: A list of int. The dimension softmax would be performed on. Defaults
*     to "[-1]" . \n

*@par Outputs:
*y1: A mutable Tensor. Has the same type as "x".
*y2: A mutable Tensor. Has the same type as "x". \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(SoftmaxV2WithDropOutDoMaskV3D)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(mask, TensorType({DT_UINT8}))
    .OUTPUT(y1, TensorType({DT_FLOAT16}))
    .OUTPUT(y2, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(keep_prob, Float)
    .ATTR(axes, ListInt, {-1})
    .OP_END_FACTORY_REG(SoftmaxV2WithDropOutDoMaskV3D)

/**
*@brief Computes log softmax activations .

*@par Inputs:
*One input:
* logits: A ND tensor. Must be one of the following data types: double, bfloat16, float16, float32 . \n

*@par Attributes:
* axes: An optional list of ints. Multi-axis reduction is supported. Defaults to "{-1}" .
* In Ascend 950 AI Processor, only single-axis reduction is supported. \n

*@par Outputs:
* logsoftmax: A ND tensor. Has the same data type as "logits" . \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator LogSoftmax.
*/
REG_OP(LogSoftmaxV2)
    .INPUT(logits, TensorType({DT_DOUBLE, DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(logsoftmax, TensorType({DT_DOUBLE, DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .ATTR(axes, ListInt, {-1})
    .OP_END_FACTORY_REG(LogSoftmaxV2)

/**
*@brief The logit function is a mathematical operation for converting probability to numeric variable. \n

*@par Inputs:
* x: A tensor of type float, float16 or bfloat16. Input data for the probability to logit transformation.
* Shape support 0D ~ 8D. The format must be ND.

*@par Attributes:
* eps: The epslion of "x", an optional attribute, the type is float. Defaults to -1.0. \n

*@par Outputs:
* y: A tensor with the same type, shape, format as "x". Output data from probability to logit transformation. \n
*/

REG_OP(Logit)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .ATTR(eps, Float, -1.0)
    .OP_END_FACTORY_REG(Logit)

/**
*@brief Backpropagation of Logit. \n

*@par Inputs:
*@li x: A input tensor of type float, float16 or bfloat16. Shape support 0D ~ 8D.
* Input data for the backpropagation of the probability to logit transformation
* The format must be ND.
*@li dy: Gradient of the positive output result. A Tensor with the same type, shape, format as "x".

*@par Attributes:
*eps: The epslion of "x", an optional attribute, the type is float. Defaults to -1.0. \n

*@par Outputs:
*dx: A output ensor with the same type, shape, format as "x". Probability to logit conversion backpropagates the output data. \n
*/

REG_OP(LogitGrad)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(dy, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(dx, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .ATTR(eps, Float, -1.0)
    .OP_END_FACTORY_REG(LogitGrad)

/**
*@brief Confuse mul, sum and sub .

*@par Inputs:
*Two inputs, including:
* @li grad: A ND tensor. Must be one of the following data types: bfloat16, float16, float32.
* @li x: A ND tensor. Has the same shape and data type as "grad". \n

*@par Outputs:
* y: A ND tensor.  Has the same shape and data type as "grad". \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(ConfusionSoftmaxGrad)
  .INPUT(grad, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
  .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
  .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
  .OP_END_FACTORY_REG(ConfusionSoftmaxGrad)

/**
*@brief Function softmax gradients ext.

*@par Inputs:
* @li grad: A tensor dtype of bfloat16, float16, float32. 
* Indicates the reverse input tensor.  
* The format must be FRACTAL_NZ unless specified. Support 2D ~ 6D.
* In Ascend 950 AI Processor, the format must be ND.

* @li x1: A tensor dtype of bfloat16, float16, float32. Indicates the inverse gradient.
* The format must be FRACTAL_NZ unless specified, Support 2D ~ 6D.
* In Ascend 950 AI Processor, the format must be ND.
* Has the same type, format and shape as input grad.
* @li x2: A scalar dtype of bfloat16, float16, float32. Indicates the inverse scaling factor.
* The format must be ND. Has the same type as input grad. \n

*@par Attributes:
*@li axes: A int attr. The axis for reduce, Defaults to 1.
*@li keep_dims: A bool attr. Whether to keep the dimension number. Defaults to true.
* Currently, this parameter can only be set to true. If true, retains reduced dimensions with length 1. \n

*@par Outputs:
* y: A tensor dtype of bfloat16, float16, float32. Indicates the reverse output.
* Has the same type, format and shape as input grad. \n

*@par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
*/
REG_OP(SoftmaxGradExt)
  .INPUT(grad, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
  .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
  .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
  .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
  .ATTR(axes, Int, 1)
  .ATTR(keep_dims, Bool, true)
  .OP_END_FACTORY_REG(SoftmaxGradExt)

/**
*@brief Normalizes the input .

*@par Inputs:
* One input:
*x: An NCHW tensor of type float16 or float32 . \n

*@par Attributes:
*@li normalize_variance: An optional bool specifying whether to normalize the 
* variance, either "true" (default) or "false"
* the value "false" indicates only to subtract the mean.
*@li across_channels: An optional bool specifying whether to perform 
* across-channel MVN, either "true" or "false" (default)
* The value "true" indicates "CHW" is treated as a vector.
*@li eps: An optional float32 epsilon for not dividing by zero. Defaults to 
* "1e-9" . \n

*@par Outputs:
*y: An NCHW tensor of type float16 or float32. \n

*@attention Constraints:
* The input tensor must have the NCHW format, whose shape length must be 4.
*@par Third-party framework compatibility
* Compatible with the Caffe operator MVN.
*/

REG_OP(MVN)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(normalize_variance, Bool, true)
    .ATTR(across_channels, Bool, false)
    .ATTR(eps, Float, 1e-9f)
    .OP_END_FACTORY_REG(MVN)

/**
*@brief Normalizes the input .

*@par Inputs:
* One input:
*x: An NCHW tensor of type float16 or float32 . \n

*@par Attributes:
*@li eps: An optional float32 epsilon for not dividing by zero. Defaults to "1e-9" . \n
*@li axes: A list of Intefers, along which axis to reduce. Defaults to "[0, 2, 3]" . \n

*@par Outputs:
*y: An NCHW tensor of type float16 or float32 . \n

*@attention Constraints:
* The input tensor must have the NCHW format, whose shape length must be 4.
*@par Third-party framework compatibility
* Compatible with the ONNX operator MeanVarianceNormalization.
*/

REG_OP(MVNV2)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16})) /* "First operand." */
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))  /* "Result, has same element type as inputs" */
    .ATTR(eps, Float, 1e-9f)
    .ATTR(axes, ListInt, {0, 2, 3})
    .OP_END_FACTORY_REG(MVNV2)

/**
*@brief Normalizes the input "x1" .

*@par Inputs:
* Two inputs, including:
*@li x1: A required NCHW or NHWC tensor of type float32, float16, or int8.
*@li x2: A required ND tensor of type float32, float16, or int8, specifying
* the scaling factor. If "channel_shared" is "true", "x2" is a [1]-dimensional
* vector. If "channel_shared" is "false", "x2" is a [C]-dimensional vector . \n

*@par Attributes:
*@li across_spatial: An optional bool, specifying the dimension of input "x1"
* to be summed. The value "true" (default) indicates dimensions C, H, W, and
* the value "false" indicates dimension C.
*@li channel_shared: An optional bool, specifying the dimension count of input
* "x2". The value "true" (default) indicates 1, and the value "false" indicates
* dimension C of "x1".
*@li eps: An optional float32, specifying the bias when "across_spatial" is
* "true". Defaults to "1e-10" . \n

*@par Outputs:
*y: A Tensor. Has the same type and format as "x1" . \n

*@par Third-party framework compatibility
* Compatible with the Caffe operator Normalize.
*/
REG_OP(Normalize)
     .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8}))
     .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8}))
     .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8}))
     .ATTR(across_spatial, Bool, true)
     .ATTR(channel_shared, Bool, true)
     .ATTR(eps, Float, 1e-10f)
     .OP_END_FACTORY_REG(Normalize);

/**
*@brief Layernorm operator interface implementation \n
*  calculating: x, gamma, beta \n
*  mean  = np.mean(x, reduce_axis, keepdims=True) \n
*  variance = np.mean(np.power((x - mean),2), reduce_axis, keepdims=True) \n
*  y = gamma*((x - mean) / np.sqrt(variance + epsilon)) + beta

*@par Inputs:
*Three inputs, including:
* @li x: A ND Tensor. Must be one of the following dtypes: float16, float32, bfloat16.
* The shape is [A1,...,Ai,R1,...,Rj].
* @li gamma: A ND Tensor. Must be one of the following dtypes: float16, float32, bfloat16.
* Has the same dtype and shape as beta. The shape is [R1,...,Rj].
* @li beta: A ND Tensor. Must be one of the following dtypes: float16, float32, bfloat16.
* Has the same dtype and shape as gamma. The shape is [R1,...,Rj]. \n

*@par Attributes:
* @li begin_norm_axis: An optional attribute, the dtype is int32. Defaults to 0.
* Indicates the index of the R1 axis in the shape of x.
* @li begin_params_axis: An optional attribute, the dtype is int32. Defaults to 0.
* In Ascend 950 AI Processor, begin_params_axis and begin_norm_axis refer to the same axis in the shape of x.
* @li epsilon: An optional attribute, the dtype is float32. Defaults to 1e-7 . \n

*@par Outputs:
*Three outputs, including:
* @li y: A ND Tensor. Must be one of the following dtypes: float16, float32, bfloat16.
* Has the same dtype, shape and format as x. 
* @li mean: A ND Tensor. Must be one of the following dtypes: float16, float32, bfloat16.
* Has the same shape as variance, which is [A1,...,Ai,1,...,1], where there are j 1's after Ai.
* @li variance: A ND Tensor. Must be one of the following dtypes: float16, float32, bfloat16.
* Has the same shape as mean, which is [A1,...,Ai,1,...,1], where there are j 1's after Ai.
*/
REG_OP(LayerNorm)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(beta, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(variance, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(begin_norm_axis, Int, 0)
    .ATTR(begin_params_axis, Int, 0)
    .ATTR(epsilon, Float, 0.0000001f)
    .OP_END_FACTORY_REG(LayerNorm)

/**
* @brief LayernormV3 operator interface implementation \n
*  calculating: x, gamma, beta \n
*  mean  = np.mean(x, reduce_axis, keepdims=True) \n
*  rstd = np.rsqrt(np.mean(np.power((x - mean),2), reduce_axis, keepdims=True) + epsilon)) \n
*  y = gamma*((x - mean) * rstd) + beta \n
*  Notes: the shape of gamma/beta is the slice shape of x(x_shape[begin_norm_axis])

*@par Inputs:
*Three inputs, including:
* @li x: A ND Tensor. Must be one of the following dtypes: float16, float32, bfloat16.
* The shape is [A1,...,Ai,R1,...,Rj].
* @li gamma: A ND Tensor. Must be one of the following dtypes: float16, float32, bfloat16.
* Has the same dtype and shape as beta. The shape is [R1,...,Rj].
* @li beta: A ND Tensor. Must be one of the following dtypes: float16, float32, bfloat16.
* Has the same dtype and shape as gamma. The shape is [R1,...,Rj]. \n

*@par Attributes:
* @li begin_norm_axis: An optional attribute, the dtype is int32. Defaults to 0.
* Indicates the index of the R1 axis in the shape of x.
* @li begin_params_axis: An optional attribute, the dtype is int32. Defaults to 0.
* In Ascend 950 AI Processor, begin_params_axis and begin_norm_axis refer to the same axis in the shape of x.
* @li epsilon: An optional attribute, the dtype is float32. Defaults to 1e-5 . \n

*@par Outputs:
*Three outputs, including:
* @li y: A ND Tensor. Must be one of the following dtypes: float16, float32, bfloat16.
* Has the same dtype, shape and format as x. 
* @li mean: A ND Tensor. Must be one of the following dtypes: float16, float32, bfloat16.
* Has the same shape as rstd, which is [A1,...,Ai,1,...,1], where there are j 1's after Ai.
* @li rstd: A ND Tensor. Must be one of the following dtypes: float16, float32, bfloat16.
* Has the same shape as mean, which is [A1,...,Ai,1,...,1], where there are j 1's after Ai.
*/
REG_OP(LayerNormV3)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(beta, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(rstd, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(begin_norm_axis, Int, 0)
    .ATTR(begin_params_axis, Int, 0)
    .ATTR(epsilon, Float, 0.00001f)
    .OP_END_FACTORY_REG(LayerNormV3)

/**
* @brief LayernormV4 operator interface implementation \n
*  calculating: x, gamma, beta \n
*  mean  = np.mean(x, reduce_axis, keepdims=True) \n
*  rstd = np.rsqrt(np.mean(np.power((x - mean),2), reduce_axis, keepdims=True) + epsilon)) \n
*  y = gamma*((x - mean) * rstd) + beta

*@par Inputs:
*Four inputs, including:
* @li x: A ND Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li normalized_shape: A ND Tensor. Must be one of the following types: int32, int64
* @li gamma: A ND Tensor. Must be one of the following types: float16, float32, bfloat16. Shape is normalized_shape.
* @li beta: A ND Tensor. Must be one of the following types: float16, float32, bfloat16. Shape is normalized_shape.\n

*@par Attributes:
* @li epsilon: An optional attribute, the type is float32. Defaults to 1e-5 . \n

*@par Outputs:
*Three outputs, including:
* @li y: A ND Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li mean: A ND Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li rstd: A ND Tensor. Must be one of the following types: float16, float32, bfloat16.
*/
REG_OP(LayerNormV4)
    .INPUT(x, "T1")
    .INPUT(normalized_shape, "T2")
    .OPTIONAL_INPUT(gamma, "T3")
    .OPTIONAL_INPUT(beta, "T4")
    .OUTPUT(y, "T5")
    .OUTPUT(mean, "T6")
    .OUTPUT(rstd, "T6")
    .ATTR(epsilon, Float, 0.00001f)
    .DATATYPE(T1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T2, TensorType({DT_INT32, DT_INT64}))
    .DATATYPE(T3, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T4, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T5, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T6, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(LayerNormV4)

/**
*@brief Layernorm operator interface implementation with given sum and square sum of input tensor \n
*  calculating: x, gamma, beta, sum, square_sum \n
*  mean  = sum / reduce_axis \n
*  variance = square_sum / reduce_aixs - mean * mean \n
*  variance = np.mean(np.power((x - mean),2), reduce_axis, keepdims=True) \n
*  y = gamma*((x - mean) / np.sqrt(variance + 0.001)) + beta

*@par Inputs:
*Five inputs, including:
* @li x: A Tensor. Must be one of the following types: float16.
* @li gamma: A Tensor. Must be one of the following types: float16.
* @li beta: A Tensor. Must be one of the following types: float16.
* @li sum. A Tensor. Must be one of the following types: float.
* @li square_sum. A Tensor. Must be one of the following types: float. \n

*@par Attributes:
* @li epsilon: A optional attribute, the type is float32. Defaults to 1e-5. \n

*@par Outputs:
*Three outputs, including:
* @li y: A Tensor. Must be one of the following types: float16.
*/
REG_OP(LayerNormUpdate)
    .INPUT(x1, TensorType({DT_FLOAT16}))
    .INPUT(beta, TensorType({DT_FLOAT16}))
    .INPUT(gamma, TensorType({DT_FLOAT16}))
    .INPUT(sum, TensorType({DT_FLOAT}))
    .INPUT(square_sum, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .ATTR(epsilon, Float, 0.00001f)
    .OP_END_FACTORY_REG(LayerNormUpdate)

/**
* @brief RmsNorm operator interface implementation. \n
*  calculating: x, gamma \n
*  rstd = np.rsqrt(np.mean(np.power(x,2), reduce_axis, keepdims=True) + epsilon)) \n
*  y = gamma * (x * rstd)

* @par Inputs
* Two inputs, including:
* @li x: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li gamma: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].

* @par Attributes
* epsilon: A optional attribute, the type is float. Defaults to 1e-6.

* @par Outputs
* Two outputs, including:
* @li y: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li rstd: A Tensor. Support dtype: [float32], support format: [ND].
*/
REG_OP(RmsNorm)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(rstd, TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .ATTR(epsilon, Float, 1e-6f)
    .OP_END_FACTORY_REG(RmsNorm)

/**
 * @brief The fusion operator of SpiltVD, RMSNorm, RotaryPositionEmbedding, Quant and Update KVCache.
 *
 * @par Inputs:
 * @li qkv: A tensor. The type support float16 and bfloat16. Its shape is [Bqkv*Sqkv,Nqkv*Dqkv]. Format: ND.
 * @li q_gamma: A tensor, used in RMS Norm. Its type is consistent with qkv. Its shape is [Dqkv,]. Format: ND.
 * @li k_gamma: A tensor, used in RMS Norm. Its type is consistent with qkv. Its shape is [Dqkv,]. Format: ND.
 * @li cos: A tensor, from position embedding. Its type is consistent with kv. Its shape is [Bqkv*Sqkv,Dqkv]. Format: ND.
 * @li sin: A tensor, from position embedding. Its type is consistent with qkv. [Bqkv*Sqkv,Dqkv]. 
 * The shape of sin should be consistent with that of cos. Format: ND.
 * @li index: A tensor. The type support int64. When the cache_mode is PA_NZ, its shape is [Bqkv*Sqkv] Format: ND.
 * @li q_out: A tensor. Its type is consistent with qkv. Its shape is [Bqkv*Sqkv, Nq*Dqkv]. Format: ND.
 * @li k_cache: A tensor. Its type is consistent with qkv or int8. When the type is int8, 
 * its shape is [BlockNum, Nk * Dqkv // 32, BlockSize, 32]. when the type consistent with qkv,
 * its shape is [BlockNum, Nk * Dqkv // 16, BlockSize, 16]. Format: ND.
 * @li v_cache: A tensor. Its type is consistent with qkv or int8. When the type is int8, 
 * its shape is [BlockNum, Nv * Dqkv // 32, BlockSize, 32]. when the type consistent with qkv,
 * its shape is [BlockNum, Nv * Dqkv // 16, BlockSize, 16]. Format: ND.
 * @li k_scale: A tensor. The type support float32. This input parameter is required when the data type of k_cache
 * is int8. Its shape can be [Nk,Dqkv]. Format: ND.
 * @li v_scale: A tensor. The type support float32. This input parameter is required when the data type of v_cache
 * is int8. Its shape can be [Nv,Dqkv]. Format: ND.
 * @li k_offset: A tensor. The type support float32. When the data type of k_cache is int8 and the corresponding
 * input exists with the quantization scenario being asymmetric quantization, this parameter input is required. Its
 * shape can be [Nk,Dqkv]. Format: ND.
 * @li v_offset: A tensor. The type support float32. When the data type of v_cache is int8 and the corresponding
 * input exists with the quantization scenario being asymmetric quantization, this parameter input is required. Its
 * shape can be [Nv,Dqkv]. Format: ND.
 *
 * @par Outputs:
 * @li q_out: A tensor. Its type is consistent with qkv. Its shape is [Bqkv*Sqkv, Nq*Dqkv]. Format: ND.
 * @li k_cache: A tensor. Its type is consistent with qkv or int8. When the type is int8, 
 * its shape is [BlockNum, Nk * Dqkv // 32, BlockSize, 32]. when the type consistent with qkv,
 * its shape is [BlockNum, Nk * Dqkv // 16, BlockSize, 16]. Format: ND.
 * @li v_cache: A tensor. Its type is consistent with qkv or int8. When the type is int8, 
 * its shape is [BlockNum, Nk * Dqkv // 32, BlockSize, 32]. when the type consistent with qkv,
 * its shape is [BlockNum, Nk * Dqkv // 16, BlockSize, 16]. Format: ND.
 * @li q_out_before_quant: A tensor. Its type is consistent with qkv. If is_output_qkv is true, this parameter output is required.
 * Its shape is [Bqkv*Sqkv, Nq*Dqkv]. Format: ND.
 * @li k_out_before_quant: A tensor. Its type is consistent with qkv. If is_output_qkv is true, this parameter output is required.
 * Its shape is [Bqkv*Sqkv, Nk*Dqkv]. Format: ND.
 * @li v_out_before_quant: A tensor. Its type is consistent with qkv. If is_output_qkv is true, this parameter output is required.
 * Its shape is [Bqkv*Sqkv, Nv*Dqkv]. Format: ND.
 *
 * @par Attributes:
 * @li qkv_size: A Listint. Provide the specific dimension sizes of the input of qkv matrices in the order of [Bqkv, Sqkv, Nqkv, Dqkv].
 * @li head_nums: A Listint. Specifying the N-dim,ensional sizes of the q, k, v components in the qkv input matrix, in the order [Nq, Nk, Nv].
 * @li epsilon: A float32. The epsilon value for RMSNorm. Default: 1e-6.
 * @li cache_mode: A string. The cache dtype. There is one types：PA_NZ. Default: PA_NZ.
 * @li is_output_qkv: A bool. Control whether k_cache_proto and v_cache_proto are output. If it is true, then k_out_before_quant and v_out_before_quant need to be
 * output. If it is false, then k_cache_proto and v_cache_proto do not need to be output. Default: false.
 *
 * @attention Constraints:
 * @li The shape format description in the parameter description:
 *     - Bqkv represents the batch size of the input qkv, and Skv represents the sequence length of the input qkv. The
 *     size is determined by the user's input scenario and has no explicit restrictions.
 *     - Nqkv is the head number of the input qkv. This operator is strongly related to the Qwen3-Moe 235b network structure.
 *     - Dqkv is the head dim of the input qkv. The data Dqkv required for rms norm calculation and the data Dqkv required
 *     for rope calculation are split from the Dqkv of the input qkv. In rope calculation, Drope=Dqkv.
 *     Therefore, Dqkv must comply with the rope rules. According to the rope rule, Dqkv is an even number.
 *     If cache_mode is an NZ scenario(cahce_mode is PA_NZ), Dqkv needs to be 32B aligned.
 *     - If cache_mode is a PA scenario, the BlockSize needs to be 32B aligned.
 *     - Regarding the above-mentioned 32B alignment situation, the alignment value is determined by the data type
 *     of the cache. Take BlockSize as an example. If the data type of the cache is int8, then BlockSize%32=0 is
 *     required. If the data type of the cache is flaot16, then BlockSize%16=0 is required.
 *     - BlockNum is the number of memory blocks written to the cache, and its size is determined by the user's input
 *     scenario without any definite limit.
 * @li The index-related constraints:
 *     When cache_mode is PA_NZ, the shape is [Bqkv*Sqkv], and the value range of the index must be
 *     [-1,BlockNum*BlockSize). The values cannot be repeated.
 */
REG_OP(QkvRmsNormRopeCache)
    .INPUT(qkv, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(q_gamma, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(k_gamma, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(cos, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(sin, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(index, TensorType({DT_INT64}))
    .INPUT(q_out, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(k_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .INPUT(v_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OPTIONAL_INPUT(k_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(v_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(k_offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(v_offset, TensorType({DT_FLOAT}))
    .OUTPUT(q_out, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(k_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OUTPUT(v_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OUTPUT(q_out_before_quant, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(k_out_before_quant, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(v_out_before_quant, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(qkv_size, ListInt)
    .REQUIRED_ATTR(head_nums, ListInt)
    .ATTR(epsilon, Float, 1e-6)
    .ATTR(cache_mode, String, "PA_NZ")
    .ATTR(is_output_qkv, Bool, false)
    .OP_END_FACTORY_REG(QkvRmsNormRopeCache)

/**
* @brief GemmaRmsNorm operator interface implementation. \n
*  calculating: x, gamma \n
*  rstd = np.rsqrt(np.mean(np.power(x, 2), reduce_axis, keepdims=True) + epsilon)) \n
*  y = (1 + gamma) * (x * rstd)

* @par Inputs
* Two inputs, including:
* @li x: A tensor of input. Represens the input data to be normalized. The format must be ND. Shape support 1D ~ 8D.
* Must be one of the following types: float16, float32, bfloat16.
* @li gamma: A tensor of input. Represents a learnable scaling factor.
* Must be one of the following types: float16, float32, bfloat16. Must have the same type as "x".
* The format must be ND. Shape support 1D ~ 8D.
* The shape must meet the requirements of gamma_shape = x_shape[n:], n < x_shape.dims().

* @par Attributes
* epsilon: Optional attribute to prevent division by 0. The type is float32. Defaults to 1e-6.

* @par Outputs
* Two outputs, including:
* @li y: A tensor of output. Indicates normalized data. The format must be ND. Shape support 1D ~ 8D.
* Must be one of the following types: float16, float32, bfloat16. Must have the same type, format and shape as "x".
* @li rstd: A tensor of output. Represents root mean square. The format must be ND. The type is float32.
* The shape is the same as the first several dimensions of the "x" shape.
* The first several dimensions refer to the dimension of "x" minus the dimension of "gamma",
* indicating that the norm is not required.
*/
REG_OP(GemmaRmsNorm)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(rstd, TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .ATTR(epsilon, Float, 1e-6f)
    .OP_END_FACTORY_REG(GemmaRmsNorm)

/**
*@brief Returns a tensor where each sub-tensor of input along dimension
*       dim is normalized such that the p-norm of the sub-tensor is lower than the value maxnorm.

*@par Inputs:
*One input, including:
* x: A tensor. Support one of the following types: float32, float16, bfloat16. The dimension range is [2, 8]. \n

*@par Attributes:
* @li p: Specify L_p norm, the type is float32, the range is greater than or equal to 0.
* @li dim: The processed dim, the type is int64, the range is within [-x dimension, x dimension - 1].
* @li maxnorm: Threshold for comparison, the type is float32, the range is greater than or equal to 0. \n

*@par Outputs:
*One output, including:
* y: shape and dtype of output, should be same shape and type as input.
*/
REG_OP(Renorm)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(p, Float)
    .REQUIRED_ATTR(dim, Int)
    .REQUIRED_ATTR(maxnorm, Float)
    .OP_END_FACTORY_REG(Renorm)

/**
*@brief LayerNormGrad operator interface implementation \n
*  calculating: dy, x, variance, mean, gamma \n
*  pd_xl = data_dy*data_gamma \n
*  pd_var = np.sum(((-0.5)*pd_xl*(data_x - data_mean) 
*           np.power((data_variance + EPSLON), (-1.5))), 
*           reduce_axis, keepdims=True) \n
*  pd_mean = np.sum(((-1.0)*pd_xl 
*            np.power((data_variance + EPSLON), (-0.5))), 
*            reduce_axis, keepdims=True) 
*            + pd_var*(1.0/m) 
*            np.sum(((-2.0)*(data_x - data_mean)), reduce_axis, keepdims=True) \n
*  pd_x = pd_xl*np.power((data_variance + EPSLON), (-0.5)) + 
*         pd_var*(2.0/m)*(data_x - data_mean) + pd_mean*(1.0/m) \n
*  pd_gamma = np.sum((data_dy*(data_x - data_mean) 
*             np.power((data_variance + EPSLON), (-0.5))), param_axis, keepdims=True) \n
*  pd_beta = np.sum(data_dy, param_axis, keepdims=True)

*@par Inputs:
*Five inputs, including:
* @li dy: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li x: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li variance: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li mean: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li gamma: A Tensor. Must be one of the following types: float16, float32, bfloat16 . \n

*@par Outputs:
*Three outputs, including:
* @li pd_x: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li pd_gamma: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li pd_beta: A Tensor. Must be one of the following types: float16, float32, bfloat16.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(LayerNormGrad)
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(variance, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(pd_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(pd_gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(pd_beta, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(LayerNormGrad)

/**
* @brief LayerNormGradV3 operator interface implementation \n
*  calculating: dy, x, rstd, mean, gamma \n
*  pd_xl = data_dy*data_gamma \n
*  pd_var = np.sum(((-0.5)*pd_xl*(data_x - data_mean) 
*           np.power(rstd, 3)), 
*           reduce_axis, keepdims=True) \n
*  pd_mean = np.sum(((-1.0)*pd_xl*rstd), reduce_axis, keepdims=True) 
*            + pd_var*(1.0/m) 
*            np.sum(((-2.0)*(data_x - data_mean)), reduce_axis, keepdims=True) \n
*  pd_x = pd_xl*rstd + 
*         pd_var*(2.0/m)*(data_x - data_mean) + pd_mean*(1.0/m) \n
*  pd_gamma = np.sum((data_dy*(data_x - data_mean)*rstd), param_axis, keepdims=True) \n
*  pd_beta = np.sum(data_dy, param_axis, keepdims=True)

*@par Inputs:
*Five inputs, including:
* @li dy: A tensor. The gradient tensor that represents the reverse calculation.
* Must be one of the following types: float16, float32, bfloat16. The format must be ND. 
* The shape is equal to the shape of x, that is, [A1, ...,Ai,R1, ...,Rj].
* Has the same type and format as x. 
* @li x: A tensor. First input of forward propagation. Must be one of the following types: float16, float32, bfloat16.
* The shape is equal to the shape of dy, that is, [A1, ...,Ai,R1, ...,Rj].
* Has the same type and format as dy. 
* @li rstd: A tensor. Third output of forward propagation, indicating the reciprocal of the standard deviation of input.
* Must be one of the following types: float16, float32, bfloat16. The format must be ND.
* Has the same shape as mean, which is [A1,...,Ai,1,...,1], where there are j 1s after Ai,
* and j is the length of the axis that requires normalization.
* Has the same type and format as mean. 
* @li mean: A tensor. Second output of forward propagation, indicating the mean value of input.
* Must be one of the following types: float16, float32, bfloat16. The format must be ND.
* Has the same shape as rstd, which is [A1,...,Ai,1,...,1], where there are j 1s after Ai,
* and j is the length of the axis that requires normalization.
* Has the same type and format as rstd. 
* @li gamma: A tensor. Indicates the weight tensor.
* Must be one of the following types: float16, float32, bfloat16.
* The format must be ND. Has the same type as x. The shape is [R1,...,Rj].

*@par Outputs:
*Three outputs, including:
* @li pd_x: A tensor. Indicates the first output of the forward calculation, which is the output tensor of the x derivative.
* Must be one of the following types: float16, float32, bfloat16. The format must be ND.
* Has the same type, shape and format as x. 
* @li pd_gamma: A tensor. Indicates the output tensor of gamma derivative.
* Must be one of the following types: float16, float32, bfloat16. The format must be ND.
* Has the same type, shape and format as gamma. 
* @li pd_beta: A tensor. The bias tensor of forward output.
* Must be one of the following types: float16, float32, bfloat16. The format must be ND.
* Has the same type, shape and format as gamma. 

*@par Attributes:
* output_mask: An optional attribute, a list composed of bool. Defaults value is {true, true, true}.
* The output_mask[0] is false, means pd_x is meaningless.
* The output_mask[1] is false, means pd_gamma is meaningless.
* The output_mask[2] is false, means pd_beta is meaningless.
* Except in Ascend 950 AI Processor, the output_mask just supports {true, true, true} in other AI Processor.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(LayerNormGradV3)
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(rstd, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(pd_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(pd_gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(pd_beta, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(output_mask, ListBool, {true, true, true})
    .OP_END_FACTORY_REG(LayerNormGradV3)

/**
*@brief LayerNormXBackprop operator interface implementation \n
*  calculating: dy, x, variance, mean, gamma \n
*  pd_xl = data_dy*data_gamma \n
*  pd_var = np.sum(((-0.5)*pd_xl*(data_x - data_mean)
*           np.power((data_variance + EPSLON), (-1.5))),
*           reduce_axis, keepdims=True) \n
*  pd_mean = np.sum(((-1.0)*pd_xl
*            np.power((data_variance + EPSLON), (-0.5))),
*            reduce_axis, keepdims=True) 
*            + pd_var*(1.0/m)
*            np.sum(((-2.0)*(data_x - data_mean)), reduce_axis, keepdims=True) \n
*  pd_x = pd_xl*np.power((data_variance + EPSLON), (-0.5)) + 
*         pd_var*(2.0/m)*(data_x - data_mean) + pd_mean*(1.0/m) \n
*  pd_gamma = np.sum((data_dy*(data_x - data_mean)
*             np.power((data_variance + EPSLON), (-0.5))), param_axis, keepdims=True) \n
*  pd_beta = np.sum(data_dy, param_axis, keepdims=True)

*@par Inputs:
*Five inputs, including:
* @li dy: A Tensor. Must be one of the following types: float16, float32.
* @li x: A Tensor. Must be one of the following types: float16, float32.
* @li variance: A Tensor. Must be one of the following types: float16, float32.
* @li mean: A Tensor. Must be one of the following types: float16, float32.
* @li gamma: A Tensor. Must be one of the following types: float16, float32 . \n

*@par Outputs:
*One outputs, including:
* @li pd_x: A Tensor. Must be one of the following types: float16, float32.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(LayerNormXBackprop)
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(variance, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(pd_x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(LayerNormXBackprop)

/**
*@brief LayerNormXBackpropV2 operator interface implementation \n
*  calculating: dy, x, variance, mean, gamma \n
*  pd_xl = data_dy*data_gamma \n
*  pd_var = np.sum(((-0.5)*pd_xl*(data_x - data_mean) 
*           np.power((data_variance + EPSLON), (-1.5))), 
*           reduce_axis, keepdims=True) \n
*  pd_mean = np.sum(((-1.0)*pd_xl 
*            np.power((data_variance + EPSLON), (-0.5))), 
*            reduce_axis, keepdims=True) 
*            + pd_var*(1.0/m) 
*            np.sum(((-2.0)*(data_x - data_mean)), reduce_axis, keepdims=True) \n
*  pd_x = pd_xl*np.power((data_variance + EPSLON), (-0.5)) + 
*         pd_var*(2.0/m)*(data_x - data_mean) + pd_mean*(1.0/m) \n
*  res_for_gamma = (data_x - data_mean) * np.power((data_variance + EPSLON), (-0.5))

*@par Inputs:
*Five inputs, including:
* @li dy: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li x: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li variance: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li mean: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li gamma: A Tensor. Must be one of the following types: float16, float32, bfloat16. \n

*@par Outputs:
*Two outputs, including:
* @li pd_x: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li res_for_gamma: A Tensor. Must be one of the following types: float32.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(LayerNormXBackpropV2)
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(variance, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(pd_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(res_for_gamma, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(LayerNormXBackpropV2)

/**
* @brief LayerNormXBackpropV3 operator interface implementation \n
*  calculating: dy, x, rstd, mean, gamma \n
*  pd_xl = data_dy*data_gamma \n
*  pd_var = np.sum(((-0.5)*pd_xl*(data_x - data_mean)
*           np.power(rstd, 3)),
*           reduce_axis, keepdims=True) \n
*  pd_mean = np.sum(((-1.0)*pd_xl*rstd), reduce_axis, keepdims=True)
*            + pd_var*(1.0/m)
*            np.sum(((-2.0)*(data_x - data_mean)), reduce_axis, keepdims=True) \n
*  pd_x = pd_xl*rstd +
*         pd_var*(2.0/m)*(data_x - data_mean) + pd_mean*(1.0/m) \n
*  res_for_gamma = (data_x - data_mean) * rstd

*@par Inputs:
*Five inputs, including:
* @li dy: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li x: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li rstd: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li mean: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li gamma: A Tensor. Must be one of the following types: float16, float32, bfloat16. \n

*@par Outputs:
*Three outputs, including:
* @li pd_x: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li res_for_gamma: A Tensor. Must be one of the following types: float32.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(LayerNormXBackpropV3)
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(rstd, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(pd_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(res_for_gamma, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(LayerNormXBackpropV3)

/**
*@brief LayerNormBetaGammaBackprop operator interface implementation \n
*  calculating: dy, x, variance, mean \n
*  pd_xl = data_dy*data_gamma \n
*  pd_var = np.sum(((-0.5)*pd_xl*(data_x - data_mean) 
*           np.power((data_variance + EPSLON), (-1.5))), 
*           reduce_axis, keepdims=True) \n
*  pd_mean = np.sum(((-1.0)*pd_xl 
*            np.power((data_variance + EPSLON), (-0.5))), 
*            reduce_axis, keepdims=True) 
*            + pd_var*(1.0/m) 
*            np.sum(((-2.0)*(data_x - data_mean)), reduce_axis, keepdims=True) \n
*  pd_x = pd_xl*np.power((data_variance + EPSLON), (-0.5)) + 
*         pd_var*(2.0/m)*(data_x - data_mean) + pd_mean*(1.0/m) \n
*  pd_gamma = np.sum((data_dy*(data_x - data_mean) 
*             np.power((data_variance + EPSLON), (-0.5))), param_axis, keepdims=True) \n
*  pd_beta = np.sum(data_dy, param_axis, keepdims=True)

*@par Inputs:
*Four inputs, including:
* @li dy: A Tensor. Must be one of the following types: float16, float32.
* @li x: A Tensor. Must be one of the following types: float16, float32.
* @li variance: A Tensor. Must be one of the following types: float16, float32.
* @li mean: A Tensor. Must be one of the following types: float16, float32 . \n

*@par Outputs:
*Two outputs, including:
* @li pd_gamma: A Tensor. Must be one of the following types: float16, float32.
* @li pd_beta: A Tensor. Must be one of the following types: float16, float32.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(LayerNormBetaGammaBackprop)
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(variance, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(pd_gamma, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(pd_beta, TensorType({DT_FLOAT, DT_FLOAT16}))
    .REQUIRED_ATTR(shape_gamma, ListInt)
    .OP_END_FACTORY_REG(LayerNormBetaGammaBackprop)

/**
* @brief LayerNormBetaGammaBackpropV2 operator interface implementation \n
*  calculating: dy, x, variance, mean \n
*  pd_gamma = np.sum((data_dy*res_for_gamma), param_axis, keepdims=True) \n
*  pd_beta = np.sum(data_dy, param_axis, keepdims=True)

* @par Inputs:
* Two inputs, including:
* @li dy: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li res_for_gamma: A Tensor. Must be one of the following types: float16, float32, bfloat16.

* @par Outputs:
* Two outputs, including:
* @li pd_gamma: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li pd_beta: A Tensor. Must be one of the following types: float16, float32, bfloat16.

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(LayerNormBetaGammaBackpropV2)
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(res_for_gamma, TensorType({DT_FLOAT}))
    .OUTPUT(pd_gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(pd_beta, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(shape_gamma, ListInt)
    .OP_END_FACTORY_REG(LayerNormBetaGammaBackpropV2)

/**
* @brief LNDropoutGrad operator interface implementation
*   calculating: dy, x, variance, mean, gamma
*   pd_xl = dy*gamma
*   sub_x_mean = x - mean
*   var_elta_2 = np.power((variance + EPSLON), (-0.5))
*   pd_var = sum(pd_xl * sub_x_mean, reduce_axis, keepdims=True) * var_elta_2 * var_elta_2 * var_elta_2 * (-0.5)
*   pd_mean = sum(pd_xl, reduce_axis, keepdims=True) * var_elta_2 * (-1.0)
*   pd_x = pd_xl * var_elta_2 + pd_var * (2.0 / m) * sub_x_mean + pd_mean * (1.0 / m)
*   pd_x_dropout = pd_x * mask * (1 / keep_prob)
*   pd_gamma = sum(dy * sub_x_mean * var_elta_2, param_axis, keepdims=True)
*   pd_beta = sum(dy, param_axis, keepdims=True)

* @par Inputs:
* Six inputs, including:
*  @li dy: A Tensor. Must be one of the following types: float16, float32.
*  @li x: A Tensor. Must be one of the following types: float16, float32.
*  @li variance: A Tensor. Must be one of the following types: float16, float32.
*  @li mean: A Tensor. Must be one of the following types: float16, float32.
*  @li gamma: A Tensor. Must be one of the following types: float16, float32.
*  @li mask: A Tensor. Must be one of the following types: uint8, bool.\n

* @par Outputs:
* Four outputs, including:
*  @li pd_x: A Tensor. Must be one of the following types: float16, float32.
*  @li pd_x_dropout: A Tensor. Must be one of the following types: float16, float32.
*  @li pd_gamma: A Tensor. Must be one of the following types:  float16, float32.
*  @li pd_beta: A Tensor. Must be one of the following types:  float16, float32.

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(LNDropoutGrad)
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(variance, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(mask, TensorType({DT_UINT8, DT_BOOL}))
    .OUTPUT(pd_x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(pd_x_dropout, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(pd_gamma, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(pd_beta, TensorType({DT_FLOAT, DT_FLOAT16}))
    .REQUIRED_ATTR(keep_prob, Float)
    .OP_END_FACTORY_REG(LNDropoutGrad)

/**
* @brief Return "output" according to the algorithm of dropout_do_mask:  \n
*  scale_x = x *(1 / keep_prob)  \n
*  output = select(mask == 1, scale_x, 0)

* @par Inputs:
* Three inputs, including:
* @li x: A mutable Tensor. A ND tensor. Support 1D ~ 8D. Must be one of the following types:
*     float16, float32, bfloat16 (bfloat16 only supported on Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component
* and Atlas A3 Training Series Product/Atlas A3 Inference Series Product).
* @li mask: A mutable Tensor. A ND tensor. Must met all of the following rules:
*     dtype of mask should be uint8 or uint1.
*     if data type of mask is uint8, shape of mask should be 1D. value of shape should met the following algorithm:
*     value = (size(x) + 128 - 1) // 128 * 128 // 8
*     if data type of mask is uint1, shape of mask should be same to x.
* @li keep_prob: A mutable Tensor. A ND tensor. Must met all of the following rules:
*     0 <= keep_prob <= 1
*     shape of "keep_prob" should be (1,) or [1,].
*     Has the same type as "x" . \n

* @par Outputs:
* y: A mutable Tensor. A ND tensor. Support 1D ~ 8D. Has the same type, shape and format as "x".
*/
REG_OP(DropOutDoMask)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(mask, TensorType({DT_UINT8, DT_UINT1}))
    .INPUT(keep_prob, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(DropOutDoMask)

/**
*@brief Return "output" according to the algorithm of dropout_do_mask:  \n
*  scale_x = x *(1 / keep_prob) \n
*  output = select(mask == 1, scale_x, 0) \n

*@par Inputs:
*Three inputs, including:
* @li x: A mutable Tensor. Must be one of the following types:
*     float16, float32
* @li mask: A mutable Tensor. Must met all of the following rules:
*     shape of mask should be 1D.
*     dtype of mask should be uint8 or bool.
*     value of shape should met the following algorithm:
*     value = (size(x) + 128 - 1) // 128 * 128
* @li keep_prob: A mutable Tensor. Must met all of the following rules:
*     shape of "keep_prob" should be (1,) or [1,].
*     Has the same type as "x" . \n

*@par Outputs:
*y: A mutable Tensor. Has the same type as "x".
*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(DropOutDoMaskV3)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(mask, TensorType({DT_UINT8, DT_BOOL}))
    .INPUT(keep_prob, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(DropOutDoMaskV3)

/**
*@brief Return "output" according to the algorithm of dropout_do_mask:
*  scale_x = x *(1 / keep_prob)
*  output = select(mask == 1, scale_x, 0)

*@par Inputs:
*Two inputs, including:
* @li x: A mutable Tensor. Must be one of the following types:
*     float16, float32
* @li mask: A mutable Tensor. Must met all of the following rules:
*     shape of mask should be 1D.
*     dtype of mask should be uint8 or bool.
*     value of shape should met the following algorithm:
*     value = (size(x) + 128 - 1) // 128 * 128
*@par Attributes:
* @li keep_prob: A mutable Tensor. Must met all of the following rules:
*     shape of "keep_prob" should be (1,) or [1,].
*     Has the same type as "x" . \n

*@par Output:
*y: A mutable Tensor. Has the same type as "x".
*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(DropOutDoMaskV3D)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(mask, TensorType({DT_UINT8, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .REQUIRED_ATTR(keep_prob, Float)
    .OP_END_FACTORY_REG(DropOutDoMaskV3D)

/**
* @brief Scales the input .

* @par Inputs:
* Three inputs, including:
* @li x: An ND tensor of type float16 or float32 or bfloat16.
* @li scale: An ND tensor of type float16 or float32 or bfloat16
* @li bias: An optional ND tensor of type float16 or float32 or bfloat16. \n

* @par Attributes:
* @li axis: An optional int32 used to compute the shape of scale and bias input from the online bottoms.
        Defaults to "1".
* @li num_axes: An optional int32 used to compute the shape of scale and bias input from a Caffe model trained offline.
        Defaults to "1".
* @li scale_from_blob: An optional bool. If "true", scale and bias are input from a Caffe model trained offline.
        If "false", scale and bias are input from online bottoms. Defaults to "true" . \n

* @par Outputs:
* y: An ND tensor of type float16 or float32 or bfloat16. \n

* @attention Constraints:
* Assume that the shape length of "x" is "n" and that of "scale" is "m".
* @li "axis" is within the range [-n, n-1]. num_axes >= -1.
* @li If "scale_from_blob = true", "num_axes = -1", and "axis >= 0",
        the ith axis of "scale" and the (i+"axis")th axis of "x" must have the same size (0 <= i < n-axis).
* If "axis < 0", the ith axis of "scale" and the (i+n+"axis")th axis of "x" must have the same size (0 <= i < -axis).
* @li If "scale_from_blob = true" and "num_axes = 0", "scale" is a scalar with shape length 1 and dimension size 1.
* @li If "scale_from_blob = true", "num_axes > 0, and "axis >= 0", "axis + num_axes" must be less than or equal to "n"
        and the ith axis of "scale" and the (i+"axis")th axis of "x" must have the same size (0 <= i < num_axes).
* If "axis < 0", "n + axis + num_axes" must be less than or equal to "n" and the ith axis of "scale"
        and the (i+n+"axis")th axis of "x" must have the same size (0 <= i < num_axes).
* @li If "scale_from_blob = false", "scale" is not a scalar, and "axis >= 0","axis + m" must be less than or
        equal to "n" and the ith axis of "scale" and the (i+"axis")th axis of "x" must have the same size (0 <= i < m).
* If "axis < 0", "n + axis + m" must be less than or equal to "n" and the ith axis of "scale" and
        the (i+n+"axis")th axis of "x" must have the same size (0 <= i < m).
* @li If "bias" is not None, the constraints for "bias" is the same as that for "scale".
* @par Third-party framework compatibility
* Compatible with the Caffe operator Scale.
*/
REG_OP(Scale)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16})) /* "First operand." */
    .INPUT(scale, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16})) /* "Second operand." */
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16})) /* "Third operand." */
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))  /* "Result, has same element type as x" */
    .ATTR(axis, Int, 1)
    .ATTR(num_axes, Int, 1)
    .ATTR(scale_from_blob, Bool, true)
    .OP_END_FACTORY_REG(Scale)

/**
*@brief Local Response Normalization .

*@par Inputs:
*One input, including:
*x: A Tensor. Must be 4-D shape, and only support the following types: float16, float32 . \n

*@par Attributes:
*@li depth_radius: An optional int32, specifying the half-width of the normalization window. Defaults to "5".
* under the caffe framework, if local_size is provided and is an odd number,
* depth_radius = (local_size - 1) / 2. local_size is the number of channels to sum over (for ACROSS_CHANNELS)
* or the side length of the square region to sum over (for WITHIN_CHANNEL).
*@li bias: An optional float32. An offset, usually > 0 to avoid dividing by 0.
* Defaults to "1.0".
*@li alpha: An optional float32. A scaling factor, usually positive.
* Defaults to "1.0".
*@li beta: An optional float32. An exponent. Defaults to "0.75" for the caffe framework, Defaults to "0.5" for others.
*@li norm_region: An optional string. A mode option. "ACROSS_CHANNELS":0. Defaults to "ACROSS_CHANNELS" . \n

*@par Outputs:
*y: A Tensor. Has the same data type and shape as "x" . \n

* @attention Constraints:
* This operator will be deprecated in the future. Replace it with LayerNorm operator. \n

*@par Third-party framework compatibility:
* Compatible with the TensorFlow operator LRN.
*/
REG_OP(LRN)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .ATTR(depth_radius, Int, 5)
    .ATTR(bias, Float, 1.0)
    .ATTR(alpha, Float, 1.0)
    .ATTR(beta, Float, 0.5)
    .ATTR(norm_region, String, "ACROSS_CHANNELS")
    .OP_END_FACTORY_REG(LRN)

/**
* @brief Computes the gradient for Local Response Normalization .

* @par Inputs:
* @li grads: A 4D Tensor of type float16 or float32.
* @li x: A 4D Tensor of type float16 or float32.
* @li y: A 4D Tensor of type float16 or float32 .

* @par Attributes:
* @li depth_radius: An optional int, specifying the half-width of the
* normalization window. Defaults to "5".
* @li bias: An optional float32. An offset, usually > 0 to avoid dividing by 0.
* Defaults to "1".
* @li alpha: An optional float32. A scaling factor, usually positive.
* Defaults to "1".
* @li beta: An optional float32. An exponent. Defaults to "0.5" .

* @par Outputs:
* z: A Tensor. Has the same type and shape as "grads" .

* @attention Constraints:
* "x" and "y" must have the same shape and type as "grads" .

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator LRNGrad.
*/
REG_OP(LRNGrad)
    .INPUT(grads, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(z, TensorType({DT_FLOAT16,DT_FLOAT}))
    .ATTR(depth_radius, Int, 5)
    .ATTR(bias, Float, 1.0)
    .ATTR(alpha, Float, 1.0)
    .ATTR(beta, Float, 0.5)
    .OP_END_FACTORY_REG(LRNGrad)

 /**
 *@brief Calculates the RNNT Loss (log probability) for each batch entry.
 Also calculates the gradient.

 *@par Inputs:
 *@li acts: 4-D, shape: `(batch x seqLength x labelLength x outputDim)`, the logits.
 *@li labels: 2-D Tensor containing all the targets of the batch with zero padded.
 *@li input_lengths: Tensor of size (batch) containing size of each output sequence.
 *@li label_lengths: Tensor of (batch) containing label length of each example.

 *@par Outputs:
 *@li costs: 1-D Tensor, the cost of each example in the batch.
 *@li grads: A Tensor. Has the same type as acts.

 *@par Attributes:
 *blank_label: An optional attribute. Defaults to 0.

 *@par Third-party framework compatibility
 * Compatible with TensorFlow RNNTLoss operator.
 */
REG_OP(RNNTLoss)
    .INPUT(acts, TensorType({DT_FLOAT}))
    .INPUT(labels, TensorType({DT_INT32}))
    .INPUT(input_lengths, TensorType({DT_INT32}))
    .INPUT(label_lengths, TensorType({DT_INT32}))
    .ATTR(blank_label, Int, 0)
    .OUTPUT(costs, TensorType({DT_FLOAT}))
    .OUTPUT(grads, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(RNNTLoss)

/**
*@brief Performs instance normalization .

*@par Inputs:
* Five inputs, including:
*@li x: A 5D Tensor of type float16 or float32.
*@li gamma: A Tensor of type float32.
A 5D Tensor for scaling factor, to scale the normalized x.
*@li beta: A Tensor of type float32.
A 5D Tensor for offset, to shift to the normalized x.
*@li mean: A Tensor of type float32.
A 5D Tensor Specifies the mean used for inference. Reserved.
*@li variance: A Tensor of type float32.
A 5D Tensor Specifies the variance used for inference. Reserved . \n

*@par Attributes:
*@li is_training: An optional bool, specifying if the operation is used for
training or inference. Defaults to "True".
*@li momentum: An optional float32,
the value used for the running_mean and running_var computation. Default: "0.1".
*@li epsilon: An optional float32, specifying the small value added to
variance to avoid dividing by zero. Defaults to "0.00001" . \n

*@par Outputs:
* Three outputs, including: (NHWC, NCHW supported)
*@li y: A 5D tensor of type float16 or float32 for the normalized "x",
*@li batch_mean: A Tensor of type float32.
Specifies the mean of "x".
*@li batch_variance: A Tensor of type float32.
Specifies the variance of "x" . \n

*@par Third-party framework compatibility
*@li Compatible with the PyTorch operator InstanceNorm.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(InstanceNormV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(gamma, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(beta, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(mean, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(variance, TensorType({DT_FLOAT}))

    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))

    .ATTR(is_training, Bool, true)
    .ATTR(momentum, Float, 0.1f)
    .ATTR(epsilon, Float, 0.00001f)
    .OP_END_FACTORY_REG(InstanceNormV2)

/**
*@brief Performs instance normalization for inference.

*@par Inputs:
* Six inputs, including:
*@li x: A Tensor of type float16 or float32.
*@li gamma: A [N, C1, 1, 1, C0] optional Tensor of type float32, for the scaling gamma.
*@li beta: A [N, C1, 1, 1, C0] optional Tensor of type float32, for the scaling beta.
*@li mean: A [N, C1, 1, 1, C0] optional Tensor of type float32, for the mean.
*@li variance: A [N, C1, 1, 1, C0] optional Tensor of type float32, for the variance.
*@li variance_sqrt: A [N, C1, 1, 1, C0] optional Tensor of type float32, for the variance_sqrt.

*@par Outputs:
*@li y: A Tensor of type float16 or float32 for the normalized "x".
*@li batch_mean: A Tensor of type float32 for the result mean.
*@li batch_variance: A Tensor of type float32 for the result variance.

*@attention Constraints:
*For Atlas 200/300/500 Inference Product, the result accuracy fails to reach 1<89> due to the square root instruction.

* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use INInferV2 instead.
*/
REG_OP(INInferV2D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(gamma, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(beta, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(mean, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(variance, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(variance_sqrt, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(INInferV2D)

/**
* @brief InstanceNorm operator interface implementation.

* @par Inputs
* Three inputs, including:
* @li x: A 4D or 5D Tensor. Support dtype: [float32, float16],
*  support format: [NCHW, NHWC, NDHWC, NCDHW].
* @li gamma: A 4D or 5D Tensor. Support dtype: [float32, float16],
*  support format: [NCHW, NHWC, NDHWC, NCDHW](DHW=1).
* @li beta: A 4D or 5D Tensor. Support dtype: [float32, float16],
*  support format: [NCHW, NHWC, NDHWC, NCDHW](DHW=1).

* @par Attributes
* @li data_format: An optional attribute. The type is string. Default to "NDHWC".
* @li epsilon: An optional attribute. The type is float. Default to 1e-6.

* @par Outputs
* Three outputs, including:
* @li y: A 4D or 5D Tensor. Support dtype: [float32, float16],
*  support format: [NCHW, NHWC, NDHWC, NCDHW]. Has the same type as "x".
* @li mean: A 4D or 5D Tensor. Support dtype: [float32, float16],
*  support format: [NCHW, NHWC, NDHWC, NCDHW](DHW=1). Has the same type as "x".
* @li variance: A 4D or 5D Tensor. Support dtype: [float32, float16],
*  support format: [NCHW, NHWC, NDHWC, NCDHW](DHW=1). Has the same type as "x".
*/
REG_OP(InstanceNorm)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(beta, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(mean, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(variance, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(data_format, String, "NDHWC")
    .ATTR(epsilon, Float, 1e-6f)
    .OP_END_FACTORY_REG(InstanceNorm)

/**
* @brief InstanceNormGrad operator interface implementation.

* @par Inputs:
* Five inputs, including:
* @li dy: Represents the input gradient tensor. Support dtype: float16, float32. Suppor shape 4D or 5D.
* Support format: [NCHW, NHWC, NDHWC, NCDHW]. Has the same dtype, format and shape as "x".
* @li x: Represents the input tensor. Support dtype: float16, float32. Suppor shape 4D or 5D.
* Support format: [NCHW, NHWC, NDHWC, NCDHW].
* @li variance: Represents the variance tensor. Support dtype: float16, float32. Suppor shape 4D or 5D.
* Support format: [NCHW, NHWC, NDHWC, NCDHW](DHW=1). Has the same dtype and format as "x".
* The shapes of "variance" and "mean" are consistent, and the N and C axes are consistent with those of "x", and the other dimensions are 1.
* @li mean: Represents the mean tensor. Support dtype: float16, float32. Suppor shape 4D or 5D.
* Support format: [NCHW, NHWC, NDHWC, NCDHW](DHW=1). Has the same dtype and format as "x".
* The shapes of "variance" and "mean" are consistent, and the N and C axes are consistent with those of "x", and the other dimensions are 1.
* @li gamma: Represents the optional weight parameter. Support dtype: float16, float32. Suppor shape 4D or 5D.
* Support format: [NCHW, NHWC, NDHWC, NCDHW].  Has the same dtype as "x".
* The C axis of "gamma" is consistent with that of "x", and the other dimensions are 1. \n

* @par Outputs:
* Three outputs, including:
* @li pd_x: Represents the gradient tensor of the input tensor. Support dtype: float16, float32. Suppor shape 4D or 5D.
* Support format: [NCHW, NHWC, NDHWC, NCDHW].  Has the same dtype, format and shape as "x".
* @li pd_gamma: Represents the gradient tensor of the weight parameter. Support dtype: float16, float32. Suppor shape 4D or 5D.
* Support format: [NCHW, NHWC, NDHWC, NCDHW]. Has the same dtype and format as "x". Has the same shape as "gamma".
* @li pd_beta: Represents the gradient tensor of the bias parameter. Support dtype: float16, float32. Suppor shape 4D or 5D.
* Support format: [NCHW, NHWC, NDHWC, NCDHW]. Has the same dtype and format as "x". Has the same shape as "gamma".
*/
REG_OP(InstanceNormGrad)
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(variance, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(pd_x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(pd_gamma, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(pd_beta, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(InstanceNormGrad)

/**
* @brief Computes Kl_div_loss_grad or Kl_div_loss_backward. \n

* @par Inputs:
* Three inputs, including:
* @li grad: A tensor. Must be one of the following types: float16, float32, bfloat16.
*  Shape needs to satisfy the broadcast relationship with input.
*  Required.
* @li input: A tensor. Has the same type as "grad". Required.
* @li target: A tensor. Has the same type as "grad". Required.
*  Shape needs to satisfy the broadcast relationship with input. \n

* @par Attributes:
* @li reduction: An optional attribute of type String. Defaults to "mean".
*  Supports "none" | "mean" | "sum" | "batchmean".  \n
*  'none' means no reduction should be applied.  \n
*  'mean' means the total output will be divided by the number of elements in the output.  \n
*  'sum' means the output will be summed. \n
*  'batchmean' means the total output will be divided by the number of batches. \n
* @li log_target: An optional attribute of type Bool. Defaults to false. \n

* @par Outputs:
* y: A tensor. Has the same type as "grad".
* Has the same shape as "input"  \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator KlDivLossGrad.
*/
REG_OP(KlDivLossGrad)
    .INPUT(grad, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(input, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(target, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(reduction, String, "mean")
    .ATTR(log_target, Bool, false)
    .OP_END_FACTORY_REG(KlDivLossGrad)

/**
* @brief Computes l1_loss_grad or l1_loss_backward.

* @par Inputs:
* Three inputs, including:
* @li grads: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* Required.
* @li predict: A Tensor. Has the same type as "grads". Required.
* @li label: A Tensor. Has the same type as "grads". Required. \n

* @par Attributes:
* reduction: An optional attribute of type String. Defaults to "mean". \n

* @par Outputs:
* y: A Tensor. Has the same type as "grads". \n

* @attention Constraints:
* @li In Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component and
* Atlas A3 Training Series Product/Atlas A3 Inference Series Product,
* broadcasting is not allowed between input grads, input predict and input label, the three inputs must have the same shape. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator L1LossGrad.
*/
REG_OP(L1LossGrad)
    .INPUT(grads, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(label, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(L1LossGrad)

/**
* @brief Computes loss of lp, p=1,2,3....

* @par Inputs:
* @li predict: An ND tensor of type float16, float32 or bfloat16. The predicted value.
* @li label: An ND tensor of type float16, float32 or bfloat16. The golden value.

* @par Attributes:
* @li p: A required int attribute that decides which loss to compute, now the p only can be 1 to compute l1_loss.
* @li reduction: An optional string which specifies the reduction to apply to the output. It can be "mean","sum" or "none". Defaults to "mean". 
* "none": no reduction will be applied.
* "mean": the sum of the output will be divided by the number of elements in the output.
* "sum": the output will be summed.

* @par Outputs:
*  y: An ND tensor tensor with the same shape and type as "predict". 

* @par Third-party framework compatibility
* Compatible with the Pytorch operator LpLoss.
*/
REG_OP(LpLoss)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(label, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(p, Int)
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(LpLoss)

/**
* @brief Computes gradients of mse loss.

* @par Inputs:
* @li predict: An ND tensor of type float16, float32 or bfloat16.
* @li label: An ND tensor of type float16, float32 or bfloat16.
* @li dout: An ND tensor of type float16, float32 or bfloat16. \n

* @par Attributes:
* reduction: An optional string.Defaults to "mean". \n

* @par Outputs:
* y: An ND tensor tensor with the same shape and type as "predict". \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator MseLossGrad.
*/
REG_OP(MseLossGrad)
    .INPUT(predict, TensorType({DT_FLOAT32, DT_FLOAT16, DT_BF16}))
    .INPUT(label, TensorType({DT_FLOAT32, DT_FLOAT16, DT_BF16}))
    .INPUT(dout, TensorType({DT_FLOAT32, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT32, DT_FLOAT16, DT_BF16}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(MseLossGrad)

/**
* @brief Computes mse loss.
* @par Inputs:
* two inputs, including:
*  @li predict: An ND Tensor of dtype float16, float32 or bfloat16.
*  @li label: An ND Tensor of dtype float16, float32 or bfloat16.\n
*
* @par Attributes:
* reduction:An optional str from sum, none, mean, Defaults to "mean".\n
*
* @par Outputs:
* y: when reduction=sum/mean, y is scale. when reduction=none, y has
*    same type and shape as "predict".\n
*/
REG_OP(MseLoss)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(label, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(MseLoss)

/**
* @brief Calculates the reversed outputs of the function "smooth_l1_loss_v2". Broadcasting is supported.

* @par Inputs:
* Three Inputs, including:
* @li predict: A ND multi-dimensional tensor of type float16, bfloat16 or float32, specifying the predictive value.
* @li label: A ND multi-dimensional tensor, has same type and shape as "predict", specifying the target value.
* @li dout: A tensor. Has the same type as "predict". \n

* @par Attributes:
* Two Attributes, including:
* @li sigma: An optional float. Specifies the threshold of loss. The value must be non-negative. Defaults to 1.0. \n

* @li reduction: An optional string. Specifies the reduction to apply to
* the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied;
* 'mean': the sum of the output will be divided by the number of elements in
* the output;'sum': the output will be summed. Default: 'mean'. \n

* @par Outputs:
*  gradient: A tensor. Has the same type as "predict". \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator SmoothL1LossBackward.
*/
REG_OP(SmoothL1LossGradV2)
    .INPUT(predict, TensorType({DT_FLOAT, DT_FLOAT16,DT_BF16}))
    .INPUT(label, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(dout, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(gradient, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(sigma, Float, 1.0)
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(SmoothL1LossGradV2)

/**
* @brief Creates a criterion that uses a squared term if the absolute
* element-wise error falls below sigma and an L1 term otherwise. It is
* less sensitive to outliers than the MSELoss and in some cases prevents
* exploding gradients.

* @par Inputs:
* @li predict: A ND multi-dimensional tensor of type float16, bfloat16 or float32, 
* specifying the predictive value. \n
* @li label: A ND multi-dimensional tensor, has same type and shape as "predict", 
* specifying the target value. \n

* @par Attributes:
* @li sigma: An optional float. Specifies the threshold of loss. The value must be positive number. Defaults
* to "1.0". \n
* @li reduction: An optional string. Specifies the reduction to apply to
* the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied;
* 'mean': the sum of the output will be divided by the number of elements in
* the output;'sum': the output will be summed. Default: 'mean'. \n

* @par Outputs:
* loss: A multi-dimensional Tensor of type float16, bfloat16 or float32. Indicates the loss between the predictive value and target value.
* Has the same dimensions as "predict". \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator smooth_l1_loss. \n
*/
REG_OP(SmoothL1LossV2)
    .INPUT(predict, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(label, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(loss, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(sigma, Float, 1.0)
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(SmoothL1LossV2)

/**
* @brief Computes Centralization. result = x - mean(x, axes)

* @par Inputs:
*  x: An ND tensor of type float16, float32.
* @par Attributes:
* axes: The dimensions to reduce. Must be one of the following types: int, list, tuple, NoneType, default: -1.
* Must be in the range [-rank(x), rank(x)).
* @par Outputs:
* y: A Tensor. Has the same type as "x". \n

* @par Third-party framework compatibility
* custom operator \n
*/
REG_OP(Centralization)
    .INPUT(x, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .OUTPUT(y, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .ATTR(axes, ListInt, {-1})
    .OP_END_FACTORY_REG(Centralization)

/**
*@brief Roll the tensor along the given dimension(s).
* Elements that are shifted beyond the last position are re-introduced at the first position.
* If a dimension is not specified, the tensor will be flattened before rolling and then restored to the original shape.

*@par Inputs:
*One input, including:
* x: A tensor. Must be one of the following types:
*     float16, bfloat16, float32, int32, uint32, int8, uint8, int64. \n

*@par Attributes:
* @li shifts: A required listInt. The number of places by which the elements of the tensor are shifted. 
*      The length of the array must be equal to dims, but if dims is empty, the array must be one-dimensional. \n
* @li dims: An optional listInt. Defaults to "None". Axis along which to roll. The value of the range is [0, number of dimensions for input -1] or [-number of dimensions for self, -1].

*@par Outputs:
* y: A Tensor with the same type and shape of x. \n

*@par Third-party framework compatibility
*Compatible with the Pytorch operator Roll. \n
*/
REG_OP(Roll)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT32, DT_INT8, DT_UINT8, DT_BF16, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT32, DT_INT8, DT_UINT8, DT_BF16, DT_INT64}))
    .REQUIRED_ATTR(shifts, ListInt)
    .ATTR(dims, ListInt, {})
    .OP_END_FACTORY_REG(Roll)

/**
* @brief Roll the tensor along the given dimension(s).

* @par Inputs:
* One inputs, including:
* @li x: A tensor. Must be one of the following types: int8, uint8, int16, uint16, int32, int64, float16, float32, double.
* @li shift: The number of places by which the elements of the tensor are shifted. Must be one of the following types: int32, int64.
* @li axes: Axis along which to roll. Must be one of the following types:  int32, int64. \n

* @par Outputs:
* y: A Tensor with the same type and shape of x's. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator Roll. \n
*/
REG_OP(RollV2)
    .INPUT(input, TensorType({DT_INT8,DT_UINT8,DT_INT16,DT_UINT16,DT_INT32,DT_INT64,DT_FLOAT16, \
                            DT_FLOAT,DT_DOUBLE}))
    .INPUT(shift, TensorType({DT_INT32,DT_INT64}))
    .INPUT(axes, TensorType({DT_INT32,DT_INT64}))
    .OUTPUT(output, TensorType({DT_INT8,DT_UINT8,DT_INT16,DT_UINT16,DT_INT32,DT_INT64,DT_FLOAT16, \
                            DT_FLOAT,DT_DOUBLE}))
    .OP_END_FACTORY_REG(RollV2)

/**
 * @brief Calculate the loss. Creates a criterion that optimizes a two-class classification
 * logistic loss between input_x and input_y (containing 1 or -1).

 * @par Inputs:
 * Tow inputs, including:
 * @li input_x: A tensor. Must be one of the following types:
 *     float16, float32, bfloat16. \n
 * @li input_y: A tensor. Must be one of the following types:
 *     float16, float32, bfloat16. \n

 * @par Attributes:
 * reduction: An optional string. Defaults to "mean". \n

 * @par Outputs:
 * output_z: while reduction == "none", A Tensor with the same type and shape of input_x's. \n
 *          while reduction == "sum" or "mean", A Tensor with the same type of input_x , shape of which is (1,)

 * @par Third-party framework compatibility
 * Compatible with the Pytorch operator SoftMarginLoss. \n
 */
REG_OP(SoftMarginLoss)
    .INPUT(input_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(input_y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(reduction, String, "mean")
    .OUTPUT(output_z, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(SoftMarginLoss)

/**
* @brief Computes gradients of sigmoid_cross_entropy_with_logits_v2.

* @par Inputs:
* @li predict: An ND tensor of type float16, float32, bfloat16, specifying the predictive value.
* @li target: An ND tensor of type float16, float32, bfloat16, specifying the target value. Shape must be the same as predict.
* @li dout: An ND tensor of type float16, float32, bfloat16, specifying the dout value. Shape needs to satisfy the broadcast relationship with predict.
* @li weight: An optional ND tensor of type float16, float32, bfloat16, specifying the weight value. Shape needs to satisfy the broadcast relationship with predict.
* @li pos_weight: An optional ND tensor of type float16, float32, bfloat16, specifying the pos_weight value. Shape needs to satisfy the broadcast relationship with predict. \n

* @par Attributes:
* reduction: A character string from "none", "mean", and "sum",
* specifying the reduction type to be applied to the output. Defaults to "mean". \n

* @par Outputs:
* gradient: An ND tensor tensor with the same shape and type as "predict". \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator SigmoidCrossEntropyWithLogitsGrad.
*/
REG_OP(SigmoidCrossEntropyWithLogitsGradV2)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(target, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(dout, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(pos_weight, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(gradient, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(SigmoidCrossEntropyWithLogitsGradV2)
/**
 * @brief Calculate the PoissonNllLoss function.  \n
 *        target follow distribution of Poisson(input)loss(input,target) = input - target * log(input) + log(target!)

 * @par Inputs:
 * Two inputs, including:
 * @li input_x: A tensor. Must be one of the following types: float16, float32.
 * @li target: A tensor. Must be one of the following types: float16, float32. \n

 * @par Attributes:
 * four Attributes, including:
 * @li log_input: An optional bool. Defaults to "True"
 * @li full: An optional bool. Defaults to "False"
 * @li eps: An optional float. Defaults to "1e-8"
 * @li reduction: An optional string from "none", "mean",and "sum". Defaults to "mean" \n

 * @par Outputs:
 * loss: A Tensor has same element type as two inputs. \n

 * @par Third-party framework compatibility
 * Compatible with the Pytorch operator PoissonNllLoss. \n
 */
REG_OP(PoissonNllLoss)
    .INPUT(input_x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(target, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(loss, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(log_input, Bool, true)
    .ATTR(full, Bool, false)
    .ATTR(eps, Float, 1e-8f)
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(PoissonNllLoss)
/**
 *@brief rnn_gen_mask
 * @par Inputs:
 * seq_length: A ND Tensor of type int32. Record the current length of each batch.\n
 *
 * @par Attributes:
 * @li num_step: A required int.\n
 * @li hidden_size: A required int. \n
 *
 *
 * @par Ouputs:
 * y: A mutable Tensor of type float16, with the shape of [num_step, batch_size, hidden_size]. \n
 *
 */
REG_OP(RnnGenMask)
    .INPUT(seq_length, TensorType({DT_INT32}))
    .OUTPUT(seq_mask, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(num_step, Int)
    .REQUIRED_ATTR(hidden_size, Int)
    .OP_END_FACTORY_REG(RnnGenMask)

/**
* @brief Creates a criterion that optimizes a multi-class multi-classification hinge loss (margin-based loss)
*        between input x (a 2D mini-batch Tensor) and output y (which is a 2D Tensor of target class indices) \n

* @par Inputs:
* Two inputs, including:
* @li x: A tensor. Must be one of the following types:
*     float16, float32, bfloat16.
* @li target: A tensor. Must be the following types:
*     int32. \n

* @par Attributes:
* reduction: An optional string. Defaults to "mean" \n

* @par Outputs:
* @li y: A Tensor has same element type as input x. \n
* @li is_target: A Tensor has same element type as input target. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator MultiLabelMarginLoss. \n
*/
REG_OP(MultilabelMarginLoss)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(target, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(is_target, TensorType({DT_INT32}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(MultilabelMarginLoss)

/**
* @brief Performs batch normalization . \n
* @par Inputs:
* Two inputs
* @li input_x: A Tensor. Support float32. shape (n, c, d).
* @li seq_len: A Tensor. Each batch normalize data num. Support Int32. Shape (n, ). \n
* @par Attributes:
* @li normalize_type: Str. Support "per_feature" or "all_features".
* @li epsilon: An optional float32, specifying the small value added to
* variance to avoid dividing by zero. Defaults to "0.00001" . \n
* @par Outputs:
* One outputs
* @li output_y: A Tensor for the normalized "x".Support float32. shape (n, c, d).\n
*/
REG_OP(NormalizeBatch)
    .INPUT(input_x, TensorType({ DT_FLOAT }))
    .INPUT(seq_len, TensorType({ DT_INT32 }))
    .OUTPUT(output_y, TensorType({ DT_FLOAT }))
    .REQUIRED_ATTR(normalize_type, String)
    .ATTR(epsilon, Float, 0.00001f)
    .OP_END_FACTORY_REG(NormalizeBatch)

/**
*@brief GroupNorm and Reul operator \n
*  calculating: x, gamma, beta \n
*  y = relu(gamma*((x - mean) / np.sqrt(variance + 0.001)) + beta)

* @par Inputs:
* Three inputs, including:
* @li x: A Tensor. Must be one of the following types: float16, float32.
* @li gamma: A Tensor. Must be one of the following types: float16, float32.
* @li beta: A Tensor. Must be one of the following types: float16, float32 . \n

* @par Attributes:
* @li num_groups: A require attribute, the type is int32.
* @li eps: A optional attribute, the type is float32. Defaults to 0.00001. \n

* @par Outputs:
* One outputs, including:
* @li y: A Tensor. Must be one of the following types: float16, float32.
* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use/
*/
REG_OP(GroupNormRelu)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(beta, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .REQUIRED_ATTR(num_groups, Int)
    .ATTR(eps, Float, 0.00001f)
    .OP_END_FACTORY_REG(GroupNormRelu)

/**
 * @brief Computes the gradients for Batch Normalization.
 * This operation computes the gradients of the input tensor, weight, and bias during the backpropagation step
 * of batch normalization, using the saved mean and inverse standard deviation from the forward pass. It is typically
 * used in the context of training deep learning models with batch normalization layers.

* @par Inputs:
* @li dy: Gradient of the loss w.r.t the output, a tensor of type float16/bfloat16/float32. Supported formats: NCHW, NHWC, NDHWC, NCDHW.
* @li x: Input feature map, a tensor of type float16/bfloat16/float32. Supported formats: NCHW, NHWC, NDHWC, NCDHW.
* @li weight: Scale parameter, a 1D tensor of type float16/bfloat16/float32. Supported formats: ND.
* @li running_mean: Running mean, a 1D tensor of type float16/bfloat16/float32. Supported formats: ND.
* @li running_var: Running variance, a 1D tensor of type float16/bfloat16/float32. Supported formats: ND.
* @li save_mean: Saved mean from the forward pass, a 1D tensor of type float32. Supported formats: ND.
* @li save_rstd: Saved inverse standard deviation from the forward pass, a 1D tensor of type float32. Supported formats: ND.
*
* @par Attributes:
* @li is_training: (Optional) A bool value. Whether the operation is in training mode. Default: true
* @li epsilon: (Optional) A small float32 value added for numerical stability. Default: 1e-5

* @par Outputs:
* @li dx: Gradient of the loss w.r.t the input, a tensor of type float16/bfloat16/float32. Supported formats: NCHW, NHWC, NDHWC, NCDHW.
* @li dweight: Gradient of the loss w.r.t the scale parameter, a 1D tensor of type float16/bfloat16/float32. Supported formats: ND.
* When is_training is false, this output is meaningless.
* @li dbias: Gradient of the loss w.r.t the bias, a 1D tensor of type float16/bfloat16/float32. Supported formats: ND.
* When is_training is false, this output is meaningless.
*/

REG_OP(BatchNormGradV3)
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(weight, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(running_mean, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(running_var, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(save_mean, TensorType({DT_FLOAT}))
    .INPUT(save_rstd, TensorType({DT_FLOAT}))
    .OUTPUT(dx, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(dweight, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(dbias, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(is_training, Bool, true)
    .ATTR(epsilon, Float, 1e-5)
    .OP_END_FACTORY_REG(BatchNormGradV3)

/**
 * @brief DeepNorm operator. \n
 *  calculating: x, gx, gamma, beta, alpha \n
 *  new_x = x * alpha + gx \n
 *  y = gamma*(new_x - mean) / np.sqrt(variance + 1e-6) + beta

 * @par Inputs
 * Four inputs, including:
 * @li x: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
 * @li gx: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
 * @li gamma: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
 * @li beta: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].

 * @par Attributes
 * @li alpha: An optional attribute, the type is float. Defaults to 0.3.
 * @li eps: An optional attribute, the type is float. Defaults to 1e-06.

 * @par Outputs
 * Three outputs, including:
 * @li mean: A Tensor. Support dtype: [float32], support format: [ND].
 * @li rstd: A Tensor. Support dtype: [float32], support format: [ND].
 * @li y: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
 */
REG_OP(DeepNorm)
  .INPUT(x, TensorType({DT_FLOAT,DT_FLOAT16,DT_BF16}))
  .INPUT(gx, TensorType({DT_FLOAT,DT_FLOAT16,DT_BF16}))
  .INPUT(beta, TensorType({DT_FLOAT,DT_FLOAT16,DT_BF16}))
  .INPUT(gamma, TensorType({DT_FLOAT,DT_FLOAT16,DT_BF16}))
  .OUTPUT(mean, TensorType({DT_FLOAT,DT_FLOAT,DT_FLOAT}))
  .OUTPUT(rstd, TensorType({DT_FLOAT,DT_FLOAT,DT_FLOAT}))
  .OUTPUT(y, TensorType({DT_FLOAT,DT_FLOAT16,DT_BF16}))
  .ATTR(alpha, Float, 0.3f)
  .ATTR(epsilon, Float, 1e-06f)
  .OP_END_FACTORY_REG(DeepNorm);

/**
 * @brief DeepNormGrad Operator.
 *
 * @par Inputs
 * @li dy: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
 * @li x: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
 * @li gx: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
 * @li gamma: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
 * @li mean: A Tensor. Support dtype: [float32], support format: [ND].
 * @li rstd: A Tensor. Support dtype: [float32], support format: [ND].

 * @par Attributes:
 * alpha: An optional attribute, the type is float. Defaults to 0.3.

 * @par Outputs
 * @li dx: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
 * @li dgx: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
 * @li dbeta: A Tensor. Support dtype: [float32], support format: [ND].
 * @li dgamma: A Tensor. Support dtype: [float32], support format: [ND].
 */
REG_OP(DeepNormGrad)
    .INPUT(dy, TensorType({DT_FLOAT,DT_FLOAT16,DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT,DT_FLOAT16,DT_BF16}))
    .INPUT(gx, TensorType({DT_FLOAT,DT_FLOAT16,DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT,DT_FLOAT16,DT_BF16}))
    .INPUT(mean, TensorType::({DT_FLOAT}))
    .INPUT(rstd, TensorType::({DT_FLOAT}))
    .OUTPUT(dx, TensorType({DT_FLOAT,DT_FLOAT16,DT_BF16}))
    .OUTPUT(dgx, TensorType({DT_FLOAT,DT_FLOAT16,DT_BF16}))
    .OUTPUT(dbeta, TensorType({DT_FLOAT}))
    .OUTPUT(dgamma, TensorType({DT_FLOAT}))
    .ATTR(alpha, Float, 0.3f)
    .OP_END_FACTORY_REG(DeepNormGrad);

/**
* @brief Function dropout with softmaxgrad and muls

* @par Inputs:
* Two inputs, including:
* @li y_grad: A mutable Tensor. The type support float16 float32.
* @li mask: A mutable Tensor. Must met all of the following rules:
*     shape of mask should be 1D.
*     dtype of mask should be uint8.
*     value of shape should met the following algorithm:
*     value = (size(x) + 128 - 1) // 128 * 128
* @li softmax_output: A mutable Tensor. Must met all of the following rules:
*     shape of softmax_output should be NZ.
*     dtype of softmax_output should be float16 float32.
*     it is the output of softmax

* @par Attributes:
* @li input_keep_prob:A attribute used to judge which units should be keep.
*     Has the same type as "x" . \n
* @li alpha: A attribute used to scale tensor.
*     Has the same type as "x" . \n
* @li axes: A list of int. The dimension softmax would be performed on. Defaults
*     to "[-1]" . \n

* @par Outputs:
* x_grad: A mutable Tensor. Has the same type as "x". \n

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(DropoutWithMulsAndSoftmaxGrad)
    .INPUT(y_grad, TensorType({ DT_FLOAT16, DT_FLOAT }))
    .INPUT(mask, TensorType({ DT_UINT8 }))
    .INPUT(softmax_output, TensorType({ DT_FLOAT16, DT_FLOAT }))
    .OUTPUT(x_grad, TensorType({ DT_FLOAT16, DT_FLOAT }))
    .REQUIRED_ATTR(input_keep_prob, Float)
    .REQUIRED_ATTR(alpha, Float)
    .ATTR(axes, ListInt, { -1 })
    .OP_END_FACTORY_REG(DropoutWithMulsAndSoftmaxGrad)

/**
* @brief Loss function that measures the softmax cross entropy.

* @par Inputs:
* Three inputs, including:
* @li scores: A Tensor. Must be one of the following types: float16, bfloat16, float32, double.
* A "batch_size * num_classes" matrix.
* @li labels: A Tensor. Must be one of the following types: "int32", "int64".
* @li weights: A manual rescaling weight given to each class, the same dtype with scores.
* If given, it has to be a 1D Tensor assigning weight to each of the classes.
* Otherwise, it is treated as if having all ones. \n

* @par Attributes:
* ignore_index:Specifies a target value that is ignored and does not contribute to the input gradient.
* It's an optional value, Defaults to 0. \n
* reduction: A character string from "none", "mean", and "sum", specifying the gradient output mode. Defaults to "mean" . \n

* @par Outputs:
* @li loss: A Tensor for per example loss (a "batch_size" vector). Has the same type as "scores".
* @li log_prop: A Tensor. Has the same type as "scores" . \n

* @par Third-party framework compatibility
* Compatible with the ONNX operator SoftmaxCrossEntropyLoss.
*/
REG_OP(SoftmaxCrossEntropyLoss)
    .INPUT(scores, TensorType({DT_DOUBLE,DT_FLOAT16,DT_FLOAT,DT_BFLOAT16}))
    .INPUT(labels, TensorType({DT_INT32, DT_INT64}))
    .OPTIONAL_INPUT(weights, TensorType({DT_DOUBLE,DT_FLOAT16,DT_FLOAT,DT_BFLOAT16}))
    .ATTR(ignore_index, Int, 0)
    .ATTR(reduction, String, "mean")
    .OUTPUT(loss, TensorType({DT_DOUBLE,DT_FLOAT16,DT_FLOAT,DT_BFLOAT16}))
    .OUTPUT(log_prop, TensorType({DT_DOUBLE,DT_FLOAT16,DT_FLOAT,DT_BFLOAT16}))
    .OP_END_FACTORY_REG(SoftmaxCrossEntropyLoss)

/**
*@brief This criterion computes the cross entropy loss between input logits and target.

* @par Inputs:
* Three inputs, including:
* @li input: A ND tensor of type float16 or float32 or bfloat16, 
* Shape only support 2D [N, C], where N is batch size, C is class.
* @li target: A ND tensor of type int32 or int64, specifying the target value.
* Shape is [N].
* @li weight: An optional ND tensor of type float32, specifying the weight value.
* Shape is [C].

* @par Attributes:
* @li reduction: An optional string attr from ["none", "mean", "sum"],
*  specifying the reduction type to be applied to the output. Defaults to "mean".
* @li ignore_index: An optional int attr. Specifies a target value that is ignored and does not contribute to the input gradient.
* Note that ignore_index is only applicable when the target contains class indices. Defaults to "-100".
* @li label_smoothing: An optional float attr in [0.0, 1.0]. Specifies the amount of smoothing when computing the loss,
* where 0.0 means no smoothing. The targets become a mixture of the original ground truth and a uniform distribution as
* described in Rethinking the Inception Architecture for Computer Vision. Default: 0.0
* @li lse_square_scale_for_zloss: An optional float attr, Default: 0.0, 
* If is 0 and return_zloss is true, output all 0 tensors.
* @li return_zloss: An optional bool attr. Default: false, Control whether zloss outputs, when true zloss outputs, 
* when false zloss dose not output.

* @par Outputs:
* @li loss: A ND tensor, Has the same dtype as input, Specifying the cross entropy loss, 
* When reduction is "none", shape is [N], When reduction is "sum" or "mean"， shape is [1].
* @li log_prob: A ND tensor, Has the same dtype and shape as input,  Specifying the result of logsoftmax.
* @li zloss: A ND tensor, Has the same dtype as input. shape is [N], Benchmarking Triton kernel implementation.
* @li lse_for_zloss: A ND tensor, Has the same dtype as input. shape is [N], Benchmarking Triton kernel implementation,
* It is passed to the intermediate variable lse for reverse calculation.

* @par Third-party framework compatibility
* Compatible with PyTorch operator CrossEntropyLoss.
*/
REG_OP(CrossEntropyLoss)
    .INPUT(input, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(target, TensorType({DT_INT64, DT_INT32}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT}))
    .OUTPUT(loss, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(log_prob, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(zloss, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(lse_for_zloss, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(reduction, String, "mean")
    .ATTR(ignore_index, Int, -100)
    .ATTR(label_smoothing, Float, 0.0)
    .ATTR(lse_square_scale_for_zloss, Float, 0.0)
    .ATTR(return_zloss, Bool, false)
    .OP_END_FACTORY_REG(CrossEntropyLoss)

/**
* @brief Function axpy with softmax and dropoutdomask . \n

* @par Inputs:
* Three inputs, including:
* @li x1: A mutable tensor. The type supports float16 and float32.
* @li x2: A mutable tensor. The type supports float16 and float32. Has the same type and shape as "x1".
* @li mask: A mutable tensor. The type supports uint8. Has the same shape as "x1". \n

* @par Attributes:
* @li alpha: A attribute used to scale tensor. The type is float . \n
* @li input_keep_prob: A attribute used to judge which units should be keep.
*     The type is float . \n
* @li axis: A list of int. The dimension softmax would be performed on. Defaults
*     to "[-1]" . \n

* @par Outputs:
* @li y1: A mutable tensor. Has the same type as "x1". \n
* @li y2: A mutable tensor. Has the same type as "x1". \n

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(AxpyWithSoftmaxAndDropOutDoMask)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(mask, TensorType({DT_UINT8}))
    .OUTPUT(y1, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y2, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(alpha, Float)
    .REQUIRED_ATTR(input_keep_prob, Float)
    .ATTR(axis, ListInt, {-1})
    .OP_END_FACTORY_REG(AxpyWithSoftmaxAndDropOutDoMask)

/**
* @brief MMCV Function: sigmoid_focal_loss_grad  .

* @par Inputs:
* Three inputs, including:
* @li pred: the predicted tensor. The type support float16 and float32.
* @li target: the target label Tensor. The type support Int32.
* @li dout: the grad of previous op grad, which has the sampe shape wth pred. The type support float16 and float32.
* @li weight: A optional input Tensor, default is None, which helps to calculate the loss by supplying sample weights:
*     shape of pred should be (B, D), B means batch size, D means the number of labels.
*     shape of target should be (D, ).
*     shape of weight should be (D, ).
      The type support float16 and float32. \n

* @par Attributes:
* @li alpha: A attribute is used to reweight the sample. The type is float . \n
* @li gamma: A attribute is used to calculate the power of the probability.
*     The type is float . \n
* @li reduction: a type of the reduce method. default is 'mean', which means computing the average loss.
                'sum' means computing the sum of the loss, 'none' means no reducing .\n

* @par Outputs:
* grad: A mutable Tensor. Has the same type and shape as "pred". \n

* @par Third-party framework compatibility
* Compatible with the MMCV operator SigmoidFocalLoss.
*/
REG_OP(SigmoidFocalLossGrad)
    .INPUT(pred, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(target, TensorType({DT_INT32}))
    .INPUT(dout, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(grad, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(alpha, Float, 0.25)
    .ATTR(gamma, Float, 2.0)
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(SigmoidFocalLossGrad)

/**
* @brief Fused Operator of Add and LayerNorm. \n

* @par Inputs
* @li x1: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li x2: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li gamma: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li beta: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li bias: A optional input Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].

* @par Attributes
* @li epsilon: A optional attribute, the type is float. Defaults to 1e-5.
* @li additional_output: A optional attribute, the type is bool. Defaults to false.

* @par Outputs
* @li y: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li mean: A Tensor. Support dtype: [float32], support format: [ND].
* @li rstd: A Tensor. Support dtype: [float32], support format: [ND].
* @li x: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
*/
REG_OP(AddLayerNorm)
    .INPUT(x1, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(x2, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(gamma, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(beta, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(bias, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(y, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(mean, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(rstd, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(x, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .ATTR(epsilon, Float, 1e-5f)
    .ATTR(additional_output, Bool, false)
    .OP_END_FACTORY_REG(AddLayerNorm)

/**
* @brief Batch normalization (also known as batch norm) is a method used to
* make training of artificial neural networks faster and more stable
* through normalization of the layers' inputs by re-centering and re-scaling.

* @par Inputs:
* @li x: A 4D/5D tensor of type float16/bfloat16/float32, with format NCHW/NHWC/NCDHW/NDHWC. describing the feature_map.
* @li weight: A 1D tensor with the shape is same as dim C of input x, support dtypes are related to input x dtype,
* the following combinations are supported: [x: float16, weight: float16/float32], [x: bfloat16, weight: bfloat16/float32], [x: float32, weight: float32].
* describing the weight.
* @li bias: A 1D tensor of the same dtype and shape as input weight, describing the bias.
* @li running_mean: A 1D tensor of type float16/bfloat16/float32, the shape is same as dim C of input x, 
* the following combinations are supported: [x: float16, running_mean: float16/float32], [x: bfloat16, running_mean: bfloat16/float32], [x: float32, running_mean: float32].
* describing the running mean.
* @li running_var: A 1D tensor of the same dtype and shape as input running_mean, describing the running var. \n

* @par Attributes:
* @li epsilon: An optional float32, small value added to variance to avoid dividing by zero. Defaults to "1e-5".
* @li momentum: An optional float32, the value used for the running_mean and running_var computation. Defaults to "0.1".
* @li is_training: An optional bool, specifying if the operation is used for training or inference. Defaults to "True", now only support "True". \n

* @par Outputs:
* @li y: A 4D/5D tensor of the same dtype, shape and format as input x, describing the result.
* @li running_mean: A 1D tensor of the same dtype and shape as input running_mean, describing the running mean.
* @li running_var: A 1D tensor of the same dtype and shape as input running_var, describing the running var.
* @li save_mean: A 1D tensor of type float32, describing the mean of "x".
* @li save_rstd: A 1D tensor of type float32, describing the rstd of "x". \n
*/
REG_OP(BatchNormV3)
    .INPUT(x, "T1")
    .INPUT(weight, "T2")
    .INPUT(bias, "T2")
    .INPUT(running_mean, "T4")
    .INPUT(running_var, "T4")
    .OUTPUT(y, "T1")
    .OUTPUT(running_mean, "T4")
    .OUTPUT(running_var, "T4")
    .OUTPUT(save_mean, "T3")
    .OUTPUT(save_rstd, "T3")
    .ATTR(epsilon, Float, 1e-5f)
    .ATTR(momentum, Float, 0.1f)
    .ATTR(is_training, Bool, true)
    .DATATYPE(T1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T3, TensorType({DT_FLOAT}))
    .DATATYPE(T4, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(BatchNormV3)

/**
* @brief AddLayerNormGrad operator interface implementation.

* @par Inputs
* @li dy: Main grad input.
*     A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li x1: Input x1 of the forward fused operator.
*     A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li x2: Input x2 of the forward fused operator.
*     A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li rstd: Rstd of the sum of forward inputs x1 and x2.
*     A Tensor. Support dtype: [float32], support format: [ND].
* @li mean: Mean of the sum of forward inputs x1 and x2.
*     A Tensor. Support dtype: [float32], support format: [ND].
* @li gamma: Describing the weight.
*     A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li dsum: Other grad input.
*      A optional input Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].

* @par Outputs
* @li dx: The gradient of input "x", Has the same type and shape as "x".
*     A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li dgamma: The gradient of input "gamma", Has the same type and shape as "gamma".
*     A Tensor. Support dtype: [float32], support format: [ND].
* @li dbeta: The gradient of input "beta", Has the same type and shape as "beta".
*     A Tensor. Support dtype: [float32], support format: [ND].
*/
REG_OP(AddLayerNormGrad)
    .INPUT(dy, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(x1, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(x2, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(rstd, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .INPUT(mean, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .INPUT(gamma, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(dsum, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(dx, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(dgamma, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(dbeta, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OP_END_FACTORY_REG(AddLayerNormGrad);

/**
* @brief MMCV Function: softmax_focal_loss_grad.

* @par Inputs:
* Three inputs, including:
* @li pred: the predicted tensor. The type support float16 and float32.
* @li target: the target label Tensor. The type support Int32.
* @li dout: the grad of previous op grad, which has the sampe shape wth pred. The type support float16 and float32.
* @li weight: A optional input Tensor, default is None, which helps to calculate the loss by supplying sample weights.
            The type support float16 and float32.
*     shape of pred should be (B, D), B means batch size, D means the number of labels.
*     shape of target should be (B, D).
*     shape of weight should be (D, ) \n

* @par Attributes:
* @li alpha: A attribute is used to reweight the sample. The type is float . Default is 0.25\n
* @li gamma: A attribute is used to calculate the power of the probability.
*     The type is float . Default is 2.0\n
* @li reduction: a type of the reduce method. default is 'mean', which means computing the average loss.
                'sum' means computing the sum of the loss, 'none' means no reducing. Default is 'mean' \n

* @par Outputs:
* grad: A mutable Tensor. Has the same type and shape as "pred". \n

* @par Third-party framework compatibility
* Compatible with the MMCV operator SoftmaxFocalLossGrad.
*/
REG_OP(SoftmaxFocalLossGrad)
    .INPUT(pred, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(target, TensorType({DT_INT32}))
    .INPUT(dout, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(grad, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(alpha, Float, 0.25)
    .ATTR(gamma, Float, 2.0)
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(SoftmaxFocalLossGrad)

/**
* @brief RmsNormGrad operator interface implementation.

* @par Inputs
* Four inputs, including:
* @li dy: The gradient returned backward.
*     A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li x: The input of the forward operator.
*     A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li rstd: The intermediate computation result of the forward operator.
*     A Tensor. Support dtype: [float32], support format: [ND].
* @li gamma: The input of the forward operator.
*     A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].

* @par Outputs
* @li dx: The gradient of input "x", Has the same type and shape as "x".
*     A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li dgamma: The gradient of input "gamma". Has the same type and shape as "gamma".
*     A Tensor. Support dtype: [float32], support format: [ND].
*/
REG_OP(RmsNormGrad)
    .INPUT(dy, TensorType({DT_FLOAT,DT_FLOAT16,DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT,DT_FLOAT16,DT_BF16}))
    .INPUT(rstd, TensorType({DT_FLOAT,DT_FLOAT,DT_FLOAT}))
    .INPUT(gamma, TensorType({DT_FLOAT,DT_FLOAT16,DT_BF16}))
    .OUTPUT(dx, TensorType({DT_FLOAT,DT_FLOAT16,DT_BF16}))
    .OUTPUT(dgamma, TensorType({DT_FLOAT,DT_FLOAT,DT_FLOAT}))
    .OP_END_FACTORY_REG(RmsNormGrad)
}  // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_NN_NORM_OPS_H_WARN_SHOWN
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_NORM_OPS_H_
