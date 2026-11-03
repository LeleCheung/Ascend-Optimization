#ifndef OP_NN_PROTO_H_
#define OP_NN_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge{

/**
* @brief Multiplies matrix "x1" by matrix "x2", producing "x1 @ x2".
* @par Inputs:
* Four inputs, including:
* @li x1: A matrix Tensor. Must be one of the following types: float16,
* float32, bfloat16. 2D-6D. Has format [ND, NHWC, NCHW].
* @li x2: A matrix Tensor. Must be one of the following types: float16,
* float32, bfloat16. 2D-6D. Has format [ND, NHWC, NCHW].
* @li bias: A optional Tensor. Must be one of the following types: float16,
* float32, bfloat16. Has format [ND, NHWC, NCHW],
* bfloat16 is only supported in Ascend 950 AI processor
* @li offset_w: A optional Tensor. Must be one of the following types:
* int8, int4. Has format [ND, NHWC, NCHW].

* @par Attributes:
* @li adj_x1: A bool. If True, changes the shape of "x1" from [B, M, K] to
* [B, K, M] before multiplication.
* @li adj_x2: A bool. If True, changes the shape of "x2" from [B, K, N] to
* [B, N, K] before multiplication.
* @li offset_x: An optional integer for quantized BatchMatMulV3.
* @li enable_hf32: An optional bool for BatchMatMulV3. If True, enable enable_hi_float_32_execution
* before multiplication. \n

* @par Outputs:
* y: The result matrix Tensor. Must be one of the following types: float16,
* float32, bfloat16. 2D-6D. Has format [ND, NHWC, NCHW]. BatchMatMulV3 supports broadcasting in the batch dimensions.
*/

REG_OP(BatchMatMulV3)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8, DT_INT4}))
    .ATTR(adj_x1, Bool, false)
    .ATTR(adj_x2, Bool, false)
    .ATTR(offset_x, Int, 0)
    .ATTR(enable_hf32, Bool, false)
    .OP_END_FACTORY_REG(BatchMatMulV3)

/**
* @brief The fusion operator of antiquant function and matmul.

* @par Inputs:
* @li x1: A matrix tensor. Format supports ND. The type supports DT_FLOAT4_E2M1. Shape is [M, K].
* @li x2: A matrix tensor. Format supports NZ. The type supports DT_FLOAT4_E2M1. Shape is [N, K].
* @li x1_level0_scale: A matrix tensor. Format supports ND. The type supports DT_FLOAT.
* First-level quantization parameter for x1. Shape is [M, Ceil(K/level0_group_size)].
* @li x1_level1_scale: A matrix tensor. Format supports ND. The type supports DT_FLOAT8_E8M0.
* Second-level quantization parameter for x1. Shape is [M, Ceil(K/(2*level1_group_size)), 2].
* @li x2_level0_scale: A matrix tensor. Format supports ND. The type supports DT_FLOAT.
* First-level quantization parameter for x2. Shape is [Ceil(K/level0_group_size), N].
* @li x2_level1_scale: A matrix tensor. Format supports ND. The type support DT_FLOAT8_E8M0.
* Second-level quantization parameter for x2. Shape is [N, Ceil(K/(2*level1_group_size)), 2].
* @li bias: An Optional tensor. Shape support (N). Format support ND. The type support DT_FLOAT.

* @par Attributes:
* @li dtype: An int. Declare the output dtype, supports 1(float16), 27(bfloat16).
* @li transpose_x1: A bool. If True, changes the shape of "x1" from [M, K] to [K, M]. Now only support false.
* Default value is false.
* @li transpose_x2: A bool. If True, changes the shape of "x2" from [K, N] to [N, K] and
* "x2_level1_scale" from [Ceil(K/(2*level1_group_size)), N, 2] to [N, Ceil(K/(2*level1_group_size)), 2]. 
* Now only support true. Default value is true.
* @li level0_group_size: An int. First-level quantization parameter. Size supports 512.
* Default value is 512.
* @li level1_group_size: An int. Second-level quantization parameter. Size supports 32.
* Default value is 32.

* @par Outputs:
* y: A matrix tensor. The format support ND. The type support float16, bfloat16.
*/
REG_OP(DualLevelQuantBatchMatmul)
    .INPUT(x1, TensorType({DT_FLOAT4_E2M1}))
    .INPUT(x2, TensorType({DT_FLOAT4_E2M1}))
    .INPUT(x1_level0_scale, TensorType({DT_FLOAT}))
    .INPUT(x1_level1_scale, TensorType({DT_FLOAT8_E8M0}))
    .INPUT(x2_level0_scale, TensorType({DT_FLOAT}))
    .INPUT(x2_level1_scale, TensorType({DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(dtype, Int)
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, true)
    .ATTR(level0_group_size, Int, 512)
    .ATTR(level1_group_size, Int, 32)
    .OP_END_FACTORY_REG(DualLevelQuantBatchMatmul)

/**
 * @brief Multiplies matrix "a" by matrix "b", producing "a @ b".
 * @par Inputs:
 * Four inputs, including:
 * @li x1: A matrix tensor. Must be one of the following types: float16, bfloat16, float32.
 * float32 is supported for the the following fused_op_types: "","relu","add","mul".
 * @li x2: A matrix tensor. Must be one of the following types: float16, bfloat16, float32.
 * float32 is supported for the the following fused_op_types: "","relu","add","mul".
 * @li bias: An optional tensor. 1D. Must be one of the following types: float16, bfloat16, float32.
 * bfloat16 is only supported in Ascend 950 AI processor
 * Must be one of the following: "","relu","add","mul".
 * @li x3: An Optional tensor. Must be one of the following types: float16, bfloat16, float32.
 * Must be one of the following: "add",mul".
 *
 * @par Attributes:
 * @li transpose_x1: A bool. If True, changes the shape of "x1" from [M, K] to
 * [K, M] before multiplication.
 * @li transpose_x2: A bool. If True, changes the shape of "x2" from [K, N] to
 * [N, K] before multiplication.
 * @li enable_hf32: A bool. This input is supported for the the following fused_op_types: "","relu","add","mul".
 * @li fused_op_type: A string. The fused_op_type include "","add","mul","gelu_erf","gelu_tanh","relu".
 * Default type is defined as "".
 * @par Outputs:
 * y: The result matrix tensor. Must be one of the following types: float16, bfloat16, float32,
 */
REG_OP(FusedMatMul)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(x3, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(enable_hf32, Bool, false)
    .ATTR(fused_op_type, String, "")
    .OP_END_FACTORY_REG(FusedMatMul)

/**
* @brief FusedQuantMatMul computes a quant matmul + vector operation fused operator.

* @par Inputs:
* @li x1: A matrix Tensor. The shape supports (m, k), and the format supports ND.
* The data type supports int8/int4. The m and k value must be in [1, 65535].
* @li x2: A matrix Tensor of quantized weight. The shape supports (k, n) for ND format if fused_op_type is not "swiglu".
* only supports (2, n1, k1, k0, n0) for NZ format, and ori shape should be (2, k, n) if fused_op_type is "swiglu".
* If fused_op_type is "gelu_erf" or "gelu_tanh", the format supports ND/NZ, the dtype supports int4/int8.
* The format only supports ND in a4w4 scenario if fused_op_type is "gelu_erf" or "gelu_tanh".
* The data type supports int8/int4. The k, n value must be in [1, 65535], k0 should be 16, n0 should be 64.
* @li bias: An Optional Tensor.
* The shape supports (1,), (n,) or (batch, 1, n), format supports ND, the type supports int32, bfloat16, float16 and float32, swiglu type is not supported yet.
* @li x1_scale: An Optional Tensor for quantization parameters. 
* The type supports float32, format supports ND if fused_op_type is "gelu_erf" or "gelu_tanh".
* @li x2_scale: An Optional Tensor for quantization parameters.
* The type supports uint64, int64, float32, bfloat16, format supports ND.
* The type only supports float32, bfloat16 if fused_op_type is "gelu_erf" or "gelu_tanh".
* The shape supprts(1), (n) if fused_op_type is not "swiglu".
* The shape supports (2, n) or (2 * n) if fused_op_type is "swiglu".
* The shape supports (n) or (1) if fused_op_type is "gelu_tanh" or "gelu_erf".
* @li y_scale:  An Optional Tensor for quantization parameters. It's not supported yet.
* @li x1_offset: An Optional Tensor for quantization parameters. It's not supported yet.
* @li x2_offset: An Optional Tensor for quantization parameters. It's not supported yet.
* @li y_offset: An Optional Tensor for quantization parameters. It's not supported yet.
* @li x2_table: An Optional Tensor for quantization parameters. It's not supported yet.
* @li x3: An Optional tensor.
* The data type supports float only.
* The shape supports (n,), only supports fused_op_type is "swiglu" and output type is int8 for quant scale fp16 to int8
*
* @par Attributes:
* @li dtype: An int32. The data type of y. It supports 1(float16), 2(int8) and 27(bfloat16).
* @li compute_type: An int32. It's compute type. It's not supported yet.
* @li transpose_x1: A bool. x1 is transposed if true. Default: false.
* When transpose_x1 is true, x1's shape is (k, m). Currently, it should always be false.
* @li transpose_x2: A bool. x2 is transposed if true. Default: false.
* When transpose_x2 is true, x2's shape is (n, k). Currently, it should always be false.
* @li group_size: An int32. It's not supported yet.

* @li fused_op_type: A string. The fused_op_type include "add","mul","gelu_erf","gelu_tanh","relu","swiglu".
* Currently, only "relu", "swiglu", "gelu_tanh" and "gelu_erf" is supported.
* Default type is defined as "".
* "relu":
* out = relu(matmul(x1, x2) + bias) * x2_scale.
* "swiglu":
* out = silu(matmul(x1, x2[0]) * x2_scale[0]) * (matmul(x1, x2[1]) * x2_scale[1]) * x3.
* "gelu_tanh":
* out = gelu_tanh(qbmmout).
* "gelu_erf":
* out = gelu_erf(qbmmout).

* @par Outputs:
* y: A matrix Tensor. The data type supports int8, float16, bfloat16. The format supports ND.
* The type only supports float16, bfloat16 if fused_op_type is "gelu_erf" or "gelu_tanh".
*/

REG_OP(FusedQuantMatMul)
    .INPUT(x1, TensorType({DT_INT8, DT_INT4}))
    .INPUT(x2, TensorType({DT_INT8, DT_INT4}))
    .OPTIONAL_INPUT(bias, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT32, DT_INT32}))
    .OPTIONAL_INPUT(x1_scale, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(x2_scale, TensorType({DT_BF16, DT_UINT64, DT_INT64, DT_FLOAT32}))
    .OPTIONAL_INPUT(y_scale, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(x1_offset, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(x2_offset, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(y_offset, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(x2_table, TensorType({DT_INT8}))
    .OPTIONAL_INPUT(x3, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .REQUIRED_ATTR(dtype, Int)
    .ATTR(compute_type, Int, -1)
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(group_size, Int, -1)
    .ATTR(fused_op_type, String, "")
    .OP_END_FACTORY_REG(FusedQuantMatMul)

/**
* @brief GemmV2 matrix "a" by matrix "b" and add matrix "c", producing "alpha * op(a) @ op(b) + beta * op(c)". \n
* @par Inputs:
* Four inputs, including:
* @li a: A matrix Tensor. 2D. Must be one of the following types: bfloat16,
* float16. Has format [ND].
* @li b: A matrix Tensor. 2D. Must be one of the following types: bfloat16,
* float16. Has format [ND].
* @li alpha: A 1D Tensor. Must be one of the following types: bfloat16,
* float16. Has format [ND]. \n
* @li beta: A 1D Tensor. Must be one of the following types: bfloat16,
* float16. Has format [ND]. \n
* @li c: A matrix Tensor. 2D. Must be one of the following types: bfloat16,
* float16. Has format [ND].

* @par Attributes:
* @li transpose_a: A bool. If True, changes the shape of "a" from [M, K] to
* [K, M] before multiplication.
* @li transpose_b: A bool. If True, changes the shape of "b" from [K, N] to
* [N, K] before multiplication. \n

* @par Outputs:
* c: The result matrix Tensor. 2D. Must be one of the following types: bfloat16,
* float16. Has format [ND]. \n
*/
REG_OP(GemmV2)
    .INPUT(a, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(b, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(alpha, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(beta, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(c, TensorType({DT_FLOAT}))
    .OUTPUT(c, TensorType({DT_FLOAT}))
    .ATTR(transpose_a, Bool, false)
    .ATTR(transpose_b, Bool, false)
    .OP_END_FACTORY_REG(GemmV2)

/**
* @brief Multiplies matrix "x1" by matrix "x2", producing "x1 @ x2". \n
* @par Inputs:
* Four inputs, including:
* @li x1: A matrix Tensor. 2D. Must be one of the following types: bfloat16,
* float16, float32. Has format [ND, NHWC, NCHW].
* @li x2: A matrix Tensor. 2D/4D(NZ格式). Must be one of the following types: bfloat16,
* float16, float32. Has format [ND, NHWC, NCHW, NZ].
* @li bias: A 1D Tensor. Must be one of the following types: float32,
* float16, float32. Has format [ND, NHWC, NCHW]. \n
* bfloat16 is only supported in Ascend 950 AI processor
* @li offset_w: An optional matrix Tensor. 2D. Must be one of the following
* types: int8, int4.

* @par Attributes:
* @li transpose_x1: A bool. If True, changes the shape of "x1" from [M, K] to
* [K, M] before multiplication.
* @li transpose_x2: A bool. If True, changes the shape of "x2" from [K, N] to
* [N, K] before multiplication.
* @li offset_x: An optional integer for quantized MatMulV2Compress.
* The negative offset added to the input x1 for int8 type. Ensure offset_x
* within the effective range of int8 [-128, 127]. Defaults to "0".
* @li opImplMode: An optional integer for MatMulV3. op_impl_mode_enum: 0x1: default 
* 0x2: high_performance 0x4: high_precision 0x8: super_performance
* 0x10: support_of_bound_index 0x20: enable_float_32_execution 0x40: enable_hi_float_32_execution
* before multiplication. \n

* @par Outputs:
* y: The result matrix Tensor. 2D. Must be one of the following types: bfloat16,
* float16, float32. Has format [ND, NHWC, NCHW]. \n
*/
REG_OP(MatMulV3)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8, DT_INT4}))
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(offset_x, Int, 0)
    .ATTR(opImplMode, Int, 0x1)
    .OP_END_FACTORY_REG(MatMulV3)

/**
* @brief Quant Batch Matmul Calculation.

* @par Inputs:
* Six inputs, including:
* @li x1: A matrix tensor. Must be one of the following types: int8, int4, hifloat8, float8_e5m2, float8_e4m3fn, float4_e2m1. \n
          when the data type is int8, supports ND and NZ formats. \n
          when the data type is int4, hifloat8, float8_e5m2, float8_e4m3fn, float4_e2m1, only supports ND format. \n
          - In ND format and Non-int4 type, the shape ranges from 2D to 6D. When transpose_x1 is false, the shape is (batch,m,k), where
          batch is optional; In int4 type, shape only supports 2D. \n
          - In NZ (Ascend affinity) format, the shape ranges from 4D to 8D. When tranpose_x1 is true, the shape is
          (batch,x1_m1,x1_k1,x1_k0,x1_m0), where batch is optional, x1_m0 = 32, and x1_k0 = 16. When transpose_x1 is false, the shape is
          (batch,x1_k1,x1_m1,x1_m0,x1_k0), where batch is optional, x1_m0 = 16, and x1_k0 = 32. \n
          When the data type is int4, float4_e2m1, the last dim must be even. \n
* @li x2: A matrix tensor. Must be one of the following types: int8, int4, hifloat8, float8_e5m2, float8_e4m3fn, float4_e2m1. \n
          when the data type is int8, supports ND and NZ formats. \n
          when the data type is int4, hifloat8, float8_e5m2, float8_e4m3fn, float4_e2m1, the format only supports ND. \n
          - In ND format and Non-int4 type, the shape ranges from 2D to 6D. When transpose_x2 is false, the shape is (batch,k,n), where
          batch is optional; In int4 type, shape only supports 2D. \n
          - In NZ (Ascend affinity) format, the shape ranges from 4D to 8D. \n
              - When tranpose_x2 is true, the shape is (batch,x2_k1,x2_n1,x2_n0,x2_k0), where batch is optional, x2_k0 = 32, and x2_n0 = 16. \n
              - When transpose_x2 is false, the shape is (batch,x2_n1,x2_k1,x2_k0,x2_n0), where batch is optional, x2_k0 = 16, and x2_n0 = 32. \n
              - When x1 is ND format, k in the shape of x1 and the shape of x2 must meet the following requirement: \n
                    ceil(k / x2_k0) == x2_k1. \n
              - When x1 is NZ format, the shape of x1 and x2 must meet the following requirement: \n
                  - when x1_k0 == x2_k0, x1_k1 == x2_k1, \n
                  - when x1_k0 > x2_k0, ceil((x2_k0 * x2_k1) / x1_k0) == x1_k1, \n
                  - when x1_k0 < x2_k0, ceil((x1_k0 * x1_k1) / x2_k0) == x2_k1. \n
          When the data type is int4, float4_e2m1, the last dim must be even.
* @li scale: A matrix tensor, quantization parameter,
             Must be one of the following types: uint64, float32, int64, bfloat16, float8_e8m0, supports ND format. \n
            - When the data type is bfloat16, uint64 or int64,
             the shape is 1D (t,), with t equal to 1 or n, where n is the same as that of x2. \n
            - When the data type is float8_e8m0,
             the shape is 2D, when the shape of x2 is (n, k), scale is (n, z). Suppose scale_k = ceil(k / 32),
                - When scale_k is an odd number, z = scale_k + 1, \n
                - when scale_k is an even number, z = scale_k. \n
             where 32 is group_size_k(refer to group_size), z must be an even number and z can be equal to (ceil(k / 64) * 2), and k is the reduce axis of x2. \n
            - when the data type is float32,
             the dimension of shape should be 1D or same as that of x2. \n
               - When the quant mode of x2 is perchannel or pertensor, the shape is 1D (t,),
                 with t equal to 1 or n, where n is the same as that of x2. \n
               - When the quant mode of x2 is perblock, the dimension of shape is same as x2, when x2 is (batch, k, n),
                 scale is (batch, ceil(k / 128), ceil(n / 128)), where 128 is group_size_k, group_size_n(refer to group_size). \n
* @li offset: An optional matrix tensor, quantization parameter. Must be one of the following types: float32.
              supports ND format. The shape is 1D (t,), with t equal to 1 or n, where n is the same as that of x2.
* @li bias: An optional matrix tensor. Must be one of the following types: int32, bfloat16, float16, float32, supports ND format.
            The shape is 1D (t,) or 3D (batch, 1, n),
            with t equal to n, where n is the same as that of x2.
* @li pertoken_scale: An optional matrix tensor. The type supports float32, float8_e8m0, supports ND format. \n
                      - When the data type is float8_e8m0, the shape is 2D,
                        when the shape of x1 is (m, k), scale is (m, z), with z = (ceil(k / 64) * 2) which is same to
                        scale, and k is the reduce axis of x1.
                      - When the data type is float32, the dimension of shape should be 1D or same as that of x1. \n,
                        - When the quant mode of x1 is pertoken or pertensor, the shape is 1D (t,),
                          with t equal to 1 or m, where m is the same as that of x1. \n
                        - When the quant mode of x1 is perblock or pergroup, the dimension of shape is same as x1,
                          when x2 is (batch, m, k), for perblock, pertoken_scale is (batch, ceil(m / 128), ceil(k / 128)),
                          for pergroup, pertoken_scale is (batch, m, ceil(k / 128)),
                          where 128 is group_size_m, group_size_k(refer to group_size). \n

* @par Attributes:
* Four attributes, including:
* @li dtype: An Int. Declare the output type, supports 0(float32), 1(float16), 2(int8), 27(bfloat16),
* 34(hifloat8), 36(float8_e4m3fn). Default: 2(int8).
* @li transpose_x1: An optional bool. If true, changes the shape of "x1" from [m, k] to
* [k, m] before multiplication. Default: false.
* @li transpose_x2: An optional bool. If true, changes the shape of "x2" from [k, n] to
* [n, k] before multiplication. Default: false.
* @li group_size: An optional Int. Indicating the ratio between pertoken_scale/scale and x1/x2 in group dequantization.
* If the value of pertoken_scale along the k-dimension is n, one value in pertoken_scale can be used to dequantize n values in x1 along the k-dimension. \n
* The group_size is composed of the group_size_m, group_size_n, and group_size_k, total occupying 48 bits.
* 0-15 bits of group_size indicate group_size_k, 16-31 bits indicate group_size_n, 32-47 bits indicate group_size_m,
* 48-63 bits of group_size are noneffective. \n
* If any of group_size_m, group_size_n, group_size_k calculated by group_size is 0, recalculate it by
* input shape, eg: group_size_m = m / scale_m （m % scale_m must be 0). \n
* Final group_size_m, group_size_n, group_size_k should satisify following requirements: \n
* In mx quantification, the supported final [group_size_m, group_size_n, group_size_k] combinations are [1, 1, 32]. \n
* In pergroup-perblock && perblock-perblock quantification,
* the supported final [group_size_m, group_size_n, group_size_k] combinations are [1, 128, 128] and [128, 128, 128].
* For other input types, the group_size value must be 0. \n

* @par Outputs:
* One output, including:
* y: A matrix Tensor. Must be one of the following types: float16, int8, bfloat16, int32, float32.
     The format supports ND. The shape ranges from 2D to 6D,
     that is (batch, m, n), where batch is optional. Broadcasting can be performed on the batch dimension of x1 and x2.
     The output batch is the same as the batch after broadcasting, m is the same as that of x1, and n is the same as
     that of x2. \n

* @attention Constraints:
* @li The shape of bias should be 1D when the shape of out is 2D, 4D, 5D or 6D, and the shape of bias should be 1D or 3D
* when the out shape is 3D.
* @li The size of the last dimension of x1 and x2 cannot exceed 65535 only on the following computing platforms:
* Atlas A2 Training Series Product/Atlas A2 Inference Series Product and
* Atlas A3 Training Series Product/Atlas A3 Inference Series Product.
* The last dimension of x1 refers to m when transpose_x1 is true or k when transpose_x1 is false. 
* The last dimension of x2 refers to k when transpose_x2 is true or n when transpose_x2 is false.
* @li If input type of x1 and x2 is int4, transpose_x1 should be false, the size of the last dimension of x1 or x2 should
* be an even number.
* @li Input does not support tensor with dimension size 0.
* @li When input type of x1 and x2 is int4, x1 should be ND format.
* @li When y type is int8, x1 should be ND format.
* @li When x2 is ND format, x1 should be ND format.
* @li The following are the supported data type combinations by platform.

* - Atlas Inference Series Product:
*\n
| x1       | x2       | scale        | offset        | bias          | pertoken | out      |
| :------: | :------: | :----------: | :-----------: | :-----------: | :------: | :------: |
| int8     | int8     | uint64/int64 | null          | null/int32    | null     | float16  |
| int8     | int8     | uint64/int64 | null/float32  | null/int32    | null     | int8     |
*\n
* - Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component or
* Atlas A3 Training Series Product/Atlas A3 Inference Series Product:
*\n
| x1       | x2       | scale            | offset        | bias                        | pertoken     | out      |
| :------: | :------: | :--------------: | :-----------: | :-------------------------: | :----------: | :------: |
| int8     | int8     | uint64/int64     | null          | null/int32                  | null         | float16  |
| int8     | int8     | uint64/int64     | null/float32  | null/int32                  | null         | int8     |
| int8     | int8     | float32/bfloat16 | null          | null/int32/bfloat16/float32 | null/float32 | bfloat16 |
| int8     | int8     | float32          | null          | null/int32/float16/float32  | float32      | float16  |
| int4     | int4     | uint64/int64     | null          | null/int32                  | null         | float16  |
| int8     | int8     | float32/bfloat16 | null          | null/int32                  | null         | int32    |
| int4     | int4     | float32/bfloat16 | null          | null/int32/bfloat16/float32 | float32      | bfloat16 |
| int4     | int4     | float32          | null          | null/int32/float16/float32  | float32      | float16  |
*\n
* - Ascend 950 AI Processor:
*\n
| x1                        | x2                        | scale                | offset        | bias                        | pertoken    | out                                    |
| :-----------------------: | :-----------------------: | :------------------: | :-----------: | :-------------------------: | :---------: | :------------------------------------: |
| int8                      | int8                      | uint64/int64         | null          | null/int32                  | null        | float16/bfloat16                       |
| int8                      | int8                      | uint64/int64         | null/float32  | null/int32                  | null        | int8                                   |
| int8                      | int8                      | float32/bfloat16     | null          | null/int32/float32/bfloat16 | null/float32| bfloat16                               |
| int8                      | int8                      | float32              | null          | null/int32/float32/float16  | float32     | float16                                |
| hifloat8                  | hifloat8                  | uint64/int64         | null          | null/float32                | null        | float16/bfloat16/float32               |
| hifloat8                  | hifloat8                  | float32              | null          | null/float32                | float32     | float16/bfloat16/float32               |
| float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2 | uint64/int64         | null          | null/float32                | null        | float16/bfloat16/float32               |
| float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2 | float32              | null          | null/float32                | float32     | float16/bfloat16/float32               |
| float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2 | float8_e8m0          | null          | null/float32                | float8_e8m0 | float16/bfloat16/float32               |
| float4_e2m1               | float4_e2m1               | float8_e8m0          | null          | null/float32                | float8_e8m0 | float16/bfloat16/float32               |
*\n
* - Ascend 950 AI Processor, supported data type and quant mode combinations:
*\n
pertensor-perchannel && pertensor-pertensor:
*\n
  | x1 type                   | x2 type                   | pertoken type    | scale type       |
  | ------------------------- | ------------------------- | ---------------- | ---------------- |
  | int8                      | int8                      | null             | uint64/int64     |
  | int8                      | int8                      | null             | float32/bfloat16 |
  | float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2 | null             | uint64/int64     |
  | float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2 | float32          | float32          |
  | hifloat8                  | hifloat8                  | null             | uint64/int64     |
  | hifloat8                  | hifloat8                  | float32          | float32          |
*\n
pertoken-perchannel && pertoken-pertensor
*\n
  | x1 type                   | x2 type                   | pertoken type    | scale type       |
  | ------------------------- | ------------------------- | ---------------- | ---------------- |
  | int8                      | int8                      | float32          | float32/bfloat16 |
  | float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2 | float32          | float32          |
  | hifloat8                  | hifloat8                  | float32          | float32          |
*\n
pergroup-perblock && perblock-perblock:
*\n
  | x1 type                   | x2 type                   | pertoken type    | scale type       |
  | ------------------------- | ------------------------- | ---------------- | ---------------- |
  | float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2 | float32          | float32          |
  | hifloat8                  | hifloat8                  | float32          | float32          |
*\n
mx quant：
*\n
  | x1 type                   | x2 type                   | pertoken type    | scale type       |
  | ------------------------- | ------------------------- | ---------------- | ---------------- |
  | float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2 | float8_e8m0      | float8_e8m0      |
  | float4_e2m1               | float4_e2m1               | float8_e8m0      | float8_e8m0      |
*\n
* - Ascend 950 AI Processor with group_sizes scenarios, supported data type and shapes combinations:
*\n
| quantization      | x1 type                            | scale type  | x1 shape      | x2 shape      | scale shape                           | pertoken shape                        | group_size      |
|-------------------|------------------------------------|-------------|---------------|---------------|---------------------------------------|---------------------------------------|-----------------|
| perblock-perblock | float8_e4m3fn/float8_e5m2/hifloat8 | float32     | (batch, m, k) | (batch, k, n) | (batch, ceil(k / 128), ceil(n / 128)) | (batch, ceil(m / 128), ceil(k / 128)) | [128, 128, 128] |
| perblock-perblock | float8_e4m3fn/float8_e5m2/hifloat8 | float32     | (batch, m, k) | (batch, n, k) | (batch, ceil(n / 128), ceil(k / 128)) | (batch, ceil(m / 128), ceil(k / 128)) | [128, 128, 128] |
| perblock-perblock | float8_e4m3fn/float8_e5m2/hifloat8 | float32     | (batch, k, m) | (batch, k, n) | (batch, ceil(k / 128), ceil(n / 128)) | (batch, ceil(k / 128), ceil(m / 128)) | [128, 128, 128] |
| perblock-perblock | float8_e4m3fn/float8_e5m2/hifloat8 | float32     | (batch, k, m) | (batch, n, k) | (batch, ceil(n / 128), ceil(k / 128)) | (batch, ceil(k / 128), ceil(m / 128)) | [128, 128, 128] |
| pergroup-perblock | float8_e4m3fn/float8_e5m2/hifloat8 | float32     | (batch, m, k) | (batch, k, n) | (batch, ceil(k / 128), ceil(n / 128)) | (batch, m, ceil(k / 128))             | [1, 128, 128]   |
| pergroup-perblock | float8_e4m3fn/float8_e5m2/hifloat8 | float32     | (batch, m, k) | (batch, n, k) | (batch, ceil(n / 128), ceil(k / 128)) | (batch, m, ceil(k / 128))             | [1, 128, 128]   |
| pergroup-perblock | float8_e4m3fn/float8_e5m2/hifloat8 | float32     | (batch, k, m) | (batch, k, n) | (batch, ceil(k / 128), ceil(n / 128)) | (batch, ceil(k / 128), m)             | [1, 128, 128]   |
| pergroup-perblock | float8_e4m3fn/float8_e5m2/hifloat8 | float32     | (batch, k, m) | (batch, n, k) | (batch, ceil(n / 128), ceil(k / 128)) | (batch, ceil(k / 128), m)             | [1, 128, 128]   |
| mx                | float8_e4m3fn/float8_e5m2          | float8_e8m0 | (batch, m, k) | (batch, n, k) | (n, ceil(k / 64) * 2)                 | (m, ceil(k / 64) * 2)                 | [1, 1, 32]      |
| mx                | float4_e2m1                        | float8_e8m0 | (batch, m, k) | (batch, n, k) | (n, ceil(k / 64) * 2)                 | (m, ceil(k / 64) * 2)                 | [1, 1, 32]      |

*\n
*/
REG_OP(QuantBatchMatmulV3)
    .INPUT(x1, TensorType({DT_INT8, DT_INT4, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1}))
    .INPUT(x2, TensorType({DT_INT8, DT_INT4, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1}))
    .INPUT(scale, TensorType({DT_UINT64, DT_FLOAT, DT_INT64, DT_BF16, DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_INT32, DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(pertoken_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_INT8, DT_BF16, DT_INT32, DT_FLOAT}))
    .REQUIRED_ATTR(dtype, Int)
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(group_size, Int, 0)
    .OP_END_FACTORY_REG(QuantBatchMatmulV3)

/**
* @brief The fusion operator of antiquant and matmul for A8W4, MxA8W4, A8W8, and A4W4.

* @par Inputs:
* @li x1: A matrix Tensor. The shape supports (m, k), and the format supports ND.
* The data type supports float8_e5m2, float8_e4m3fn, int8. The m and k value must be at least 1. The k value
* must be at least 1 and must be a multiple of 64.
* - In A4W4 pergroup scenario: The data type supports int4.
* @li x2: A matrix Tensor of quantized weight. The shape supports (n, k), and the format supports ND/FRACTAL_NZ.\n
* - In MxA8W4 scenario:
* For ND format, the data type supports float4_e2m1. For FRACTAL_NZ format, the data type supports float4_e2m1.\n
* - In A8W4 scenario:
* The data type supports float4_e2m1.
* The k, n value must be at least 1 and k, n value must be a multiple of 64.
* The shape must be (k1, n1, n0, k0) (n0 = 16, k0 = 32) when the format is FRACTAL_NZ.
* - In A8W8 scenario: The data type supports int8.
* - In A4W4 pergroup scenario: The data type supports int4.
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
* @li x2_offset: An Optional Tensor for quantization parameters.
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
    .INPUT(x1, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_INT8, DT_INT4}))
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
* @li x1: A matrix tensor, with data type int8, format ND, 3D shape (batch, m, k).
* @li x2: A matrix tensor, with data type int8, format FRACTAL_NZ, 3D original shape (batch, k, n).
* @li dims: A 1D list or tuple of int64. Specifies the dimensions to reduce.
* @li bias: Reserved parameter, not supported now, please pass nullptr.
* @li x1_scale: A matrix tensor, with data type float32, format ND, 2D shape (b, m).
* @li x2_scale: A tensor, with data type bfloat16, format ND, 1D shape (n,).
* @li y_scale: Reserved parameter, not supported now, please pass nullptr.
* @li x1_offset: Reserved parameter, not supported now, please pass nullptr.
* @li x2_offset: Reserved parameter, not supported now, please pass nullptr.
* @li y_offset: Reserved parameter, not supported now, please pass nullptr.
* @li x2_table: Reserved parameter, not supported now, please pass nullptr. \n

* @par Attributes:
* @li dtype: A Int. Declare the output dtype, supports 27(bfloat16).
* @li compute_type: Reserved attributes, not supported now. Default is -1.
* @li transpose_x1: A bool. If true, changes the shape of "x1" from [m, k] to [k, m] before multiplication.
* Default: false. Only supports false now.
* @li transpose_x2: A bool. If true, changes the shape of "x2" from [k, n] to [n, k] before multiplication.
* Default: false. Only supports false now.
* @li group_size: Reverved attributes, not supported now. Default is -1.
* @li keep_dims: A bool. If true, keeps the original input dims in the output.
* Default: false. Only support false now. \n

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
*@brief Multiplies SparseTensor A (of rank 2) by dense matrix B.

*@par Inputs:
* @li x1_indices: A 2D tensor of type int32 or int64.
*The indices of the matrix "SparseTensor", with size [nnz, 2].
* @li x1_values: A 1D tensor. The values of the SparseTensor, with size [nnz].
* @li x1_shape: A 1D tensor of type int64. The shape of the SparseTensor, with size [2].
* @li x2: A dense matrix tensor of the same type as "x1_values". 2D. \n

*@par Outputs:
*y: A "tensor". Has the same type as "x1_values". \n

*@par Attributes:
*@li adjoint_a: An optional bool. Defaults to "False".Use the adjoint of A in the matrix multiply.
*If A is complex, this is transpose(conj(A)). Otherwise it is transpose(A).
*@li adjoint_b: An optional bool. Defaults to "False".Use the adjoint of B in the matrix multiply.
*If B is complex, this is transpose(conj(B)). Otherwise it is transpose(B). \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator SparseTensorDenseMatMul.
*/
REG_OP(SparseTensorDenseMatMul)
    .INPUT(x1_indices, TensorType({DT_INT32, DT_INT64}))
    .INPUT(x1_values, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_COMPLEX64, DT_COMPLEX128, DT_FLOAT16, DT_INT64}))
    .INPUT(x1_shape, TensorType({DT_INT64}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT64, DT_INT32, DT_COMPLEX64, DT_COMPLEX128, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT64, DT_INT32, DT_COMPLEX64, DT_COMPLEX128, DT_FLOAT16}))
    .ATTR(adjoint_a, Bool, false)
    .ATTR(adjoint_b, Bool, false)
    .OP_END_FACTORY_REG(SparseTensorDenseMatMul)

/**
* @brief The fusion operator of antiquant function and matmul.

* @par Inputs:
* @li x: A matrix tensor. Shape supports (m,k)/(k,m), Format supports ND.
* The type support float16, bfloat16. The m value must be in [1, 2147483647]. The k value
* must be at least 1.
* @li weight: A matrix tensor of quantized weight. Shape supports (n,k)/(k,n),
* Format supports ND/NZ. The type support int8, int4, int32, float8_e4m3fn, hifloat8, float4_e2m1. \n
* Format must be ND when the type is float8_e4m3fn or hifloat8.
* For Ascend 950 AI Processor, transpose_weight must be false when format is NZ. \n
* The k, n value must be at least 1.
* The k value must be even when type is int4/float4_e2m1 and transpose_weight
* is true, and the n value must be even when type is int4/float4_e2m1 and transpose_weight
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
* Additionally, the type supports float8_e8m0 when the weight dtype is float4_e2m1.
* @li antiquant_offset: An Optional tensor for antiquant offset. Shape, format
* and type should be the same with antiquant_scale if antiquant_scale's type is
* not uint64/int64. If antiquant_scale's type is uint64/int64, antiquant_offset should be int32.
* When weight dtype is float8_e4m3fn, hifloat8, float4_e2m1, antiquant_offset is ​excluded from selection.
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
* is bfloat16, the type of bias should be bfloat16. When type of x is bfloat16
* and weight is int4 or int8, the type of bias should be float32. \n
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
    .INPUT(
        weight, TensorType(
                    {DT_INT8, DT_INT4, DT_INT32, DT_FLOAT8_E4M3FN, DT_HIFLOAT8, DT_FLOAT4_E2M1}))
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
* @brief Conv2Dv2 computes a 2D convolution given 4D "x", "filter" and "bias" tensors.
* Like this, output = CONV(x, filter) + bias.
* @par Inputs:
* @li x: A required 4D tensor of input image. With the format "NHWC" which shape is
* [n, h, w, in_channels] or the format "NCHW" which shape is [n, in_channels, h, w].
* @li filter: A required 4D tensor of convolution kernel.
* With the format "HWCN" which shape is [kernel_h, kernel_w, in_channels / groups, out_channels]
* or the format "NCHW" which shape is [out_channels, in_channels / groups, kernel_h, kernel_w].
* @li bias: An optional 1D tensor of additive biases to the outputs.
* The data is stored in the order of: [out_channels].
* @li offset_w: An optional quantitative offset tensor. A tensor of type int8. Reserved.
*\n
* The following are the supported data types and data formats:
*\n
| Tensor    | x        | filter   | bias     | y        |\n
| :-------: | :------: | :------: | :------: | :------: |\n
| Data Type | float16  | float16  | float16  | float16  |\n
|           | bfloat16 | bfloat16 | bfloat16 | bfloat16 |\n
|           | float32  | float32  | float32  | float32  |\n
|           | hifloat8 | hifloat8 | float32  | hifloat8 |\n
| Format    | NCHW     | NCHW     | ND       | NCHW     |\n
|           | NHWC     | HWCN     | ND       | NHWC     |\n
*\n
* @par Attributes:
* @li strides: Required. A list of 4 integers. The stride of the sliding window
* for each dimension of input. The dimension order is determined by the data
* format of "x". The n and in_channels dimensions must be set to 1.
* When the format is "NHWC", its shape is [1, stride_h, stride_w, 1],
* when the format is "NCHW", its shape is [1, 1, stride_h, stride_w].
* @li pads: Optional. A list of 4 integers. The number of pixels to add to each
* (pad_top, pad_bottom, pad_left, pad_right) side of the input. Defaults to [0, 0, 0, 0].
* @li dilations: Optional. A list of 4 integers. The dilation factor for each
* dimension of input. The dimension order is determined by the data format of
* "x". The n and in_channels dimensions must be set to 1.
* When the format is "NHWC", its shape is [1, dilation_h, dilation_w, 1],
* when the format is "NCHW", its shape is [1, 1, dilation_h, dilation_w]. Defaults to [1, 1, 1, 1].
* @li groups: Optional. An integer of type int32. The number of groups
* in group convolution. In_channels and out_channels must both be divisible by "groups". Defaults to 1.
* @li data_format: Optional. It is a string represents input's data format. Defaults to "NCHW". Reserved.
* @li offset_x: Optional. An integer of type int32. It means offset in quantization algorithm
* and is used for filling in pad values. Ensure that the output is within the
* effective range. Defaults to 0. Reserved.
* @li pad_mode: Optional. An string parameter, indicating the mode of
* pad. It must be "SPECIFIC" or "SAME" or "VALID" or "SAME_UPPER" or "SAME_LOWER".
* Defaults to "SPECIFIC".
* @li enable_hf32: Optional. An optional bool parameter. Used to enable hf32 computation.
* If true, enable hf32 computation, otherwise, disable hf32 computation. Defaults to false.
* @par Outputs:
* y: A 4D tensor of output feature map.
* With the format "NHWC" which shape is [n, out_height, out_width, out_channels]
* or the format "NCHW" which shape is [n, out_channels, out_height, out_width].
*\n
*     out_height = (h + pad_top + pad_bottom -
*                   (dilation_h * (kernel_h - 1) + 1))
*                  / stride_h + 1
*\n
*     out_width = (w + pad_left + pad_right -
*                  (dilation_w * (kernel_w - 1) + 1))
*                 / stride_w + 1
*\n
* @attention Constraints:
* @li The following value range restrictions must be met:
*\n
| Name             | Field                | Scope        |\n
| :--------------: | :------------------: | :----------: |\n
| x size           | n                    | [1, 1000000] |\n
|                  | in_channels          | [1, 1000000] |\n
|                  | h                    | [1, 100000]  |\n
|                  | w                    | [1, 4096]    |\n
| filter size      | out_channels         | [1, 1000000] |\n
|                  | in_channels / groups | [1, 1000000] |\n
|                  | kernel_h             | [1, 511]     |\n
|                  | kernel_w             | [1, 511]     |\n
| bias size        | out_channels         | [1, 1000000] |\n
| offset_w size    | out_channels         | [1, 1000000] |\n
| strides          | stride_h             | [1, 63]      |\n
|                  | stride_w             | [1, 63]      |\n
| pads             | pad_top              | [0, 255]     |\n
|                  | pad_bottom           | [0, 255]     |\n
|                  | pad_left             | [0, 255]     |\n
|                  | pad_right            | [0, 255]     |\n
| dilations        | dilation_h           | [1, 255]     |\n
|                  | dilation_w           | [1, 255]     |\n
| groups           | -                    | [1, 65535]   |\n
| data_format      | -                    | ["NHWC", "NCHW"] |\n
| offset_x         | -                    | [-128, 127]  |\n
| pad_mode         | -                    | ["SPECIFIC", "SAME", "VALID", "SAME_UPPER", "SAME_LOWER"] |\n
| enable_hf32      | -                    | [true, false] |\n
*\n
* @li The w dimension of the input image supports cases exceeding 4096, but it may
* cause compilation errors.
*\n
* @li If any dimension of x/filter/bias/offset_w/y shape exceeds max
* 1000000, the product of each dimension of x/filter/bias/offset_w/y
* shape exceeds max int32(2147483647) or the value of
* strides/pads/dilations/groups/data_format/offset_x/pad_mode/enable_hf32
* exceeds the range in the above table, the correctness of the operator cannot be guaranteed.
*\n
*/
REG_OP(Conv2DV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_HIFLOAT8}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_HIFLOAT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_HIFLOAT8}))
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(pads, ListInt, {0, 0, 0, 0})
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NCHW")
    .ATTR(offset_x, Int, 0)
    .ATTR(pad_mode, String, "SPECIFIC")
    .ATTR(enable_hf32, Bool, false)
    .OP_END_FACTORY_REG(Conv2DV2)

/**
*@brief Computes the gradients of convolution3D with respect to the filter
*@par Inputs:
 * @li x: A 5D tensor. Must be one of the following types: float16, float32, bfloat16.
 * The format of the x tensor must be one of the followings:
 * [batch, in_depth, in_height, in_width, in_channels]
 * or [batch, in_channels, in_depth, in_height, in_width]. \n
 * @li filter_size: A 1D tensor of type int32 or int64. The tensor representing the
 * shape of filter, where filter is a 5D tensor.
 * The order of integers in the tensor is determined by filter format,
 * and integers represent the length of each dimension of filter.
 * The axes sequence that can be entered are as follows:
 * [out_channels, in_channels, filter_depth, filter_height, filter_width]
 * or [out_channels, filter_depth, filter_height, filter_width, in_channels].
 * height (H) and width (W) dimensions must be in [1,511]. \n
 * @li out_backprop: A 5D tensor. Must have the same type and format as x.
 * The format of the out_backprop tensor must be one of the followings:
 * [batch, out_depth, out_height, out_width, out_channels] or
 * [batch, out_channels, out_depth, out_height, out_width].
 * Gradients with respect to the "output" of the convolution. \n
*@par Attributes:
 * @li strides: Required. A tuple/list of 5 integers. Specifies the stride
 * of the sliding window for each dimension of feature map.
 * The strides have the same axes sequence as feature map:
 * [batch, stride_depth, stride_height, stride_width, channels] or
 * [batch, channels, stride_depth, stride_height, stride_width].
 * The batch(N) and channels(C) dimensions must be 1.
 * The height (H) and width (W) dimensions must be in [1, 63].
 * The depth (D) dimension must be in [1, 255]. \n
 * @li pads: Required. A tuple/list of 6 integers. Specifies the pads factor of
 * feature map in each directions. Supports only pads along the depth(D),
 * height(H) and width(W) dimensions.
 * The pads sequence is as follows: [front, back, top, bottom, left, right].
 * The height (H), width (W) and depth (D) dimensions must be in [0, 255].
 * Modes "SAME" and "VAILD" padding can be achieved with appropriate values of each direction in pads. \n
 * @li dilations: Optional. A tuple/list of 5 integers. The dilation factor
 * for each dimension of input. Defaults to [1, 1, 1, 1, 1]
 * The dilations have the same axes sequence feature map:
 * [batch, dilation_depth, dilation_height, dilation_width, channels] or
 * [batch, channels, dilation_depth, dilation_height, dilation_width].
 * The batch(N) and channels dimensions must be 1.
 * The width (W), height (H) and depth(D) dimensions must be in [1,255]. \n
 * @li groups: An optional integer within the effective range of [1, 65535]. Default to 1.
 * Number of blocked connections from in_channels to out_channels.
 * Currently in_channels and out_channels must be divisible by groups
 * and x data type must be one of the following types: float16, bfloat16, float32. \n
 * @li data_format: An optional string. The value must be one of ["NDHWC", "NCDHW"]. Defaults to "NDHWC".
 * The correspondence is as follows: batch(N), depth(D), height(H), width(W), channels(C).
 * Specify the data format of the x and out_backprop. \n
 * @li enable_hf32: Optional. An optional bool parameter. Used to enable hf32 computation.
 * If true, enable hf32 computation, otherwise, disable hf32 computation. Defaults to false. \n

*@par Outputs:
 * y: A Tensor that has the type float32. The format is NCDHW or NDHWC or DHWCN.
* @attention Constraints:
 * In Ascend 950PR/Ascend 950DT: The behavior of gradient computation in the padding region depends on the input shape.
 * Depending on the operator optimization strategy, the padding gradients may be directly set to 0.\n
*@par Third-party framework compatibility
 * Compatible with Tensorflow's conv3d_backprop_filter
*/
REG_OP(Conv3DBackpropFilterV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(filter_size, TensorType({DT_INT32, DT_INT64}))
    .INPUT(out_backprop, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NDHWC")
    .ATTR(enable_hf32, Bool, false)
    .OP_END_FACTORY_REG(Conv3DBackpropFilterV2)

/**
*@brief Computes the gradients of convolution3D with respect to the input.
*@par Inputs:
 * @li input_size: A 1D tensor of type int32 or int64.
 * The tensor representing the shape of feature map (the input of the convolution),
 * where feature map is a 5D tensor.
 * The order of integers in the tensor is determined by feature map format,
 * and integers represent the length of each dimension of feature map.
 * The axes sequence that can be entered are as follows:
 * [batch, in_depth, in_height, in_width, in_channels] or
 * [batch, in_channels, in_depth, in_height, in_width].
 * @li filter: A 5D tensor.
 * The format of the filter tensor must be one of the followings:
 * [out_channels, in_channels/groups, filter_depth, filter_height, filter_width] or
 * [out_channels, filter_depth, filter_height, filter_width, in_channels] or
 * [filter_depth, filter_height, filter_width, in_channels, out_channels].
 * In short, the filter tensor support format NCDHW, NDHWC and DHWCN.
 * The NCDHW can be one of the following types: float16, bfloat16, float32.
 * The DHWCN and NDHWC can be one of the following types: float16, bfloat16, float32.
 * kernel_height (H) and kernel_width (W) must have dimension in [1, 511].
 * @li out_backprop: A 5D tensor.
 * The format of the out_backprop tensor must be one of the followings:
 * [batch, out_depth, out_height, out_width, out_channels] or
 * [batch, out_channels, out_depth, out_height, out_width].
 * Gradients with respect to the "output" of the convolution.
*@par Attributes:
 * @li strides: Required. A tuple/list of 5 integers. Specifies the stride of the sliding window
 * for each dimension of feature map. The strides have the same axes sequence as feature map:
 * [batch, stride_depth, stride_height, stride_width, channels] or
 * [batch, channels, stride_depth, stride_height, stride_width].
 * The batch(N) and channels(C) dimensions must be 1.
 * The height (H) and width (W) dimensions must be in [1, 63].
 * The depth (D) dimension must be in [1, 255].
 * @li pads: Required. A tuple/list of 6 integers. Specifies the pads factor of
 * feature map in each directions. Supports only pads along the depth(D),
 * height(H) and width(W) dimensions.
 * The pads sequence is as follows: [front, back, top, bottom, left, right].
 * The height (H), width (W) and depth (D) dimensions must be in [0, 255].
 * Modes "SAME" and "VAILD" padding can be achieved with appropriate values of each direction in pads.
 * @li dilations: Optional. A tuple/list of 5 integers. The dilation factor
 * for each dimension of feature map. Defaults to [1, 1, 1, 1, 1]. \n
 * The dilations have the same axes sequence as feature map:
 * [batch, channels, dilation_depth, dilation_height, dilation_width] or
 * [batch, dilation_depth, dilation_height, dilation_width, channels].
 * The batch(N) and channels dimensions must be 1.
 * The width (W), height (H) and depth(D) dimensions must be in [1, 255].
 * @li groups: An optional integer within the effective range of [1, 65535]. Default to 1.
 * Number of blocked connections from in_channels to out_channels.
 * The in_channels and out_channels must be divisible by groups.
 * When the groups value differs, the supported data type may vary, specifically as follows: \n
 * \n
 * | groups  |        dtype           | out_backprop format | filter format |     y format   |
 * |---------|------------------------|---------------------|---------------|----------------|
 * |  =1     |float16/bfloat16/float32|       NDHWC         |      NDHWC    |     NDHWC      |
 * |  =1     |float16/bfloat16/float32|       NDHWC         |      NCDHW    |     NDHWC      |
 * |  =1     |float16/bfloat16/float32|       NDHWC         |      DHWCN    |     NDHWC      |
 * |  >=1    |float16/bfloat16/float32|       NCDHW         |      NCDHW    |     NCDHW      |
 * |  >=1    |float16/bfloat16/float32|       NCDHW         |      NDHWC    |     NCDHW      |
 * |  >=1    |float16/bfloat16/float32|       NCDHW         |      DHWCN    |     NCDHW      |
 * \n
 * @li data_format: An optional string. The value must be one of ["NDHWC", "NCDHW"]. Defaults to "NDHWC".
 * The correspondence is as follows: batch(N), depth(D), height(H), width(W), channels(C).
 * Specify the data format of the out_backprop and y.
 * @li enable_hf32: Optional. An optional bool parameter. Used to enable hf32 computation.
 * If true, enable hf32 computation, otherwise, disable hf32 computation. Defaults to false. \n
*@par Outputs:
 * y: A tensor. It has the same format as out_backprop.
 * The type is float16, bfloat16, float32.
 * The gradients of feature map.
 * @attention Constraints:
 * In Ascend 950PR/Ascend 950DT: The behavior of gradient computation in the padding region depends on the input shape.
 * Depending on the operator optimization strategy, the padding gradients may be directly set to 0.\n
*@par Third-party framework compatibility
 * Compatible with Tensorflow's conv3d_backprop_input
*/
REG_OP(Conv3DBackpropInputV2)
    .INPUT(input_size, TensorType({DT_INT32, DT_INT64}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(out_backprop, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NDHWC")
    .ATTR(enable_hf32, Bool, false)
    .OP_END_FACTORY_REG(Conv3DBackpropInputV2)

/**
*@brief Computes the transpose of convolution 3d with respect to the input.

*@par Inputs:
 * @li input_size: A tensor of type int32 or int64. An integer vector
 * representing the shape of input.
 * @li x: A tensor of the following types, float16, float32, bfloat16, hifloat8, float8_e4m3fn. The format
 * is NDHWC or NCDHW.
 * @li filter: A 5D tensor.
 * The format is NCDHW，NDHWC, and DHWCN.
 * The NCDHW can be one of the following types: float16, bfloat16, float32, hifloat8, float8_e4m3fn.
 * The DHWCN and NDHWC can be one of the following types: float16, bfloat16, float32.
 * The height (H), width (W) dimension must be in [1, 511].
 * @li bias: Optional. An optional 1D tensor of type float16 or float32. Reserved.
 * @li offset_w: Optional. An optional 1D tensor of type int8 for quantized deconvolution.
 *  Reserved. \n

*@par Attributes:
 * @li strides: Required. A tuple/list of 5 integers. Specifies the stride of the sliding window
 * for each dimension of "x". The strides have the same axes sequence as "x":
 * [batch, stride_depth, stride_height, stride_width, channels] or
 * [batch, channels, stride_depth, stride_height, stride_width].
 * The N and C dimensions must be 1.
 * The height (H) and width (W) dimensions must be in [1, 63].
 * The depth (D) dimension must be in [1, 255].
 * @li pads: Required. A tuple/list of 6 integers.
 * All dimensions must be in [0, 255].
 * @li dilations: Optional. A tuple/list of 5 integers,
 * The dilation factor for each dimension of input. Defaults to [1, 1, 1, 1, 1]. \n
 * The batch(N) and channels dimensions must be 1.
 * The width (W), height (H) and depth(D) dimensions must be in [1, 255].
 * The dilations have the same axes sequence "x":
 * [batch, channels, dilation_depth, dilation_height, dilation_width] or
 * [batch, dilation_depth, dilation_height, dilation_width, channels].
 * @li groups: An optional integer within the effective range of [1, 65535]. Default to 1.
 * Number of blocked connections from in_channels to out_channels.
 * The in_channels and out_channels must be divisible by groups.
 * When the groups value differs, the supported data type may vary, specifically as follows: \n
 * \n
 * | groups  |        dtype           |   x format  | filter format |    y format    |
 * |---------|------------------------|-------------|---------------|----------------|
 * |  =1     |hifloat8/float8_e4m3fn  |    NCDHW    |      NCDHW    |     NCDHW      |
 * |  =1     |hifloat8/float8_e4m3fn  |    NDHWC    |      NCDHW    |     NDHWC      |
 * |  =1     |float16/bfloat16/float32|    NDHWC    |      NDHWC    |     NDHWC      |
 * |  =1     |float16/bfloat16/float32|    NDHWC    |      NCDHW    |     NDHWC      |
 * |  =1     |float16/bfloat16/float32|    NDHWC    |      DHWCN    |     NDHWC      |
 * |  >=1    |float16/bfloat16/float32|    NCDHW    |      NCDHW    |     NCDHW      |
 * |  >=1    |float16/bfloat16/float32|    NCDHW    |      NDHWC    |     NCDHW      |
 * |  >=1    |float16/bfloat16/float32|    NCDHW    |      DHWCN    |     NCDHW      |
 * |  >1     |hifloat8/float8_e4m3fn  |    NCDHW    |      NCDHW    |     NCDHW      |
 * \n
 * @li data_format:  An optional string. The value must be one of ["NDHWC", "NCDHW"]. Defaults to "NDHWC".
 * The correspondence is as follows: batch(N), depth(D), height(H), width(W), channels(C).
 * Specify the data format of the x and y.
 * @li output_padding: Optional. The size will be added in the output shape.
 * Defaults to [0, 0, 0, 0, 0]. The N and C dimensions must be 0.
 * @li offset_x: Optional. Defaults to 0. Reserved. \n
 * @li enable_hf32: Optional. An optional bool parameter. Used to enable hf32 computation.
 * If true, enable hf32 computation, otherwise, disable hf32 computation. Defaults to false. \n

*@par Outputs:
 * y: A tensor that has the type bfloat16, float16, float32, hifloat8, float8_e4m3fn. It has the same format as "x".
 * @attention Constraints:
 * In Ascend 950PR/Ascend 950DT: The behavior of gradient computation in the padding region depends on the input shape.
 * Depending on the operator optimization strategy, the padding gradients may be directly set to 0.\n
*/
REG_OP(Conv3DTransposeV2)
    .INPUT(input_size, TensorType({DT_INT32, DT_INT64}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_HIFLOAT8, DT_FLOAT8_E4M3FN}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_HIFLOAT8, DT_FLOAT8_E4M3FN}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_HIFLOAT8, DT_FLOAT8_E4M3FN}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NDHWC")
    .ATTR(output_padding, ListInt, {0, 0, 0, 0, 0})
    .ATTR(offset_x, Int, 0)
    .ATTR(enable_hf32, Bool, false)
    .OP_END_FACTORY_REG(Conv3DTransposeV2)

/**
* @brief Computes a 3D convolution with 5D "x", "filter" and "bias" tensors.
* Like this, output = CONV(x, filter) + bias. \n
* If case with 'int8' dtype appears in Atlas A3 Training Series Product/Atlas A3 Inference Series Product or
* Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component,
* like this: output = CONV(x, filter) * scale + bias.
* @par Inputs:
* @li x: A required 5D tensor of input image. \n
* In Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component or
* Atlas A3 Training Series Product/Atlas A3 Inference Series Product,
* a tensor with data type bfloat16, float16, float32, int8 and format "NCDHW" is supported. \n
* In Ascend 950 AI Processor, a tensor of type bfloat16, float16, float32 or hifloat8 and format "NCDHW" or
* "NDHWC" can be supported.
* @li filter: A required 5D tensor of convolution kernel. \n
* In Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component or
* Atlas A3 Training Series Product/Atlas A3 Inference Series Product,
* a tensor with data type bfloat16, float16, float32, int8 and format "NCDHW" is supported.
* Kernel_h and kernel_w should be both less than 512. \n
* In Ascend 950 AI Processor, a tensor with data type bfloat16, float16, float32 or hifloat8 and format "NCDHW" or
* "DHWCN" can be supported. Kernel_h and kernel_w should be both less than 256.
* @li bias: An optional 1D tensor of additive biases to the outputs.
* The data is stored in the order of: [out_channels]. \n
* In Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component or
* Atlas A3 Training Series Product/Atlas A3 Inference Series Product,
* a tensor with data type float16, float32 and format "ND" is supported. \n
* In Ascend 950 AI Processor, a tensor with data type bfloat16, float16 or float32 and format "ND" can be supported.
* @li scale: A optional 1D tensor of scaling factors.
* The data is stored in the order of: [out_channels]. \n
* In Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component or
* Atlas A3 Training Series Product/Atlas A3 Inference Series Product,
* a tensor with data type float32 and format "ND" is supported. \n
* In Ascend 950 AI Processor, this parameter is not supported.
* @li offset: An optional 1D tensor of bias.
* The data is stored in the order of: [out_channels].
* In Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component or
* Atlas A3 Training Series Product/Atlas A3 Inference Series Product,
* a tensor with data type float32 and format "ND" is supported. \n
* In Ascend 950 AI Processor, this parameter is not supported.
* @li offset_w: An optional quantitative offset tensor. A tensor of type int8. Reserved.
*\n
* @li The following are the supported data types and data formats for
* (Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component and
* Atlas A3 Training Series Product/Atlas A3 Inference Series Product):
\n
| Tensor    | x        | filter   | bias     | scale   | offset  |    y     |\n
| :-------: | :------: | :------: | :------: | :-----: | :-----: | :------: |\n
| Data Type | float16  | float16  | float16  | float32 | float32 | float16  |\n
|           | bfloat16 | bfloat16 | float32  | float32 | float32 | bfloat16 |\n
|           | float32  | float32  | float32  | float32 | float32 | float32  |\n
|           | int8     | int8     | float32  | float32 | float32 | bfloat16 |\n
| Format    | NCDHW    | NCDHW    | ND       | ND      | ND      | NCDHW    |\n
\n
* The following are the supported data types and data formats for Ascend 950 AI Processor:
\n
| Tensor    | x        | filter   | bias     |    y     |\n
| :-------: | :------: | :------: | :------: | :------: |\n
| Data Type | float16  | float16  | float16  | float16  |\n
|           | bfloat16 | bfloat16 | bfloat16 | bfloat16 |\n
|           | float32  | float32  | float32  | float32  |\n
|           | hifloat8 | hifloat8 | float32  | hifloat8 |\n
| Format    | NCDHW    | NCDHW    | ND       | NCDHW    |\n
|           | NDHWC    | DHWCN    | ND       | NDHWC    |\n
\n
* @par Attributes:
* @li strides: Required. A list of 5 integers. The stride of the sliding window
* for each dimension of input. The dimension order is determined by the data
* format of "x". The n and in_channels dimensions must be set to 1.
* When the format is "NDHWC", its shape is [1, stride_d, stride_h, stride_w, 1],
* when the format is "NCDHW", its shape is [1, 1, stride_d, stride_h, stride_w].
* @li pads: Optional. A list of 6 integers. The number of pixels to add to each
* (pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right) side of the input. Defaults to [0, 0, 0, 0, 0, 0].
* @li dilations: Optional. A list of 5 integers. The dilation factor for each
* dimension of input. The dimension order is determined by the data format of
* "x". The n and in_channels dimensions must be set to 1.
* When the format is "NDHWC", its shape is [1, dilation_d, dilation_h, dilation_w, 1],
* when the format is "NCDHW", its shape is [1, 1, dilation_d, dilation_h, dilation_w]. Defaults to [1, 1, 1, 1].
* @li groups: Optional. An integer of type int32. The number of groups
* in group convolution. In_channels and out_channels must both be divisible by "groups". Defaults to 1.
* In Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component or
* Atlas A3 Training Series Product/Atlas A3 Inference Series Product, groups can only be equal to 1.
* @li data_format: Optional. It is a string represents input's data format. Defaults to "NCDHW". Reserved.
* @li offset_x: Optional. An integer of type int32. It means offset in quantization algorithm
* and is used for filling in pad values. Defaults to 0. It can only be supported in
* Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component or
* Atlas A3 Training Series Product/Atlas A3 Inference Series Product.
* @li pad_mode: Optional. An optional string parameter, indicating the mode of pad.
* It must be "SPECIFIC" or "SAME" or "VALID". Defaults to "SPECIFIC".
* @li enable_hf32: Optional. An optional bool parameter. Used to enable hf32 computation.
* If true, enable hf32 computation, otherwise, disable hf32 computation. Defaults to false.
* @par Outputs:
* y: A 5D tensor of output. \n
* In Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component or
* Atlas A3 Training Series Product/Atlas A3 Inference Series Product,
* a tensor with data type bfloat16,float32,float16 and format "NCDHW" is supported,
* which the data is stored in [n, out_channels, out_depth, out_height, out_width]. \n
* In Ascend 950 AI Processor, a tensor with data type bfloat16, float16, float32 or hifloat8 and
* format "NCDHW" or "NDHWC" can be supported. which the data is stored in
* [n, out_channels, out_depth, out_height, out_width] or [n, out_depth, out_height, out_width, out_channels].
*\n
*     out_depth  = (d + pad_head + pad_tail -
*                   (dilation_d * (kernel_d - 1) + 1))
*                  / stride_d + 1
*\n
*     out_height = (h + pad_top + pad_bottom -
*                   (dilation_h * (kernel_h - 1) + 1))
*                  / stride_h + 1
*\n
*     out_width = (w + pad_left + pad_right -
*                  (dilation_w * (kernel_w - 1) + 1))
*                 / stride_w + 1
*\n
* @attention Constraints:
* @li The following value range restrictions must be met:
*\n
| Name             | Field                | Scope        |\n
| :--------------: | :------------------: | :----------: |\n
| x size           | n                    | [1, 1000000] |\n
|                  | in_channels          | [1, 1000000] |\n
|                  | d                    | [1, 1000000] |\n
|                  | h                    | [1, 100000]  |\n
|                  | w                    | [1, 4096]    |\n
| filter size      | out_channels         | [1, 1000000] |\n
|                  | in_channels / groups | [1, 1000000] |\n
|                  | kernel_d             | [1, 1000000] |\n
|                  | kernel_h             | [1, 255]     |\n
|                  | kernel_w             | [1, 255]     |\n
| bias size        | out_channels         | [1, 1000000] |\n
| scale size       | out_channels         | [1, 1000000] |\n
| offset size      | out_channels         | [1, 1000000] |\n
| strides          | stride_d             | [1, 1000000] |\n
|                  | stride_h             | [1, 63]      |\n
|                  | stride_w             | [1, 63]      |\n
| pads             | pad_head             | [0, 1000000] |\n
|                  | pad_tail             | [0, 1000000] |\n
|                  | pad_top              | [0, 255]     |\n
|                  | pad_bottom           | [0, 255]     |\n
|                  | pad_left             | [0, 255]     |\n
|                  | pad_right            | [0, 255]     |\n
| dilations        | dilation_d           | [1, 1000000] |\n
|                  | dilation_h           | [1, 255]     |\n
|                  | dilation_w           | [1, 255]     |\n
| groups           | -                    | [1, 65535]   |\n
| data_format      | -                    | ["NDHWC", "NCDHW"] |\n
| offset_x         | -                    | [-128, 127]  |\n
| pad_mode         | -                    | ["SPECIFIC", "SAME", "VALID"] |\n
| enable_hf32      | -                    | [true, false] |\n
*\n
* @li The w dimension of the input image supports cases exceeding 4096, but it may
* cause compilation errors.
*\n
* @li If any dimension of x/filter/bias/scale/offset/y shape exceeds max 1000000,
* the product of each dimension of x/filter/bias/scale/offset/y
* shape exceeds max int32 minus one (2147483646) or the value of
* strides/pads/dilations/groups/data_format/offset_x/pad_mode/enable_hf32
* exceeds the range in the above table, the correctness of the operator cannot be guaranteed.
\n
*/

REG_OP(Conv3DV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT8, DT_HIFLOAT8}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT8, DT_HIFLOAT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_HIFLOAT8}))
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(pads, ListInt, {0, 0, 0, 0, 0, 0})
    .ATTR(dilations, ListInt, {1, 1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NCDHW")
    .ATTR(offset_x, Int, 0)
    .ATTR(pad_mode, String, "SPECIFIC")
    .ATTR(enable_hf32, Bool, false)
    .OP_END_FACTORY_REG(Conv3DV2)

/**
*@brief Computes the deformed convolution output with the expected input
* @par Inputs:
* Two inputs:
* @li x: A 4D tensor of input image. A tensor of type float16, float32, bfloat16. The format support NHWC.
* Shape support 4D.
* @li offsets: A tensor of type float16, float32, bfloat16. Deformation offset parameter.
* The format support NHWC. Shape support 4D. Has the same format and dtype as "x".

*@par Attributes:
* @li strides: A tuple/list of 4 integers. The stride of the sliding window for
* height and width for H/W dimension. Required and no default value.
* @li pads: A tuple/list of 4 integers. Padding added to H/W dimension
* of the input. Required and no default value.
* @li ksize: A tuple/list of 2 integers. Kernel size. Required and no default value.
* @li dilations: A tuple/list of 4 integers. The dilation factor for each dimension
* of input. Defaults to [1, 1, 1, 1]
* @li data_format: An optional string from: "NCHW", "NHWC". The default value "NCHW" is not supported.
* Specify the data format of the input x. The format of the attribute
* @li deformable_groups: An optional int specify the c-axis grouping number of input x. Defaults to "1".
* @li modulated: An optional bool specify version of DeformableConv2D, true means v2, false means v1. Defaults to
"true".
* Only support true now.

*@par Outputs:
* y: Deformed convolution output. A tensor of type float16, float32, bfloat16. The format support NHWC.
* Shape support 4D. Has the same format and dtype as input "x".
*/
REG_OP(DeformableOffsets)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(offsets, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .REQUIRED_ATTR(ksize, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(data_format, String, "NHWC")
    .ATTR(deformable_groups, Int, 1)
    .ATTR(modulated, Bool, true)
    .OP_END_FACTORY_REG(DeformableOffsets)

/**
* @brief ExtendConv2D computes a 2D convolution + fixpipe fused operator.
* @par Inputs:
* @li x: A required 4D tensor of input image. With the format "NCHW" which shape is
* [n, in_channels, h, w].
* @li filter: A required 4D tensor of convolution kernel.
* With the format "NCHW" which shape is [out_channels, in_channels / groups, kernel_h, kernel_w].
* @li bias: An optional 1D tensor of additive biases to the outputs.
* The data is stored in the order of: [out_channels].
* @li offset: An optional quantitative offset tensor. Reserved.
* @li scale0: An optional 1D tensor. Quantization/dequantization/weighting parameter corresponding to
* the first output, which is of the channelwise type.
* @li relu_weight0: An optional 1D tensor. Indicates the relu slope parameter corresponding to the
* first output, which is of the scalar type or channelwise type.
* @li clip_value0: An optional 1D tensor. Truncated value of clip relu corresponding to the first output.
* The value is of the scalar type.
* @li scale1: An optional 1D tensor. Quantization/dequantization/weighting parameter corresponding to
* the second output, which is of the scalar type or channelwise type.
* @li relu_weight1: An optional 1D tensor. Quantization/dequantization/weighting parameter corresponding to
* the second output, which is of the scalar type or channelwise type.
* @li clip_value1: An optional 1D tensor. Truncated value of clip relu corresponding to the seccond output.
* The value is of the scalar type.
*\n
* The following are the supported data types and data formats:
*\n
| Tensor    | x        | filter   | bias    | scale0/1     | relu_weight0/1 | clip_value0/1 | y0/1    |\n
| :-------: | :------: | :------: | :-----: | :----------: | :------------: | :-----------: | :-----: |\n
| Data Type | int8     | int8     | int32   | uint64/int64 | float32        | int8          | float16/int8 |\n
|           | hifloat8 | hifloat8 | float32 | uint64/int64 | float32        | hifloat8      | float32/float16/bfloat16/hifloat8 |\n
|           | float8   | float8   | float32 | uint64/int64 | float32        | float8        | float32/float16/bfloat16/float8_e4m3 |\n
| Format    | NCHW     | NCHW     | ND      | ND           | ND             | ND            | NCHW    |\n

| Tensor    | x        | filter   | bias    | scale0/1     | relu_weight0/1 | clip_value0/1 | y0/1    |\n
| :-------: | :------: | :------: | :-----: | :----------: | :------------: | :-----------: | :-----: |\n
| Data Type | int8     | int8     | int32   | uint64/int64 | float32        | int8          | float16/int8 |\n
| Format    | NHWC     | HWCN     | ND      | ND           | ND             | ND            | NHWC    |\n
*\n
* @par Attributes:
* @li strides: Required. A list of 4 integers. The stride of the sliding window
* for each dimension of input. The dimension order is determined by the data
* format of "x". The n and in_channels dimensions must be set to 1.
* When the format is "NCHW", its shape is [1, 1, stride_h, stride_w].
* @li pads: Optional. A list of 4 integers. The number of pixels to add to each
* (pad_top, pad_bottom, pad_left, pad_right). Defaults to [0, 0, 0, 0].
* @li dilations: Optional. A list of 4 integers. The dilation factor for each
* dimension of input. The dimension order is determined by the data format of
* "x". The n and in_channels dimensions must be set to 1.
* When the format is "NCHW", its shape is [1, 1, dilation_h, dilation_w]. Defaults to [1, 1, 1, 1].
* @li groups: Optional. An integer of type int32. The number of groups
* in group convolution. In_channels and out_channels must both be divisible by "groups". Defaults to 1.
* @li data_format: Optional. It is a string represents input's data format.
* Defaults to "NCHW". Reserved.
* @li offset_x: Optional. An integer of type int32. It means offset in quantization algorithm
* and is used for filling in pad values. Ensure that the output is within the
* effective range. Defaults to 0. Reserved.
* @li round_mode: Rounding mode enable quantization.
* If output's dtype is hifloat8, round_mode can be set to 'round'. Otherwise, it can be set to 'rint'.
* Defaults to "rint".
* @li pad_mode: Optional. An optional string parameter, indicating the mode of pad.
* It must be "SPECIFIC" or "SAME" or "VALID" or "SAME_UPPER" or "SAME_LOWER". Defaults to "SPECIFIC".
* @li enable_hf32: Optional. An bool for extendConv2D. If True, enable hf32 calculation. Defaults to false.
* If False, disable hf32 calculation. Must be false.
* @li enable_relu0: Optional. Indicates whether relu is enabled for the first output. Defaults to false.
* If False, relu is disable for the first output. Must be false.
* @li enable_relu1: Optional. Indicates whether relu is enabled for the second output. Defaults to false.
* If False, relu is disable for the second output. Must be false.
* @li dual_output: Optional. Indicates whether dual outputs are used. Defaults to false.
* When dual_output is false, extendConv2D only has output y0. When dual_output is true,
* extendConv2D has output y0 and y1. Must be false.
* @li dtype0: Optional. A integer of type int8. It means the dtype of output y0.
* Support list is [-1(Default), 0(DT_FLOAT), 1(DT_FLOAT16), 2(DT_INT8), 27(DT_BF16),
* 34(DT_HIFLOAT8), 36(DT_FLOAT8_E4M3FN)]. Defaults to -1, means the dtype is the same as x.
* @li dtype1: Optional. A integer of type int8. It means the dtype of output y1.
* Support list is [-1(Default), 0(DT_FLOAT), 1(DT_FLOAT16), 2(DT_INT8), 27(DT_BF16),
* 34(DT_HIFLOAT8), 36(DT_FLOAT8_E4M3FN)]. Defaults to -1, means the dtype is the same as x. Must be -1.
*\n
* @par Outputs:
* @li y0: The first output of ExtendConv2D. A 4D Tensor of output feature map. Has the same type as "x".
* With the format "NCHW", the data is stored in the order of: [n, out_channels, out_height, out_width].
* @li y1: The second output of ExtendConv2D. A 4D Tensor of output feature map. Has the same type as "x".
* With the format "NCHW", the data is stored in the order of: [n, out_channels, out_height, out_width].
*\n
*     out_height = (h + pad_top + pad_bottom -
*                   (dilation_h * (kernel_h - 1) + 1))
*                  / stride_h + 1
*\n
*     out_width = (w + pad_left + pad_right -
*                  (dilation_w * (kernel_w - 1) + 1))
*                 / stride_w + 1
*\n
* @attention Constraints:
* @li The following value range restrictions must be met:
*\n
| Name              | Field                | Scope        |\n
| :---------------: | :------------------: | :----------: |\n
| x size            | n                    | [1, 1000000] |\n
|                   | in_channels          | [1, 1000000] |\n
|                   | h                    | [1, 100000]  |\n
|                   | w                    | [1, 4096]    |\n
| filter size       | out_channels         | [1, 1000000] |\n
|                   | in_channels / groups | [1, 1000000] |\n
|                   | kernel_h             | [1, 511]     |\n
|                   | kernel_w             | [1, 511]     |\n
| bias size         | out_channels         | [1, 1000000] |\n
| offset_w size     | out_channels or 1    | [1, 1000000] |\n
| scale0 size       | out_channels or 1    | [1, 1000000] |\n
| relu_weight0 size | out_channels or 1    | [1, 1000000] |\n
| clip_value0 size  | out_channels or 1    | [1, 1000000] |\n
| scale1 size       | out_channels or 1    | [1, 1000000] |\n
| relu_weight1 size | out_channels or 1    | [1, 1000000] |\n
| clip_value1 size  | out_channels or 1    | [1, 1000000] |\n
| strides           | stride_h             | [1, 63]      |\n
|                   | stride_w             | [1, 63]      |\n
| pads              | pad_top              | [0, 255]     |\n
|                   | pad_bottom           | [0, 255]     |\n
|                   | pad_left             | [0, 255]     |\n
|                   | pad_right            | [0, 255]     |\n
| dilations         | dilation_h           | [1, 255]     |\n
|                   | dilation_w           | [1, 255]     |\n
| groups            | -                    | [1, 65535]   |\n
| data_format       | -                    | ["NCHW"] |\n
| offset_x          | -                    | [-128, 127]  |\n
| round_mode        | -                    | ["rint", "round"] |\n
| pad_mode          | -                    | ["SPECIFIC", "SAME", "VALID", "SAME_UPPER", "SAME_LOWER"] |\n
| enable_hf32       | -                    | [true, false] |\n
| enable_relu0      | -                    | [true, false] |\n
| enable_relu1      | -                    | [true, false] |\n
| dual_output       | -                    | [true, false] |\n
*\n
* @li In Ascend 950 AI Processor: If any dimension of
* x/filter/bias/offset_w/scale0/relu_weight0/clip_value0/scale1/relu_weight1/clip_value1/y shape exceeds max
* 1000000, the product of each dimension of x/filter/bias/scale/offset/y
* shape exceeds max int32(2147483647) or the value of strides/pads/dilations/offset_x
* exceeds the range in the above table, the correctness of the operator cannot be guaranteed.
*\n
*/
REG_OP(ExtendConv2D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_BF16, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_BF16, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_BF16}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OPTIONAL_INPUT(scale0, TensorType({DT_INT64, DT_UINT64}))
    .OPTIONAL_INPUT(relu_weight0, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(clip_value0, TensorType({DT_FLOAT16, DT_INT8, DT_BF16, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .OPTIONAL_INPUT(scale1, TensorType({DT_INT64, DT_UINT64}))
    .OPTIONAL_INPUT(relu_weight1, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(clip_value1, TensorType({DT_FLOAT16, DT_INT8, DT_BF16, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .OUTPUT(y0, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_BF16, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .OUTPUT(y1, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_BF16, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(pads, ListInt, {0, 0, 0, 0})
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NCHW")
    .ATTR(offset_x, Int, 0)
    .ATTR(round_mode, String, "rint")
    .ATTR(pad_mode, String, "SPECIFIC")
    .ATTR(enable_hf32, Bool, false)
    .ATTR(enable_relu0, Bool, false)
    .ATTR(enable_relu1, Bool, false)
    .ATTR(dual_output, Bool, false)
    .ATTR(dtype0, Int, -1)
    .ATTR(dtype1, Int, -1)
    .OP_END_FACTORY_REG(ExtendConv2D)

/**
*@brief Computes the transpose of convolution 3d with respect to the input.

*@par Inputs:
 * @li input_size: A tensor of type int32 or int64. An integer vector
 * representing the shape of input.
 * @li x: A tensor of the following types, DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT8. Only support DT_FLOAT16 and DT_INT8 right now.
 * The format is NCDHW.
 * @li filter: A 5D tensor.
 * The format is NCDHW，and NDHWC.
 * The height (H), width (W) dimension must be in [1, 511].
 * @li bias: Optional. An optional 1D tensor of type float16, float32 or int32.
 * The data is stored in the order of: [out_channels].
 * @li scale: An optional 1D tensor. Quantization/dequantization/weighting parameter corresponding to
* the first output, which is of the channelwise type. \n

*@par Attributes:
 * @li strides: Required. A tuple/list of 5 integers. Specifies the stride of the sliding window
 * for each dimension of "x". The strides have the same axes sequence as "x":
 * [batch, stride_depth, stride_height, stride_width, channels] or
 * [batch, channels, stride_depth, stride_height, stride_width].
 * The N and C dimensions must be 1.
 * The height (H) and width (W) dimensions must be in [1, 63].
 * The depth (D) dimension must be in [1, 255].
 * @li pads: Required. A tuple/list of 6 integers.
 * All dimensions must be in [0, 255].
 * @li dilations: Optional. A tuple/list of 5 integers,
 * The dilation factor for each dimension of input. Defaults to [1, 1, 1, 1, 1]. \n
 * The batch(N) and channels dimensions must be 1.
 * The width (W), height (H) and depth(D) dimensions must be in [1, 255].
 * The dilations have the same axes sequence "x":
 * [batch, channels, dilation_depth, dilation_height, dilation_width] or
 * [batch, dilation_depth, dilation_height, dilation_width, channels].
 * @li groups: An optional integer within the effective range of [1, 65535]. Default to 1.
 * Number of blocked connections from in_channels to out_channels.
 * The in_channels and out_channels must be divisible by groups.
 * When the groups value differs, the supported data type may vary, specifically as follows: \n
 * \n
 * | groups |        dtype           |   x format  | filter format |    y format    |
 * |--------|------------------------|-------------|---------------|----------------|
 * |  =1    |int8                    |    NCDHW    |      NCDHW    |     NCDHW      |
 * |  =1    |int8                    |    NCDHW    |      NDHWC    |     NCDHW      |
 * |  =1    |float16                 |    NCDHW    |      NCDHW    |     NCDHW      |
 * |  =1    |float16                 |    NCDHW    |      NDHWC    |     NCDHW      |
 * |  >1    |int8                    |    NCDHW    |      NCDHW    |     NCDHW      |
 * |  >1    |float16                 |    NCDHW    |      NCDHW    |     NCDHW      |
 * |  >1    |float16                 |    NCDHW    |      NDHWC    |     NCDHW      |
 * \n
 * @li data_format:  An optional string. The value must be one of ["NDHWC", "NCDHW"]. Defaults to "NDHWC".
 * The correspondence is as follows: batch(N), depth(D), height(H), width(W), channels(C).
 * Specify the data format of the x and y.
 * @li output_padding: Optional. The size will be added in the output shape.
 * Defaults to [0, 0, 0, 0, 0]. The N and C dimensions must be 0.
 * @li offset_x: Optional. Defaults to 0. Reserved. 
 * @li fusion_mode: Optional. Only support 0 or 1, Indicates whether relu is enabled for the output. Defaults to 0.
* If 0, relu is disable for the output.
* @li y_quant_mode: Optional. Defaults to 0. Reserved. \n

*@par Outputs:
 * y: It has the same type and format as "x".
*/
REG_OP(ExtendConvTranspose)
    .INPUT(input_size, TensorType({DT_INT32, DT_INT64}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT8}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT32, DT_INT32}))
    .OPTIONAL_INPUT(scale, TensorType({DT_UINT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT8}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NDHWC")
    .ATTR(output_padding, ListInt, {0, 0, 0, 0, 0})
    .ATTR(offset_x, Int, 0)
    .ATTR(fusion_mode, Int, 0)
    .ATTR(y_quant_mode, Int, 0)
    .OP_END_FACTORY_REG(ExtendConvTranspose)

/**
* @brief Computes a 3D convolution with 5D "x", "filter" and "bias" tensors,
* and then executes a per-channel dequant operation with "scale" tensors.
* Like this, output = (CONV(x, filter) + bias) * scale.
* @par Inputs:
* @li x: A required 5D tensor of input image. With the format "NDHWC" which shape is
* [n, d, h, w, in_channels] or the format "NCDHW" which shape is [n, in_channels, d, h, w].
* @li filter: A required 5D tensor of convolution kernel.
* With the format "DHWCN" which shape is [kernel_d, kernel_h, kernel_w, in_channels / groups, out_channels]
* or the format "NCDHW" which shape is [out_channels, in_channels / groups, kernel_d, kernel_h, kernel_w].
* @li scale: A required 1D tensor of scaling factors. The data is stored in the order of: [out_channels].
* @li bias: An optional 1D tensor of additive biases to the outputs.
* The data is stored in the order of: [out_channels].
* @li offset: An optional quantitative offset tensor. Reserved.
*\n
* The following are the supported data types and data formats (for Ascend 950 AI Processor):
*\n
| Tensor    | x           | filter      | scale        | bias    | offset  | y                                   |\n
| :-------: | :---------: | :---------: | :----------: | :------:| :------:| :---------------------------------: |\n
| Data Type | int8        | int8        | uint64/int64 | int32   | float32 | float16                             |\n
|           | float8_e4m3 | float8_e4m3 | uint64/int64 | float32 | float32 | float32/float16/bfloat16/float8_e4m3|\n
|           | hifloat8    | hifloat8    | uint64/int64 | float32 | float32 | float32/float16/bfloat16/hifloat8   |\n
| Format    | NCDHW       | NCDHW       | ND           | ND      | ND      | NCDHW                               |\n
*\n
* @par Attributes:
* @li dtype: Required. A integer of type int8. It means output's dtype.
* @li strides: Required. A list of 5 integers. The stride of the sliding window
* for each dimension of input. The dimension order is determined by the data
* format of "x". The n and in_channels dimensions must be set to 1.
* When the format is "NDHWC", its shape is [1, stride_d, stride_h, stride_w, 1],
* when the format is "NCDHW", its shape is [1, 1, stride_d, stride_h, stride_w].
* @li pads: Required. A list of 6 integers. The number of pixels to add to each
* (pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right) side of the input.
* @li dilations: Optional. A list of 5 integers. The dilation factor for each
* dimension of input. The dimension order is determined by the data format of
* "x". The n and in_channels dimensions must be set to 1.
* When the format is "NDHWC", its shape is [1, dilation_d, dilation_h, dilation_w, 1],
* when the format is "NCDHW", its shape is [1, 1, dilation_d, dilation_h, dilation_w]. Defaults to [1, 1, 1, 1].
* @li groups: Optional. An integer of type int32. The number of groups
* in group convolution. In_channels and out_channels must both be divisible by "groups". Defaults to 1.
* @li data_format: Optional. It is a string represents input's data format.
* Defaults to "NCDHW". Reserved.
* @li offset_x: Optional. An integer of type int32. It means offset in quantization algorithm
* and is used for filling in pad values. Ensure that the output is within the
* effective range. Defaults to 0. Reserved.
* @li round_mode: Optional. Defaults to "rint". It is rounding mode of calculation.
* If output's data type is hifloat8, round_mode can be set to 'round'. Otherwise, it can be set to 'rint'.
* @li pad_mode: Optional. An optional string parameter, indicating the mode of pad.
* It must be "SPECIFIC" or "SAME" or "VALID". Defaults to "SPECIFIC".
*\n
* @par Outputs:
* y: A 5D tensor of output feature map.
* With the format "NDHWC", the data is stored in the order of: [n, out_depth, out_height, out_width, out_channels]
* or the format "NCDHW", the data is stored in the order of: [n, out_channels, out_depth, out_height, out_width].
*\n
*     out_depth  = (d + pad_head + pad_tail -
*                   (dilation_d * (kernel_d - 1) + 1))
*                  / stride_d + 1
*\n
*     out_height = (h + pad_top + pad_bottom -
*                   (dilation_h * (kernel_h - 1) + 1))
*                  / stride_h + 1
*\n
*     out_width = (w + pad_left + pad_right -
*                  (dilation_w * (kernel_w - 1) + 1))
*                 / stride_w + 1
*\n
* @attention Constraints:
* @li The following value range restrictions must be met:
*\n
| Name             | Field                | Scope        |\n
| :--------------: | :------------------: | :----------: |\n
| x size           | n                    | [1, 1000000] |\n
|                  | in_channels          | [1, 1000000] |\n
|                  | d                    | [1, 1000000] |\n
|                  | h                    | [1, 100000]  |\n
|                  | w                    | [1, 4096]    |\n
| filter size      | out_channels         | [1, 1000000] |\n
|                  | in_channels / groups | [1, 1000000] |\n
|                  | kernel_d             | [1, 1000000] |\n
|                  | kernel_h             | [1, 511]     |\n
|                  | kernel_w             | [1, 511]     |\n
| scale size       | out_channels         | [1, 1000000] |\n
| bias size        | out_channels         | [1, 1000000] |\n
| offset size      | out_channels         | [1, 1000000] |\n
| strides          | stride_d             | [1, 1000000] |\n
|                  | stride_h             | [1, 63]      |\n
|                  | stride_w             | [1, 63]      |\n
| pads             | pad_head             | [0, 1000000] |\n
|                  | pad_tail             | [0, 1000000] |\n
|                  | pad_top              | [0, 255]     |\n
|                  | pad_bottom           | [0, 255]     |\n
|                  | pad_left             | [0, 255]     |\n
|                  | pad_right            | [0, 255]     |\n
| dilations        | dilation_d           | [1, 1000000] |\n
|                  | dilation_h           | [1, 255]     |\n
|                  | dilation_w           | [1, 255]     |\n
| groups           | -                    | [1, 65535]   |\n
| data_format      | -                    | ["NDHWC", "NCDHW"] |\n
| offset_x         | -                    | [-128, 127]  |\n
| pad_mode         | -                    | ["SPECIFIC", "SAME", "VALID"] |\n
| enable_hf32      | -                    | [true, false] |\n
*\n
* @li The W dimension of the input image supports cases exceeding 4096, but it may
* cause compilation errors.
*\n
* @li If any dimension of x/filter/scale/bias/offset/y shape exceeds max int32 minus one (2147483646),
* the product of each dimension of x/filter/scale/bias/offset/y shape exceeds max int32 minus one (2147483646) or
* the value of strides/pads/dilations/groups/data_format/offset_x/pad_mode/enable_hf32 exceeds the range
* in the above table, the correctness of the operator cannot be guaranteed.
*\n
* @par Quantization supported or not
* Yes
*/
REG_OP(QuantConv3D)
    .INPUT(x, TensorType({DT_INT8, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .INPUT(filter, TensorType({DT_INT8, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .INPUT(scale, TensorType({DT_UINT64, DT_INT64}))
    .OPTIONAL_INPUT(bias, TensorType({DT_INT32, DT_FLOAT}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .REQUIRED_ATTR(dtype, Int)
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(pads, ListInt, {0, 0, 0, 0, 0, 0})
    .ATTR(dilations, ListInt, {1, 1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NCDHW")
    .ATTR(offset_x, Int, 0)
    .ATTR(round_mode, String, "rint")
    .ATTR(pad_mode, String, "SPECIFIC")
    .OP_END_FACTORY_REG(QuantConv3D)

/**
    * @brief Activation function of SwiGlu with clipping.

    * @par Inputs:
    * Two inputs, including:
    * @li x: A tensor. Type is bfloat16, float16, float32.
    * @li group_index: An optional tensor. Shape is (N,). Type is int64. 

    * @par Outputs:
    * one output, including:
    * y: A tensor. Type is bfloat16, float16, float32.

    * @par Attributes:
    * Five attributes, including:
    * @li dim: An optional int. The dimension to be split, value in [-xDim, xDim-1], default is -1.
    * @li alpha: An optional float. The activation coefficient for the GLU activation function, default is 1.702.
    * @li limit: An optional float. The threshold limit for SWIGLU input, default is 7.0.
    * @li bias: An optional float. The bias applied during SWIGLU linear computation, default is 1.0.
    * @li interleaved: An optional bool. The way of splitting x: true for interleaved splitting, false for front-back splitting, default is true.

    * @attention Constraints:
    * The dim dimension of x must be divisible by 2, and the dim dimension of y must be equal to the dim dimension of x divided by 2.
    */
    REG_OP(ClippedSwiglu)
        .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
        .OPTIONAL_INPUT(group_index, TensorType({DT_INT64}))
        .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
        .ATTR(dim, Int, -1)
        .ATTR(alpha, Float, 1.702)
        .ATTR(limit, Float, 7.0)
        .ATTR(bias, Float, 1.0)
        .ATTR(interleaved, Bool, true)
        .OP_END_FACTORY_REG(ClippedSwiglu)

/**
* @brief Activation function fused from sigmoid and ReLU, with soft saturation
* on the left and no saturation on the right .

* @par Inputs:
* x: An ND or 5HD tensor. Support 1D~8D. Must be one of the following types: bfloat16, float16, float32.

* @par Attributes:
* @li alpha: An optional float32. Defines at which negative value the ELU saturates. Defaults to "1.0".
* @li scale: An optional float32. Input data scaling factor. Defaults to "1.0".
* @li input_scale: An optional float32. Negative data scaling factor. Defaults to "1.0".

* @par Outputs:
* y: A bfloat16, float16, float32, for the normalized result.
* Has the same type, shape and format as input x.

* @par Third-party framework compatibility
* @li Compatible with Tensorflow's Elu operator
* @li Compatible with Caffe's ELULayer operator
* @li Compatible with Pytorch's elu Opeartor
*
*/
REG_OP(Elu)
    .INPUT(x, TensorType({FloatingDataType, DT_BF16}))
    .OUTPUT(y, TensorType({FloatingDataType, DT_BF16}))
    .ATTR(alpha, Float, 1.0)
    .ATTR(scale, Float, 1.0)
    .ATTR(input_scale, Float, 1.0)
    .OP_END_FACTORY_REG(Elu)

/**
* @brief Computes gradients for the exponential linear (Elu) operation.
*
* @par Inputs:
* @li grads: An ND or 5HD tensor. Support 1D~8D. Must be one of the following types:
* bfloat16, float16, float32, float64.
* The backpropagated gradients to the corresponding Elu operation.
* @li activations: A tensor. Has the same type, format and shape as "grads".
* The outputs of the corresponding Elu operation.
*
* @par Outputs:
* y: A tensor. Has the same type, format and shape as "grads".
*
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator EluGrad.
*
*/
REG_OP(EluGrad)
    .INPUT(grads, TensorType({FloatingDataType, DT_BF16}))
    .INPUT(activations, TensorType({FloatingDataType, DT_BF16}))
    .OUTPUT(y, TensorType({FloatingDataType, DT_BF16}))
    .OP_END_FACTORY_REG(EluGrad)

/**
*@brief Calculate the elu_grad_v2 function.
*Applies the element-wise function:
*Computes the backward for the elu.
*
*@par Inputs:
*Two inputs, including:
* @li grads: A tensor. Must be one of the following types:
*     float16, float32, bfloat16.
* @li activations: A tensor. Must be one of the following types:
*     float16, float32, bfloat16.
*
*@par Outputs:
*y: A Tensor with the same type and shape of grads's.
*
*@par Attributes:
* @li alpha: scalar parameter of type float32, the same as elu forward function, 
*     it is a hyperparameter controls the value to which an ELU saturates for negative net input, 
*     default value = 1.0 .
* @li scale: scalar parameter of type float32, the same scale as elu forward function scale, 
*     default value = 1.0 .
* @li input_scale: scalar parameter of type float32, which adjusts the output when activations are less than zero
*     default value = 1.0 .
* @li is_result: optional bool, if true, activations is result tensor, 
*     if false, activations is self tensor, default value is false.
*
*@attention Constraints:
*Shapes and datatypes of grads and activations must be the same.
*/
REG_OP(EluGradV2)
    .INPUT(grads, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(activations, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(alpha, Float, 1.0)
    .ATTR(scale, Float, 1.0)
    .ATTR(input_scale, Float, 1.0)
    .ATTR(is_result, Bool, false)
    .OP_END_FACTORY_REG(EluGradV2)

/**
* @brief FatreluMul divides the input tensor into left and right tensors x1 and x2 based on the last dimension,
* performs Threshold calculation on x1 on the left, and multiplies the calculation result by x2. \n

* @par Inputs:
* @li x: A tensor of type float, float16 or bfloat16. Shape support 2D ~ 8D.
* The format must be ND.
* @li threshold: A scalar, type is float, used to set the threshold of "x".

* @par Outputs:
* y: A tensor has the same type and format as "x".
* Other dimensions of its shape are the same as those of "x".
* The value of the last dimension is half the value of the last dimension of "x". \n
*/
REG_OP(FatreluMul)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(threshold, TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(FatreluMul)

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
* @brief The GELU activation function is x*Φ(x),
* where Φ(x) the standard Gaussian cumulative distribution function.

* @par Inputs:
* x: A Tensor. Must be one of the following types: bfloat16, float16, float32. \n

* @par Outputs:
* y: A Tensor. Has the same type as "x". \n

* @par Third-party framework compatibility:
* Compatible with the TensorFlow operator Gelu.
*/
REG_OP(Gelu)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(Gelu)

/**
* @brief Computes the gradient for the gelu of "x" .

* @par Inputs:
* Three inputs, including:
* @li dy: A Tensor. Support 1D ~ 8D. Must be one of the following types:bfloat16, float16, float32.
* @li x: A Tensor of the same type, shape and format as "dy".
* @li y: A Tensor of the same type, shape and format as "dy" . \n

* @par Outputs:
* z: A Tensor. Has the same type, format and shape as "dy".
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator GeluGrad.
* @attention Constraints:
* In Ascend 950 AI Processor, inputs and outputs support broadcasting.
*/
REG_OP(GeluGrad)
    .INPUT(dy, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(z, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(GeluGrad)

/**
    * @brief GeluMul divides the input tensor into left and right tensors x1 and x2 based on the last dimension,
    * performs GELU calculation on x1 on the left, and multiplies the calculation result by x2. \n

    * @par Inputs:
    * x: A tensor of type float, float16 or bfloat16. Shape support 2D ~ 8D.
    * The format must be ND.

    * @par Attributes:
    * approximate: A optional string. The GELU approximation algorithm to use: 'none' or 'tanh', default is 'none'.

    * @par Outputs:
    * y: A tensor has the same type and format as "x".
    * Other dimensions of its shape are the same as those of "x".
    * The value of the last dimension is half the value of the last dimension of "x". \n
    */
    REG_OP(GeluMul)
        .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
        .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
        .ATTR(approximate, String, "none")
        .OP_END_FACTORY_REG(GeluMul)

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
* @brief The GELUV2 activation function is x*Φ(x),
* where Φ(x) the standard Gaussian cumulative distribution function.

* @par Inputs:
* One input, including:
* x: A Tensor. Must be one of the following types: bfloat16, float16, float32.

* @par Outputs:
* y: A Tensor. Has the same type as "x".

* @par Attributes:
* approximate: A optional string. The gelu approximation algorithm to use: 'none' or 'tanh', default is 'none'.

* @par Third-party framework compatibility:
* Compatible with the Pytorch operator Gelu.
*/
REG_OP(GeluV2)
    .INPUT(x, "T")
    .OUTPUT(y, "T")
    .DATATYPE(T, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .ATTR(approximate, String, "none")
    .OP_END_FACTORY_REG(GeluV2)

/**
* @brief Activation function called Gated Linear Unit, calculate result by input ND tensor x and integer dim.
*
* @par Inputs:
* x: ND input tensor. Must be one of the following types: float16, float32, bfloat16.
*
* @par Attributes:
* dim: the dimension will be chunked into halves, default to be -1, the dimension itself must be even.
*
* @par Outputs:
* y: A Tensor. Has the same type as "x".

* @par Third-party framework compatibility
* Compatible with caffe correlation custom operator.
*/

REG_OP(GLU)
      .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
      .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
      .ATTR(dim, Int, -1)
      .OP_END_FACTORY_REG(GLU)

/**
*@brief Computes the gradient for the hard_swish of "self" .
*calculating formula:
*@code{.c}
*gradSelf_{i} = begin{cases}
* 0, self_{i} <= -3, 
* self_{i} / 3 + 0.5, -3 < self_{i} < 3,
* 1, self_{i} >= 3
* end{cases}
*
* out_{i} = gradOutput_{i} times gradSelf_{i}
*@endcode
*
*@par Inputs:
*Two inputs, including:
* @li gradOutput: A tensor. Must be one of the following types: float16, float32, bfloat16
* @li self: A tensor with the same type as "gradOutput" . \n
*@par Outputs:
* out: A tensor with the same type as "gradOutput".
*@par Third-party framework compatibility
* Compatible with the PyTorch operator HardSwishGrad after v2.8.
*/
REG_OP(HardSwishGradV2)
    .INPUT(gradOutput, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(self, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(out, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(HardSwishGradV2)

/**
* @brief Apply the heaviside step function element-wise to the input. \n

* @par Inputs:
* @li input: A tensor of type float32, float16 or bfloat16. Shape support 0D ~ 8D.
* The format must be ND.
* @li values: A tensor of type float32, float16 or bfloat16. Shape support 0D ~ 8D.
* The format must be ND.

* @par Outputs:
* output: A tensor has the same type, shape and format as "input". \n
*/
REG_OP(Heaviside)
    .INPUT(input, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(values, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(output, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(Heaviside)

/**
*@brief Computes hyperbolic tangent of "x" element-wise .
    
*@par Inputs:
* One input:
* x: An ND tensor. support 1D ~ 8D. Must be one of the following types:
* float16, float32, bfloat16.
*
*@par Outputs:
* y: A Tensor. Has the same type as "x" .
*
*@par Third-party framework compatibility
* Compatible with TensorFlow operator Mish.
*/
    
REG_OP(Mish)
    .INPUT(x, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16 }))
    .OUTPUT(y, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16 }))
    .OP_END_FACTORY_REG(Mish)

/**
* @brief Computes rectified linear: "max(x, 0)".
*
* @par Inputs:
* x: An ND or 5HD tensor. support 1D ~ 8D. Must be one of the following types:
* float32, float64, int32, uint8, int16, int8, int64, uint16, float16, qint8, bfloat16.
*
* @par Outputs:
* y: A tensor. Has the same type as "x".
*
* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator Relu.
* @li Compatible with the Caffe operator ReLULayer.
*
*/
REG_OP(Relu)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE,
                          DT_INT8, DT_INT32, DT_INT16, DT_INT64,
                          DT_UINT8, DT_UINT16, DT_QINT8, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE,
                           DT_INT8, DT_INT32, DT_INT16, DT_INT64,
                           DT_UINT8, DT_UINT16, DT_QINT8, DT_BF16}))
    .OP_END_FACTORY_REG(Relu)

/**
* @brief Computes rectified linear gradients for a ReLU operation.
*  Gradients/features support broadcasting operations. \n
*
* @par Inputs:
* Two inputs, including:
* @li gradients: A tensor of type RealNumberType. The backpropagated gradients to the corresponding Relu operation .
* @li features: A tensor of type RealNumberType. The features passed as input to the corresponding Relu operation . \n
*
* @par Outputs:
* backprops: A tensor of type RealNumberType . \n
*
* @attention Constraints:
* The corresponding Relu operator needs to be called before using this operator on the network . \n
*
* @par Third-party framework compatibility
* Compatible with TensorFlow operator ReluGrad.
*/
REG_OP(ReluGrad)
    .INPUT(gradients, TensorType::RealNumberType())
    .INPUT(features, TensorType::RealNumberType())
    .OUTPUT(backprops, TensorType::RealNumberType())
    .OP_END_FACTORY_REG(ReluGrad)

/**
 * @brief Computes rectified linear gradients for a ReLU operation.
 *
 * @par Inputs:
 * Two inputs, including:
 * @li gradients: A tensor of input gradient. Indicates the gradient of backpropagation. Supported type:
 * TensorType::RealNumberType(). \n In Ascend 950 AI Processor, support ND format. Others support 5D, format must be
 * NC1HWC0.
 * @li mask: A tensor of input mask. Indicates the positive output. Must be the following types: uint8, uint1. \n
 * In Ascend 950 AI Processor, support ND format. Others support 5D, format must be NC1HWC0.
 *
 * @par Outputs:
 * backprops: A tensor of output. Indicates the backpropagation result.
 * When gradient is greater than 0, backprops is the value of gradient.
 * When the value of gradient is less than or equal to 0, the value of backprops is 0.
 * Must have the same type, format and shape as "gradients". \n
 *
 * @attention Constraints:
 * The corresponding Relu operator needs to be called before using this operator on the network. \n
 */
REG_OP(ReluGradV2)
    .INPUT(gradients, TensorType::RealNumberType())
    .INPUT(mask, TensorType({DT_UINT8, DT_UINT1}))
    .OUTPUT(backprops, TensorType::RealNumberType())
    .OP_END_FACTORY_REG(ReluGradV2)

/**
 * @brief Computes rectified linear: "max(x, 0)".
 *
 * @par Inputs:
 * x: A tensor. Must be one of the following types: float32, float16, double, int8, int32, int16, int64, uint8, uint16,
 * qint8, bfloat16.\n In Ascend 950 AI Processor, support ND format. Others support 4D, format must be one of [NHWC,
 * NCHW, HWCN].
 *
 * @par Outputs:
 * @li y: A tensor with the same dtype and shape as the "x".
 * @li mask: A tensor of value "1" (for x > 0) or "0" (for x <= 0). Type is uint8 or uint1.
 *
 * @par Third-party framework compatibility
 * Incompatible with TensorFlow or Caffe.
 *
 * @attention Constraints:
 * The last dimension of "x" must be divisible by 8.
 */
REG_OP(ReluV2)
    .INPUT(
        x, TensorType(
               {DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT8, DT_INT32, DT_INT16, DT_INT64, DT_UINT8, DT_UINT16, DT_QINT8,
                DT_BF16}))
    .OUTPUT(
        y, TensorType(
               {DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT8, DT_INT32, DT_INT16, DT_INT64, DT_UINT8, DT_UINT16, DT_QINT8,
                DT_BF16}))
    .OUTPUT(mask, TensorType({DT_UINT8, DT_UINT1}))
    .OP_END_FACTORY_REG(ReluV2)

/**
* @brief Compute sigmoid of "x" element-wise .

* @par Inputs:
* A Tensor of type complex64, complex128, bfloat16, float16, float32 or double . \n

* @par Outputs:
* A Tensor. Has the same type as "x" . \n

* @see Relu()

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Sigmoid.
*/
REG_OP(Sigmoid)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Sigmoid)

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
    * @brief SquaredRelu first applies ReLU to the input tensor x, and then squares the result of the ReLU operation.
    *
    * @par Inputs:
    * x: A tensor of type float, bf16 or bfloat16. Shape support 0D ~ 8D.
    * The format must be ND.
    *
    * @par Outputs:
    * y: A Tensor with the same type, shape, format as "x". \n
    */
    REG_OP(SquaredRelu)
        .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
        .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
        .OP_END_FACTORY_REG(SquaredRelu)

/**
    * @brief Compute the SwiGlu,
    * where the activations function in GLU is Swish.

    * @par Inputs:
    * One input, including:
    * @x: A Tensor. Must be one of the following types: bfloat16, float16, float32.

    * @par Outputs:
    * one output, including:
    * @y: A Tensor. Must be one of the following types: bfloat16, float16, float32.

    * @par Attributes:
    * one attribute, including:
    * @li dim: A optional int. The dimension to be split, default is -1.

    * @par Third-party framework compatibility:
    * New operator SwiGlu.
    */
    REG_OP(SwiGlu)
        .INPUT(x, "T")
        .OUTPUT(y, "T")
        .DATATYPE(T, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
        .ATTR(dim, Int, -1)
        .OP_END_FACTORY_REG(SwiGlu)

/**
    * @brief Compute the SwiGluGrad,
    * where the activations function in GLU is SwishGrad.

    * @par Inputs:
    * two input, including:
    * @li y_grad: A Tensor, which is the output gradient of forward operator and which \n
    * has the same shape as "x" except for the dimension specified by the "dim" parameter. \n
    * The dimension size specified by "dim" is half of the corresponding dimension of x. \n
    * Must be one of the following types: bfloat16, float16, float32.
    * @li x: A Tensor. Must be one of the following types: bfloat16, float16, float32.

    * @par Outputs:
    * one Output, including:
    * x_grad: A Tensor, which is the gradient of x and has the same shape as "x". \n
    * Must be one of the following types: bfloat16, float16, float32.

    * @par Attributes:
    * one attribute, including:
    * @li dim: A optional int. The dimension to be split, default is -1.

    * @par Third-party framework compatibility:
    * New operator SwiGluGrad.
    */
    REG_OP(SwiGluGrad)
        .INPUT(y_grad, "T")
        .INPUT(x, "T")
        .OUTPUT(x_grad, "T")
        .DATATYPE(T, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
        .ATTR(dim, Int, -1)
        .OP_END_FACTORY_REG(SwiGluGrad)

/**
 * @brief Apply abs operation for each tensor in a tensor list in manner of element-wise
 * @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the abs value of the x
 */
REG_OP(ForeachAbs)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachAbs)

/**
 * @brief Apply arc cos operation for each tensor in a tensor list in manner of element-wise
 * @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the arc cos value of the x
 */
REG_OP(ForeachAcos)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachAcos)

/**
 * @brief Apply add operation for each tensor in a tensor list with each tensor in another
 * tensor list in manner of element-wise
 * @par Inputs:
 * Three inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Another tensor list containing multiple tensors
 * @li alpha: The elements in x2 should perform multipy with alpha which is a scalar
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are add by the scalars in scalar list
 */
REG_OP(ForeachAddList)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(alpha, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachAddList)

/**
 * @brief Apply add operation for each tensor in tensor list with a scalar in manner of element-wise
 * @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors, the length cannot exceed 50,
 *        the dtype can be BFloat16, Float16, Int32 or Float32, and the format support ND.
 * @li scalar: A scalar in form of tensor with only one element,
 *        the dtype can be Float16, Int32 or Float32, and the format supports ND.
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are add by the scalar,
 *        has the same length, dtype and format as input "x".
 */
REG_OP(ForeachAddScalar)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachAddScalar)

/**
 * @brief Apply add operation for each tensor in tensor list with a list of scalar in manner
 * of element-wise the number of tensors in tensor list shall be equal to the number of scalars
 * in scalar list
 * @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalars: A scalar list in form of tensor with only multiple elements
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are add by the scalars in scalar list
 */
REG_OP(ForeachAddScalarList)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_INT64}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachAddScalarList)

/**
 * @brief Apply AddcDiv operation for each tensor in tensor list with a scalar in scalar list
 *  or a tensor in manner of element-wise
 * @par Inputs:
 * Three inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Second tensor list containing multiple tensors
 * @li x3: Third tensor list containing multiple tensors
 * @li scalar: A scalar list or a tensor
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are AddcDiv with the scalars
 */
REG_OP(ForeachAddcdivList)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(x3, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachAddcdivList)

/**
 * @brief Apply AddcDiv operation for each tensor in tensor list with a scalar in manner
 * of element-wise
 * @par Inputs:
 * Three inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Second tensor list containing multiple tensors
 * @li x3: Third tensor list containing multiple tensors
 * @li scalar: A scalar value
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are AddcDiv with the scalar
 */
REG_OP(ForeachAddcdivScalar)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(x3, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachAddcdivScalar)

/**
 * @brief Apply AddcDiv operation for each tensor in tensor list with a list of scalar in manner
 * of element-wise the number of tensors in tensor list shall be equal to the number of scalars
 * in scalar list
 * @par Inputs:
 * Four inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Second tensor list containing multiple tensors
 * @li x3: Third tensor list containing multiple tensors
 * @li scalars: A list of scalar value
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are AddcDiv with the scalar
 */
REG_OP(ForeachAddcdivScalarList)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(x3, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachAddcdivScalarList)

/**
 * @brief Apply AddcMul operation for each tensor in tensor list with a scalar in scalar list
 *  or a tensor in manner of element-wise
 * @par Inputs:
 * Three inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Second tensor list containing multiple tensors
 * @li x3: Third tensor list containing multiple tensors
 * @li scalar: A scalar list or a tensor
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are AddcMul with the scalars
 */
REG_OP(ForeachAddcmulList)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_INPUT(x3, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachAddcmulList)

/**
 * @brief Apply AddcMul operation for each tensor in tensor list with a scalar in manner
 * of element-wise
 * @par Inputs:
 * Three inputs:
 * @li x1: A tensor list containing multiple tensors, the length cannot exceed 50,
 *         the dtype can be BFloat16, Float16, Int32 or Float32, and the format support ND.
 * @li x2: Second tensor list containing multiple tensors, must has the same length, dtype and format as input "x1".
 * @li x3: Third tensor list containing multiple tensors, must has the same length, dtype and format as input "x1".
 * @li scalar: A scalar in form of tensor with only one element,
 *        the dtype can be Float16, Int32 or Float32, and the format supports ND.
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are AddcMul with the scalar,
 *        has the same length, dtype and format as input "x1".
 */
REG_OP(ForeachAddcmulScalar)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_INPUT(x3, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachAddcmulScalar)

/**
 * @brief Apply AddcMul operation for each tensor in tensor list with a list of scalar in manner
 * of element-wise the number of tensors in tensor list shall be equal to the number of scalars
 * in scalar list
 * @par Inputs:
 * Four inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Second tensor list containing multiple tensors
 * @li x3: Third tensor list containing multiple tensors
 * @li scalars: A list of scalars value
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are AddcMul with the scalar
 */
REG_OP(ForeachAddcmulScalarList)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_INPUT(x3, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachAddcmulScalarList)

/**
 * @brief Apply arc sin operation for each tensor in a tensor list in manner of element-wise
 * @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the arc sin value of the x
 */
REG_OP(ForeachAsin)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachAsin)

/**
 * @brief Apply arc tan operation for each tensor in a tensor list in manner of element-wise
 * @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the arc tan value of the x
 */
REG_OP(ForeachAtan)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachAtan)

/**
 * @brief Apply copy operation for each tensor in a tensor list in manner of element-wise
 * @par Inputs:
 * One inputs:
 * x: A tensor list containing multiple tensors. The data type can only be
 * float16, float, bfloat16, int8, int16, int32, uint8, uint16, uint32, int64, float64, bool.
 * The format support ND. Shape support 1D ~ 8D.
 * @par Outputs:
 * y: A tensor list which store the tensors whose value are the copy value of the x.
 * The data type can only be float16, float, bfloat16, int8, int16, int32, uint8, uint16, uint32, int64, float64, bool.
 * The format support ND. Shape support 1D ~ 8D. Has the same dtype adn shape as input "x".
 */
REG_OP(ForeachCopy)
    .DYNAMIC_INPUT(
        x, TensorType(
               {DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_UINT32, DT_INT64,
                DT_DOUBLE, DT_BOOL}))
    .DYNAMIC_OUTPUT(
        y, TensorType(
               {DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_UINT32, DT_INT64,
                DT_DOUBLE, DT_BOOL}))
    .OP_END_FACTORY_REG(ForeachCopy)

/**
 * @brief Apply cos operation for each tensor in a tensor list in manner of element-wise
 * @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the cos value of the x
 */
REG_OP(ForeachCos)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachCos)

/**
 * @brief Apply cosh operation for each tensor in a tensor list in manner of element-wise
 * @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the cosh value of the x
 */
REG_OP(ForeachCosh)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachCosh)

/**
 * @brief Apply div operation for each tensor in a tensor list with each tensor in another
 * tensor list in manner of element-wise
 * @par Inputs:
 * Two inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Another tensor list containing multiple tensorsr
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are div by the scalars in scalar list
 */
REG_OP(ForeachDivList)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachDivList)

/**
 * @brief Apply div operation for each tensor in tensor list with a scalar in manner of element-wise
 * @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalar: A scalar in form of tensor with only one element, the shape must be (1,)
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are div by the scalar
 */
REG_OP(ForeachDivScalar)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachDivScalar)

/**
 * @brief Apply division operation for each tensor in tensor list with a list of scalar in manner
 * of element-wise the number of tensors in tensor list shall be equal to the number of scalars
 * in scalar list.
 * @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors, the length cannot exceed 50,
 *        the dtype can be BFloat16, Float16 or Float32, and the format support ND.
 * @li scalars: A scalar list in form of tensor with only multiple elements,
 *        the dtype can be Float32, and the format supports ND.
 * @par Outputs:
 * A tensor list which store the tensors whose value are x divide by the scalars in scalar list,
 *       has the same length, dtype and format as input "x".
 */
REG_OP(ForeachDivScalarList)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(scalars, TensorType({DT_FLOAT}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachDivScalarList)

/**
 * @brief Apply erf operation for each tensor in a tensor list in manner of element-wise
 * @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the erf value of the x
 */
REG_OP(ForeachErf)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachErf)

/**
 * @brief Apply erfc operation for each tensor in a tensor list in manner of element-wise
 * @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the erfc value of the x
 */
REG_OP(ForeachErfc)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachErfc)

/**
 * @brief Apply exp operation for each tensor in a tensor list in manner of element-wise
 * @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the exp value of the x
 */
REG_OP(ForeachExp)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachExp)

/**
 * @brief Apply expm1 operation for each tensor in a tensor list in manner of element-wise
 * @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the expm1 value of the x
 */
REG_OP(ForeachExpm1)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachExpm1)

/**
 * @brief Apply lerp operation for each tensor in tensor list with tensors in another tensor list and
 * an additonal tensor in manner of element-wise
 * @par Inputs:
 * Three inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Another tensor list containing multiple tensors
 * @li weights: A tensor contain multiple elements
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are produced by lerp
 */
REG_OP(ForeachLerpList)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(weight, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachLerpList)

/**
 * @brief Apply lerp operation for each tensor in tensor list with tensors in another tensor list and
 * a scalar in manner of element-wise
 * @par Inputs:
 * Three inputs:
 * @li x1: A tensor list containing multiple tensors, the length cannot exceed 50,
 *         the dtype can be BFloat16, Float16 or Float32, and the format support ND.
 * @li x2: Another tensor list containing multiple tensors, must has the same length, dtype and format as input "x1".
 * @li weight: A scalar in form of tensor with only one element,
 *        the dtype can be Float32, and the format supports ND.
 * @par Outputs:
 *     y: A tensor list which store the tensors whose value are produced by lerp,
 *        has the same length, dtype and format as input "x1".
 */
REG_OP(ForeachLerpScalar)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(weight, TensorType({DT_FLOAT}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachLerpScalar)

/**
 * @brief Apply log operation for each tensor in a tensor list in manner of element-wise
 * @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the log value of the x
 */
REG_OP(ForeachLog)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachLog)

/**
 * @brief Apply log10 operation for each tensor in a tensor list in manner of element-wise
 * @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the log10 value of the x
 */
REG_OP(ForeachLog10)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachLog10)

/**
 * @brief Apply log1p operation for each tensor in a tensor list in manner of element-wise
 * @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the log1p value of the x
 */
REG_OP(ForeachLog1p)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachLog1p)

/**
 * @brief Apply log2 operation for each tensor in a tensor list in manner of element-wise
 * @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the log2 value of the x
 */
REG_OP(ForeachLog2)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachLog2)

/**
 * @brief Apply maximum operation for each tensor in a tensor list with each tensor in another
 * tensor list in manner of element-wise
 * @par Inputs:
 * Two inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Another tensor list containing multiple tensorsr
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are maximum with the scalars in scalar list
 */
REG_OP(ForeachMaximumList)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachMaximumList)

/**
 * @brief Apply maximum operation for each tensor in tensor list with a scalar in manner of element-wise
 * @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalar: A scalar in form of tensor with only one element, the shape must be (1,)
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are maximum with the scalar
 */
REG_OP(ForeachMaximumScalar)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachMaximumScalar)

/**
 * @brief Apply maximum operation for each tensor in tensor list with a list of scalar in manner
 * of element-wise the number of tensors in tensor list shall be equal to the number of scalars
 * in scalar list
 * @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalars: A scalar list in form of tensor with only multiple elements
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are maximum with the scalars in scalar list
 */
REG_OP(ForeachMaximumScalarList)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_INT64}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachMaximumScalarList)

/**
 * @brief Apply minimum operation for each tensor in tensor list with a list of scalar in manner
 * of element-wise the number of tensors in tensor list shall be equal to the number of scalars
 * in scalar list
 * @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalars: A scalar list in form of tensor with only multiple elements
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are minimum with the scalars in scalar list
 */
REG_OP(ForeachMinimumList)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachMinimumList)

/**
 * @brief Apply minimum operation for each tensor in tensor list with a scalar in manner of element-wise
 * @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalar: A scalar in form of tensor with only one element, the shape must be (1,)
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are minimum with the scalar
 */
REG_OP(ForeachMinimumScalar)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachMinimumScalar)

/**
 * @brief Apply minimum operation for each tensor in tensor list with a list of scalar in manner
 * of element-wise the number of tensors in tensor list shall be equal to the number of scalars
 * in scalar list
 * @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalars: A scalar list in form of tensor with only multiple elements
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are minimum with the scalars in scalar list
 */
REG_OP(ForeachMinimumScalarList)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_INT64}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachMinimumScalarList)

/**
 * @brief Apply mul operation for each tensor in a tensor list with each tensor in another
 * tensor list in manner of element-wise
 * @par Inputs:
 * Two inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Another tensor list containing multiple tensorsr
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are mul by the scalars in scalar list
 */
REG_OP(ForeachMulList)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachMulList)

/**
 * @brief Apply mul operation for each tensor in tensor list with a scalar in manner of element-wise
 * @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors, the length cannot exceed 50,
 *        the dtype can be BFloat16, Float16, Int32 or Float32, and the format support ND.
 * @li scalar: A scalar in form of tensor with only one element,
 *        the dtype can be Float16, Int32 or Float32, and the format supports ND.
 *        When the data type of x is FLOAT32 or INT32, the data type of scalar must match that of x.
 *        When the data type of x is BFLOAT16, scalar supports the FLOAT32 data type.
 *        When the data type of x is FLOAT16, scalar supports both FLOAT16 and FLOAT32 data types.
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are mul by the scalar,
 *        has the same length, dtype and format as input "x".
 */
REG_OP(ForeachMulScalar)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachMulScalar)

/**
 * @brief Apply mul operation for each tensor in tensor list with a list of scalar in manner
 * of element-wise the number of tensors in tensor list shall be equal to the number of scalars
 * in scalar list
 * @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors,
 * maximum length of x is 50
 * @li scalars: A scalar list in form of tensor with only multiple elements
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are mul by the scalars in scalar list
 */
REG_OP(ForeachMulScalarList)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_INT64}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachMulScalarList)

/**
 * @brief Apply neg operation for each tensor in a tensor list in manner of element-wise
 * @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the neg value of the x
 */
REG_OP(ForeachNeg)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachNeg)

/**
 * @brief Detect whether there is Inf or Nan in scaled_grads, set found_inf to 1 if it exists,
 * and do not operate on found_inf if it does not. Finally, multiply all values of scaled_grads by inv_scale
 * @par Inputs:
 * Three inputs:
 * @li scaled_grads: A tensor list containing multiple tensors, format supports ND, can be float16, float, bfloat16,
 * maximum length of scaled_grads is 256,
 * meanwhile, this value is also an output, store the value multiplied by inv_scale.
 * @li found_inf: A single-element float tensor to which 1.0 will be written if any scaled_grad contain infs/nans,
 * with only one element, format supports ND, must be float,
 * meanwhile, this value is also an output, indicating whether there is Inf or Nan present.
 * @li inv_scale: A single-element float tensor by which scaled_grads are currently multiplied,
 * with only one element, format supports ND, must be float.
 */
REG_OP(ForeachNonFiniteCheckAndUnscale)
    .DYNAMIC_INPUT(scaled_grads, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(found_inf, TensorType({DT_FLOAT}))
    .INPUT(inv_scale, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(ForeachNonFiniteCheckAndUnscale)

/**
 * @brief Apply norm operation for each tensor in a tensor list in manner of element-wise
 * @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors. Shape dimension cannot be greater than 8 dimensions.
 * Format supports ND, and data type must be float16, float32 or bfloat16. The list supports a maximum length of 256.
 * @li scalar: A scalar with one element, indicating the order of norm.
 * Format supports ND, and data type must be int64, float32.
 * @par Outputs:
 * y: A tensor list which store the tensors whose value are the norm value of the x.
 * Shape dimension cannot be greater than 8 dimensions, and shapesize should be 1.
 * Format supports ND, and data type must be float16, float32 or bfloat16. The list supports a maximum length of 256.
 */
REG_OP(ForeachNorm)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_INT64}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachNorm)

/**
 * @brief Apply power operation for a scalar with each tensor in a tensor list
 * in manner of element-wise
 * @par Inputs:
 * Two inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Another tensor list containing multiple tensorsr
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are power with the scalars in scalar list
 */
REG_OP(ForeachPowList)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachPowList)

/**
 * @brief Apply power operation for each tensor in tensor list with a scalar in manner of element-wise
 * @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalar: A scalar in form of tensor with only one element, the shape must be (1,)
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are power with the scalar
 */
REG_OP(ForeachPowScalar)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_INT64}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachPowScalar)

/**
 * @brief Apply power operation for each tensor in tensor list with a scalar in manner
 * of element-wise the number of tensors in tensor list shall be equal to the number of scalars
 * in scalar list
 * @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalar: A scalar
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are power with the scalars in scalar list
 */
REG_OP(ForeachPowScalarAndTensor)
    .INPUT(scalar, TensorType({DT_FLOAT, DT_INT64}))
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachPowScalarAndTensor)

/**
 * @brief Apply power operation for each tensor in tensor list with a list of scalar in manner
 * of element-wise the number of tensors in tensor list shall be equal to the number of scalars
 * in scalar list
 * @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalars: A scalar list in form of tensor with only multiple elements
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are power with the scalars in scalar list
 */
REG_OP(ForeachPowScalarList)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_INT64}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachPowScalarList)

/**
 * @brief Apply reciprocal operation for each tensor in a tensor list in manner of element-wise
 * @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the reciprocal value of the x
 */
REG_OP(ForeachReciprocal)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachReciprocal)

/**
* @brief round off number foreach element in each tensor in tesnorlist, this is an in-place operation.
* @par Inputs:
 * Two inputs
 * @li x: A tensor list containing multiple tensors, can be float16, float.
 * @li roundMode: mode of round off which currently supports 2(floor) and 3(ceil).
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are produced by round off
*/
REG_OP(ForeachRoundOffNumber)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(roundMode, TensorType({DT_INT8}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachRoundOffNumber)

/**
 * @brief Apply sigmoid operation for each tensor in a tensor list in manner of element-wise
 * @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the sigmoid value of the x
 */
REG_OP(ForeachSigmoid)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachSigmoid)

/**
 * @brief Apply sign operation for each tensor in a tensor list in manner of element-wise
 * @par Inputs:
 * One inputs:
 * x: A tensor list containing multiple tensors. The data type can only be float16, float, int32, int8, int64, bfloat16.
 * The format support ND. Shape support 1D ~ 8D.
 * @par Outputs:
 * y: A tensor list which store the tensors whose value are the sign value of the x.
 * The format support ND. The data type can only be float16, float, int32, int8, int64, bfloat16.
 * Shape support 1D ~ 8D. The data type and shape are same as input "x".
 */
REG_OP(ForeachSign)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8, DT_INT64, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8, DT_INT64, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachSign)

/**
 * @brief Apply cos operation for each tensor in a tensor list in manner of element-wise
 * @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the cos value of the x
 */
REG_OP(ForeachSin)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachSin)

/**
 * @brief Apply sinh operation for each tensor in a tensor list in manner of element-wise
 * @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the sinh value of the x
 */
REG_OP(ForeachSinh)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachSinh)

/**
 * @brief Apply sqrt operation for each tensor in a tensor list in manner of element-wise
 * @par Inputs:
 * One inputs:
 * x: A tensor list containing multiple tensors. Format supports ND. The type support float16, float, bfloat16.
 * Maximum length of x is 50.
 * @par Outputs:
 * y: A tensor list which store the tensors whose value are the sqrt value of the x. Format supports ND.
 * The type support float16, float, bfloat16 and is consistent with x. Maximum length of y is 50.
 */
REG_OP(ForeachSqrt)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachSqrt)

/**
 * @brief Apply sub operation for each tensor in a tensor list with each tensor in another
 * tensor list in manner of element-wise
 * @par Inputs:
 * Three inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Another tensor list containing multiple tensors
 * @li alpha: The elements in x2 should perform multipy with alpha which is a scalar
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are sub by the scalars in scalar list
 */
REG_OP(ForeachSubList)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(alpha, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachSubList)

/**
 * @brief Apply sub operation for each tensor in tensor list with a scalar in manner of element-wise
 * @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalar: A scalar in form of tensor with only one element, the shape must be (1,)
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are sub by the scalar
 */
REG_OP(ForeachSubScalar)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachSubScalar)

/**
 * @brief Apply sub operation for each tensor in tensor list with a list of scalar in manner
 * of element-wise the number of tensors in tensor list shall be equal to the number of scalars
 * in scalar list
 * @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalars: A scalar list in form of tensor with only multiple elements
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are sub by the scalars in scalar list
 */
REG_OP(ForeachSubScalarList)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_INT64}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachSubScalarList)

/**
 * @brief Apply tan operation for each tensor in a tensor list in manner of element-wise
 * @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the tan value of the x
 */
REG_OP(ForeachTan)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachTan)

/**
 * @brief Apply tanh operation for each tensor in a tensor list in manner of element-wise
 * @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the tanh value of the x
 */
REG_OP(ForeachTanh)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachTanh)

/**
 * @brief Apply zero operation for each tensor in a tensor list in manner of element-wise
 * @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
 */
REG_OP(ForeachZeroInplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachZeroInplace)

/**
* @brief multi-scale deformable attention grad.
*
* @par Inputs:
* @li value: A Tensor. Must be one of the following types: float32.
* @li value_spatial_shapes: A Tensor. Must be one of the following types: int32.
* @li value_level_start_index: A Tensor. Must be one of the following types: int32.
* @li sampling_locations: A Tensor. Must be one of the following types: float32.
* @li attention_weights: A Tensor. Must be one of the following types: float32.
* @li grad_output: A Tensor. Must be one of the following types: float32.
*
* @par Outputs:
* grad_value: A Tensor. Must be one of the following types: float32.
* grad_sampling_locations: A Tensor. Must be one of the following types: float32.
* grad_attention_weights: A Tensor. Must be one of the following types: float32.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(MultiScaleDeformableAttentionGrad)
    .INPUT(value, TensorType({DT_FLOAT}))
    .INPUT(value_spatial_shapes, TensorType({DT_INT32}))
    .INPUT(value_level_start_index, TensorType({DT_INT32}))
    .INPUT(sampling_locations, TensorType({DT_FLOAT}))
    .INPUT(attention_weights, TensorType({DT_FLOAT}))
    .INPUT(grad_output, TensorType({DT_FLOAT}))
    .OUTPUT(grad_value, TensorType({DT_FLOAT}))
    .OUTPUT(grad_sampling_locations, TensorType({DT_FLOAT}))
    .OUTPUT(grad_attention_weights, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(MultiScaleDeformableAttentionGrad)

/**
* @brief multi-scale deformable attention.
*
* @par Inputs:
* @li value: A Tensor. Must be one of the following types: float16, float32.
* @li value_spatial_shapes: A Tensor. Must be one of the following types: int32, int64.
* @li value_level_start_index: A Tensor. Must be one of the following types: int32, int64.
* @li sampling_locations: A Tensor. Must be one of the following types: float16, float32.
* @li attention_weights: A Tensor. Must be one of the following types: float16, float32.
*
* @par Outputs:
* output: A Tensor. Must be one of the following types: float16, float32.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(MultiScaleDeformableAttnFunction)
    .INPUT(value, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(value_spatial_shapes, TensorType({DT_UINT64, DT_INT32}))
    .INPUT(value_level_start_index, TensorType({DT_UINT64, DT_INT32}))
    .INPUT(sampling_locations, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(attention_weights, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(output, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(MultiScaleDeformableAttnFunction)

/**
* @brief Calculates the reversed outputs of the function "embedding". \n

* @par Inputs:
* Four inputs, including:
* @li weight: A required Tensor. A mutable Tensor of word grad. Must be one of the following types: float16, float32,
bfloat16.
* @li indices: A required Tensor. A mutable word index Tensor. Must be one of the following types: int32, int64.
* @li offsets: A optional Tensor. A mutable word index Tensor. Must be one of the following types: int32, int64.
* @li per_sample_weights: A optional Tensor. to indicate all weights should be taken to be 1. Must be one of the
following types: float16, float32, bfloat16.
*     If specified, per_sample_weights must have exactly the same shape as input
*     and is treated as having the same offsets, if those are not None.
*     Only supported for mode='sum'.\n

* @par Attributes:
* @li mode: An string attr which use "sum"``, ``"mean"`` or ``"max"``. Specifies the way to reduce the bag. Defaults to
"mean".\n
* @li padding_idx: An int attr judge which word to fill zeros. Defaults to "-1". \n

* @li scale_grad_by_freq: An optional bool. Defaults to "False".
*     If "True", "grad_weight" will be scale by word_frequency.
*     If "False", "grad_weight" will not be scale by word_frequency. \n
* @li sparse: An optional bool. Defaults to "False". if True, gradient w.r.t.attr weight matrix will be a sparse tensor.
\n
* @li include_last_offset: An optional bool. Defaults to "False". if True, attr offsets has one additional element,
where the last element
*     is equivalent to the size of indices. This matches the CSR format. \n

* @par Outputs:
* four outputs: \n
* y: A mutable output Tensor of new word grad has the same type as "grads". \n
* offset2bag:A Tensor. Must be one of the following types: int32, int64. \n
* bag_size:A Tensor. Must be one of the following types: int32, int64. \n
* max_indices:A Tensor. Must be one of the following types: int32, int64. \n
* @par Third-party framework compatibility
* Compatible with the Pytorch operator EmbeddingBag.
*/
REG_OP(EmbeddingBag)
    .INPUT(weight, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(indices, TensorType({DT_INT32, DT_INT64}))
    .OPTIONAL_INPUT(offsets, TensorType({DT_INT32, DT_INT64}))
    .OPTIONAL_INPUT(per_sample_weights, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(offset2bag, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(bag_size, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(max_indices, TensorType({DT_INT32, DT_INT64}))
    .ATTR(mode, String, "mean")
    .ATTR(scale_grad_by_freq, Bool, false)
    .ATTR(sparse, Bool, false)
    .ATTR(include_last_offset, Bool, false)
    .ATTR(padding_idx, Int, -1)
    .OP_END_FACTORY_REG(EmbeddingBag)

/**
* @brief Calculates the reversed outputs of the function "embedding". \n

* @par Inputs:
* Two inputs, including:
* @li grad: A mutable Tensor of word grad. Must be one of the following types:
*     float32, float16, bfloat16
* @li sort_indices: A mutable word index Tensor. Must be one of the following types:
*     int32, int64
* @li pos_idx: A mutable position of sort_indices in the grad Tensor of the int32 type.\n

* @par Attributes:
* @li num_weights: An int attr which use to judge how many words in dict. \n

* @li padding_idx: An int attr judge which word to fill zeros. Defaults to "-1". \n

* @li scale_grad_by_freq: An optional bool. Defaults to "False".
*     If "True", "grad_weight" will be scale by word_frequency.
*     If "False", "grad_weight" will not be scale by word_frequency. \n

* @par Outputs:
* y: A mutable output Tensor of new word grad has the same type as "grads". \n

* @par Restrictions:
* Warning: The grad of input last dim size should be less than 8192 bytes and the
*     sort_indices size should be larger than 32 for Ascend 950 AI Processors. \n
*/
REG_OP(EmbeddingDenseGradV2)
    .INPUT(grad, TensorType({DT_FLOAT32, DT_FLOAT16, DT_BF16})) /* "First operand." */
    .INPUT(sort_indices, TensorType({DT_INT32, DT_INT64}))      /* "Second operand." */
    .INPUT(pos_idx, TensorType({DT_INT32}))                     /* "Thrid operand." */
    .OUTPUT(y, TensorType({DT_FLOAT32, DT_FLOAT16, DT_BF16}))   /* "Result, has same element type as two inputs" */
    .REQUIRED_ATTR(num_weights, Int)
    .ATTR(padding_idx, Int, -1)
    .ATTR(scale_grad_by_freq, Bool, false)
    .OP_END_FACTORY_REG(EmbeddingDenseGradV2)

/**
* @brief Gather slices from "x" according to "indices" by corresponding axis, produces a output tensor
* with shape(x.shape[:axis]+indices.shape[batch_dims:]+x.shape[axis+1:]). When the impl_mode is set
* as "support out of bound index", if the indices data is out of bound, the corresponding results
* will be set as 0. Otherwise, an aic_error will occur.

* @par Inputs:
* @li x: A ND (Support 1D~8D) tensor. Must be one of the following types: complex128, complex64, float64, float32, float16,
*     int16, int32, int64, int8, qint16, qint32, qint8, quint16, quint8, uint16, uint32, uint64, uint8,
*     bool, FLOAT8_E5M2, FLOAT8_E8M0, FLOAT8_E4M3FN, string, bfloat16.
* @li indices: A ND (Support 1D~8D) tensor of type int32 or int64.
* @li axis: A Scalar with type as int32 or int64. Must be in the range [-rank(input_tensor), rank(input_tensor)).

* @par Attributes:
* @li batch_dims: An optional int which means the number of data to be deal with. Defaults to 0.
* @li is_preprocessed: An optional bool. Whether to preprocess, wihch is true means need to be preprocess and false means not. Defaults to false.
* @li negative_index_support: An optional bool, which is true means support index is negative, and false means not. Defaults to false.

* @par Outputs:
* y: A ND Tensor which has the same type as "x".

* @attention Constraints:
* @li Value in indices must be in range [0, x.shape[axis]).
* @li Default mode is HIGH_PERCISION.
      Only HIGH_PERCISION mode support negative index, and negative index in HIGH_PERFORMANCE mode may cause precision abnormal or aicore error.
* @li Batch_dims must be in the range [max(-rank(input_tensor),-rank(indices)), min(rank(input_tensor), rank(indices))).
* @li (batch_dims + rank(input_tensor)) % rank(input_tensor) must be less than or equal to (axis + rank(input_tensor)) % rank(input_tensor).
* @li The first batch_dims dimensions of params and indices are same.

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator GatherV2 .

*/
REG_OP(GatherV2)
    .INPUT(x, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64,
                          DT_INT8, DT_QINT16, DT_QINT32, DT_QINT8, DT_QUINT16, DT_QUINT8, DT_UINT16, DT_UINT32,
                          DT_UINT64, DT_UINT8, DT_BOOL, DT_STRING, DT_BF16, DT_FLOAT8_E5M2, DT_FLOAT8_E8M0, DT_FLOAT8_E4M3FN}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(axis, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64,
                          DT_INT8, DT_QINT16, DT_QINT32, DT_QINT8, DT_QUINT16, DT_QUINT8, DT_UINT16, DT_UINT32,
                          DT_UINT64, DT_UINT8, DT_BOOL, DT_STRING, DT_BF16, DT_FLOAT8_E5M2, DT_FLOAT8_E8M0, DT_FLOAT8_E4M3FN}))
    .ATTR(batch_dims, Int, 0)
    .ATTR(is_preprocessed, Bool, false)
    .ATTR(negative_index_support, Bool, false)
    .OP_END_FACTORY_REG(GatherV2)

/**
* @brief Returns a tensor containing the indices of all non-zero
* elements of input.

* @par Inputs:
* x: A Tensor. Must be one of the following types: float16, bfloat16, float32,
*    int32, int64, double, int8, uint8, int16, uint16, uint32, uint64, bool.
* @par Attributes:
* @li transpose: An optional attribute. Type is bool. Defaults to False.
* @li dtype: An optional attribute. Specifying the output data type.
*     Either "int32" or "int64". Default to "int64". \n
* @par Outputs:
* y: A Tensor. Must be one of the following types: int32, int64. \n

* @par Third-party framework compatibility
* Compatible with the PyTorch operator NonZero.
*/

REG_OP(NonZero)
    .INPUT(
        x, TensorType(
               {DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64,
                DT_UINT64, DT_BOOL, DT_BF16}))
    .OUTPUT(y, TensorType({DT_INT64, DT_INT32}))
    .ATTR(transpose, Bool, false)
    .ATTR(dtype, Type, DT_INT64)
    .OP_END_FACTORY_REG(NonZero)

/**
* @brief Repeat elements of input with copies of data along a specified dimension.
* @par Inputs:
* Two inputs:
* @li x: A Tensor with ND format. Support float, float16, bfloat16, int8, int16, int32, int64,
uint8, uint16, uint32, uint64, bool.
* @li repeats: A Tensor with 0-D / 1-D or a Scalar. Support int32, int64.

* @par Attributes:
* axis: An optional int32, specifying the axis to repeat. Defaults to 1000.

* @par Outputs:
* y: A Tensor, which is the same dtype as x.

* @attention Constraints:
* @li When "repeats" is 1-D tensor, the size of "repeats" must be 1 or x.size()[axis].
* @li "axis" must be within the rank of the input tensor.

* @par Third-party framework compatibility
* Compatible with the PyTorch operator RepeatInterleave.
*/
REG_OP(RepeatInterleave)
    .INPUT(x, TensorType::BasicType())
    .INPUT(repeats, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType::BasicType())
    .ATTR(axis, Int, 1000)
    .OP_END_FACTORY_REG(RepeatInterleave)

/**
* @brief Gradient op for RepeatInterleave op.
* @par Inputs:
* Two inputs:
* @li y_grad: A Tensor with any format. Support float, float16, bf16.
* @li repeats: A Tensor with dim = 1 or a Scalar. Support int32 or int64.

* @par Attributes:
* axis: An optional int32, specifying the axis to repeat. Defaults to -1.

* @par Outputs:
* x_grad: A Tensor, which is the same dtype as y_grad. Support float, float16, bf16.

* @attention Constraints:
* @li "axis" must be within the rank of the input tensor.
* @li The elements in "repeats" cannot all be zero.

* @par Third-party framework compatibility
* Compatible with the PyTorch operator the grad of RepeatInterleave.
*/
REG_OP(RepeatInterleaveGrad)
    .INPUT(y_grad, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(repeats, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(x_grad, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(axis, Int, -1)
    .OP_END_FACTORY_REG(RepeatInterleaveGrad)

/**
* @brief Uses "updates" to update tensor "var" by "indices". \n

* @par Inputs:
* Three inputs, including:
* @li var: An ND Tensor . \n
* Must be one of the following types: float32, float16, int16, int32, int64, int8, uint8, bfloat16.
* @li indices: An ND Tensor of type int32 or int64
* @li updates: An ND Tensor . \n
* Must be one of the following types: float32, float16, int16, int32, int64, int8, uint8, bfloat16.

* @par Attributes:
* @li axis: An optional int. Defaults to 0.
* @li reduction: An optional string. Defaults to string "none" and can be
* "add" or "mul". \n

* @attention Constraints:
* @li In non-last axis scenarios, you are advised to convert x, indices, and updates to the last axes,
* use ScatterElementsV2 for calculation, and then convert them to the original axes.
* @li Only Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component and
* Atlas A3 Training Series Product/Atlas A3 Inference Series Product support ScatterElementsV2. \n

* @par Outputs:
* var: A Tensor. Has the same type and format as input "var" . \n
*/
REG_OP(ScatterElementsV2)
    .INPUT(var, TensorType({DT_FLOAT,DT_FLOAT16,DT_INT16,DT_INT32,DT_INT64,DT_INT8,DT_UINT8,DT_BF16}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT,DT_FLOAT16,DT_INT16,DT_INT32,DT_INT64,DT_INT8,DT_UINT8,DT_BF16}))
    .OUTPUT(var, TensorType({DT_FLOAT,DT_FLOAT16,DT_INT16,DT_INT32,DT_INT64,DT_INT8,DT_UINT8,DT_BF16}))
    .ATTR(axis, Int, 0)
    .ATTR(reduction, String, "none")
    .OP_END_FACTORY_REG(ScatterElementsV2) // namespace ge

/*!
 * \file scatter_list_proto.cpp
 * \brief
 */

/**
* @brief Applies sparse updates into a variable reference.

* @par Inputs:
* @li var: The rewritten tensor list. Format is ND. Support 1D ~ 7D. Must be one of the following types:
* float16, bfloat16, float32, int8, int16, int32, int64, uint8, uint16, uint32, uint64.
* @li indice: The index tensor. Format is ND. Support 1D ~ 2D, the first dimension should be equal to "updates", the
* second dimension should be 2. Must be one of the following types: int32, int64. Index out of bounds is not supported.
* @li updates: The source tensor. Format is ND. Support 2D ~ 8D, the first dimension should be equal to the number of
* tensors of "var", and the dimension of "axis" should not be greather than "var", other dimensions should be equal to
* "var". Must have the same type of "var".
* @li mask: The mask tensor. Format is ND. Support 1D, the first dimension should be equal to "updates". Type should be
* uint8.

* @par Attributes:
* @li reduce: An optional string. Defaults to "update".
* @li axis: An optional int. Defaults to -2.

* @par Outputs:
* var: A tensor list. Must have the same type, shape and format as input "var".

* @par Third-party framework compatibility
* Compatible with the Mindspore operator ScatterList.
*/
REG_OP(ScatterList)
    .DYNAMIC_INPUT(var, "T")
    .INPUT(indice, TensorType::IndexNumberType())
    .INPUT(updates, "T")
    .OPTIONAL_INPUT(mask, TensorType({DT_UINT8}))
    .DYNAMIC_OUTPUT(var, "T")
    .ATTR(reduce, String, "update")
    .ATTR(axis, Int, -2)
    .DATATYPE(
        T, TensorType(
               {DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
                DT_UINT64}))
    .OP_END_FACTORY_REG(ScatterList)

/**
* @brief Computes the sum along segments of a tensor.

* @par Inputs:
* Two inputs, including:
* @li x: A Tensor of type NumberType. Support 1D ~ 8D.
* @li segment_ids: A 1D Tensor of type IndexNumberType, whose shape is same
* with "x.shape[0]".

* @par Outputs:
* y: A Tensor of type NumberType. \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator SegmentSum.

* @attention Constraints:
* @li segment_ids.dimNum = 1, and segment_ids.shape[0] = x.shape[0].
* @li segment_ids should sorted in ascending order, and segment_ids.value >= 0.
* @li y.type must be same with x.type.
* @li y.dimNum = x.dimNum, and y.shape = [max(segment_ids) + 1, x.shape[1:]]
*/

REG_OP(SegmentSum)
    .INPUT(x, TensorType::NumberType())
    .INPUT(segment_ids, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .OP_END_FACTORY_REG(SegmentSum)

/**
* @brief Converts a sparse representation into a dense tensor. \n

* @par Inputs:
* Four inputs, including:
* @li indices: An ND Tensor of type int32 or int64. Dimension must be 0D, 1D or 2D.
* Data between different rows(Corresponding index of output) cannot be duplicated, and should be sorted by ascending.
* Index cannot exceed the size of each dimension of output.
* @li output_shape: A 1D Tensor has the same dtype of indices.
* @li values: A 1D Tensor, Values corresponding to each row of indices, or a scalar value to be used for all sparse indices. \n
* Must be one of the following types: float32, float16, bfloat16, int16, uint16, int32, int64, int8, uint8, bool, double.
* @li default_value: An ND Tensor of the same dtype as values .
* Size must be 1.  \n

* @par Attributes:
* @li validate_indices: An optional bool. 
* If true, indices are checked to make sure they are sorted by ascending, no repeats, and cannot exceed the size of each dimension.
* This param is currently not effective in Ascend950PR/Ascend950DT. \n

* @par Outputs:
* y: A Tensor. Has the same type and format as input "values" . \n
*/
REG_OP(SparseToDense)
    .INPUT(indices, TensorType({DT_INT32,DT_INT64}))
    .INPUT(output_shape, TensorType({DT_INT32,DT_INT64}))
    .INPUT(values, TensorType({DT_FLOAT,DT_FLOAT16,DT_BF16,DT_INT16,DT_UINT16,DT_INT32,DT_INT64,DT_INT8,DT_UINT8,DT_BOOL,DT_DOUBLE}))
    .INPUT(default_value, TensorType({DT_FLOAT,DT_FLOAT16,DT_BF16,DT_INT16,DT_UINT16,DT_INT32,DT_INT64,DT_INT8,DT_UINT8,DT_BOOL,DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT,DT_FLOAT16,DT_BF16,DT_INT16,DT_UINT16,DT_INT32,DT_INT64,DT_INT8,DT_UINT8,DT_BOOL,DT_DOUBLE}))
    .ATTR(validate_indices, Bool, true)
    .OP_END_FACTORY_REG(SparseToDense) // namespace ge

/**
* @brief Perform topk-topp-sample sampling calculations based on input word frequency logits, top-k/top-p sampling parameters, and random sampling weight distribution q. Output the maximum word frequency logits_select_idx for each batch, along with the word frequency distribution logits_top_kp_select after topk-topp sampling.\n

* @par Inputs:
* Four inputs, including:
* @li logits: A Required tensor in [batch, voc_size] on default. Acts as the input word frequencies to be sampled. The frequency index is fixed as the last dimension. Currently supports 2 dimensions. Supports non-contiguous tensors, and must be a non-null pointer.
*   The supported types are:float16, bfloat16, format supports ND.
* @li top_k: A Required tensor in [batch, ] on default. Acts as the sampled k-value for each batch. The supported types are: int32, Supports non-contiguous tensors, and must be a non-null pointer.
* @li top_p: A Required tensor in [batch, ] on default. Acts as the p-value sampled for each batch. The supported types are: float16, bfloat16, Supports non-contiguous tensors, and must be a non-null pointer.
* @li q: An optional tensor in [batch, voc_size], Defaults to "None". A Required tensor in [batch, ] on default. Acts as the sampled k-value for each batch. The supported types are: float32, Supports non-contiguous tensors.\n

* @par Outputs:
* @li logits_select_idx: A 2D tensor which is the sequence of numbers, format supports ND, and must be a non-null pointer.
*    The supported types are: int64. \n
* @li logits_top_kp_select: An optional 2D tensor. Acts as the logits remaining in the logits file after undergoing the topK-topP calculation process. When `is_need_logits=False`, it is disabled and outputs a null pointer; when `is_need_logits=True`, it is enabled and outputs a dimension that must be [batch, voc_size]. The supported types are: float32.
*
* @par Attributes:
* @li eps: An optional attribute of type float. Prevent division by zero in softmax and weight sampling; default value 1e-8. 
* @li is_need_logits：An optional attribute of type bool. Control the output condition of logits_top_kp_select, defaulting to False for empty return. When set to True, a Tensor sized of [batch, voc_size] contains only remaining logits at their origin position in the input logits, will be returned as the logits_top_kp_select.
* @li top_k_guess: An optional attribute of type int32. Indicates the size of candidate logits for each batch when attempting top-P partial traversal sampling. Must be a positive integer, defaulting to 32.
*
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Range or PyTorch operator Range/Arange.
*/
 REG_OP(TopKTopPSample)
    .INPUT(logits, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(top_k, TensorType({DT_INT32}))
    .INPUT(top_p, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(q, TensorType({DT_FLOAT}))
    .OUTPUT(logits_select_idx, TensorType({DT_INT64}))
    .OUTPUT(logits_top_kp_select, TensorType({DT_FLOAT}))
    .ATTR(eps, Float, 1E-8)
    .ATTR(is_need_logits, Bool, false)
    .ATTR(top_k_guess, Int, 32)
    .OP_END_FACTORY_REG(TopKTopPSample)

/**
* @brief Perform topk-topp-sample sampling calculations based on input word frequency logits, top-k/top-p/min-p sampling parameters, and random sampling weight distribution q.
*   When is_need_sample_result is False, output the maximum word frequency logitsSelectIdx per batch and the word frequency distribution logitsTopKPSelect after topK-topP-minP sampling.
*   When is_need_sample_result is True, output the intermediate results logitsIdx and logitsSortMasked of topK-topP-minP sampling, where logitsSortMasked is the processed intermediate result of word frequency logits, and logitsIdx is its corresponding index in logits.\n
* @par Inputs:
* Five inputs, including:
* @li logits: A Required tensor in [batch, voc_size] on default. Acts as the input word frequencies to be sampled. The frequency index is fixed as the last dimension. Currently supports 2 dimensions. Supports non-contiguous tensors, and must be a non-null pointer.
*   The supported types are:float16, bfloat16, float32, format supports ND.
* @li top_k: A Required tensor in [batch, ] on default. Acts as the sampled k-value for each batch. The supported types are: int32, Supports non-contiguous tensors, and must be a non-null pointer.
* @li top_p: A Required tensor in [batch, ] on default. Acts as the p-value sampled for each batch. The supported types are: float16, bfloat16, float32, Supports non-contiguous tensors, and must be a non-null pointer.
* @li q: An optional tensor in [batch, voc_size], Defaults to "None". Acts as the sampled q-value for each batch. The supported types are: float32, Supports non-contiguous tensors.
* @li min_ps: An optional tensor in [batch, ], Defaults to "None". Acts as the minp-value sampled for each batch. The supported types are: float16, bfloat16, float32, Supports non-contiguous tensors.\n
* @par Outputs:
* @li logits_select_idx: A 2D tensor which is the sequence of numbers, format supports ND, and must be a non-null pointer.
*    The supported types are: int64. \n
* @li logits_top_kp_select: An optional 2D tensor. Acts as the logits remaining in the logits file after undergoing the topK-topP-minP calculation process. When `is_need_logits=False`, it is disabled and outputs a null pointer; when `is_need_logits=True`, it is enabled and outputs a dimension that must be [batch, voc_size]. The supported types are: float32.
* @li logits_idx: An optional 2D tensor, When `is_need_sample_result=False`, it is disabled and outputs a null pointer; when `is_need_sample_result=True`, it is enabled and outputs a dimension that must be [batch, voc_size]. The supported types are: int64. The format supports ND.
* @li logits_sort_masked: An optional 2D tensor. Acts as the intermediate result of the logits remaining after the topK-topP-minP calculation process. When `is_need_sample_result=False`, it is disabled and outputs a null pointer; when `is_need_sample_result=True`, it is enabled and outputs a dimension that must be [batch, voc_size]. The supported types are: float32.\n
*
* @par Attributes:
* @li eps: An optional attribute of type float. Prevent division by zero in softmax and weight sampling; default value 1e-8. 
* @li is_need_logits: An optional attribute of type bool. Control the output condition of logits_top_kp_select, defaulting to False for empty return. When set to True, a tensor sized of [batch, voc_size] contains only remaining logits at their origin position in the input logits, will be returned as the logits_top_kp_select.
* @li top_k_guess: An optional attribute of type int32. Indicates the size of candidate logits for each batch when attempting top-P partial traversal sampling. Defaulting to 32.
* @li ks_max: An optional attribute of type int32. Maximum value for topK enablement, defaulting to 1024.
* @li input_is_logits: An optional attribute of type bool. Indicates whether the input logits are not normalized, defaulting to True. When input_is_logits is True, logits will be normalized; when input_is_logits is False, logits will not be normalized.
* @li is_need_sample_result: An optional attribute of type bool. Indicates whether to output the intermediate results, defaulting to False. When is_need_sample_result is True, logits_idx and logits_sort_masked are valid output; when is_need_sample_result is False, logits_idx and logits_sort_masked are undefined initial data.
*
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Range or PyTorch operator Range/Arange.
*/
 REG_OP(TopKTopPSampleV2)
    .INPUT(logits, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(top_k, TensorType({DT_INT32}))
    .INPUT(top_p, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(q, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(min_ps, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(logits_select_idx, TensorType({DT_INT64}))
    .OUTPUT(logits_top_kp_select, TensorType({DT_FLOAT}))
    .OUTPUT(logits_idx, TensorType({DT_INT64}))
    .OUTPUT(logits_sort_masked, TensorType({DT_FLOAT}))
    .ATTR(eps, Float, 1E-8)
    .ATTR(is_need_logits, Bool, false)
    .ATTR(top_k_guess, Int, 32)
    .ATTR(ks_max, Int, 1024)
    .ATTR(input_is_logits, Bool, true)
    .ATTR(is_need_sample_result, Bool, false)
    .OP_END_FACTORY_REG(TopKTopPSampleV2)

/**
* @brief Computes the minimum along segments of a tensor.
    Computes a tensor such that (output[i] = min_{j...} x[j...].
    the value of segment_ids must be  in [0, num_segments - 1].
    for example:x = [[0,1,2],[3,4,5],[6,7,8]], segment_ids = [0,0,1] num_segments = 5
    output[0] = [0, 1, 2]
    output[1] = [6, 7, 8]
    output[2] = [2147483648, 2147483648, 2147483648]
    output[3] = [2147483648, 2147483648, 2147483648]
    output[4] = [2147483648, 2147483648, 2147483648]

* @par Inputs:
* Three inputs, including:
* @li x: A Tensor of type double, float32, float16, bfloat16, int8, int16, int32, int64, uint8, uint16, uint32, uint64, format is ND. bank of shape must greater zero.
* @li segment_ids: A Tensor of type int32, int64, whose shape is a prefix of "x", format is ND.
* @li num_segments: A 1D Tensor contains a single element of type int32, int64, format is ND.
* Indicates the output segment.

* @par Outputs:
* y: Have the same type and format of "x".

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator UnsortedSegmentMin.
*/
REG_OP(UnsortedSegmentMin)
    .INPUT(x, TensorType::RealNumberType())
    .INPUT(segment_ids, TensorType::IndexNumberType())
    .INPUT(num_segments, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::RealNumberType())
    .OP_END_FACTORY_REG(UnsortedSegmentMin)

/**
* @brief Returns locations of nonzero / true values in a tensor. \n

* @par Inputs:
* Including:
* x: A Tensor. Must be one of the following types:
  DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_QINT8,
  DT_QUINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_QINT32,
  DT_INT64, DT_UINT64, DT_BOOL, DT_COMPLEX64, DT_COMPLEX128 \n

* @par Outputs:
* y: A Tensor of type DT_INT64. \n

* @attention Constraints:
* Where runs on the Ascend AI CPU, which delivers poor performance.\n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Where.
*/

REG_OP(Where)
    .INPUT(x, TensorType({BasicType(), DT_BOOL}))
    .OUTPUT(y, TensorType({DT_INT64}))
    .OP_END_FACTORY_REG(Where)

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
*dx: A output ensor with the same type, shape, format as "x". Probability to logit conversion backpropagates the output
data. \n
*/

REG_OP(LogitGrad)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(dy, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(dx, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .ATTR(eps, Float, -1.0)
    .OP_END_FACTORY_REG(LogitGrad)

/**
* @brief The negative log likelihood loss .

* @par Inputs:
* The input x and weight must have the same type. Inputs include:
* @li x: A 2D or 4D tensor with shape (N, C) or (N, C, H, W). Support dtype: float32/bfloat16/float16.
* @li target: A 1D or 3D tensor with shape (N) or (N, H, W). When x is 2D(shape (N, C)),target should be 1D(shape(N)) or
scalar, when x is 4D(shape(N, C, H, W)), target should be 3d(shape(N, H, W)). Indicating the real label. Support dtype:
int32/int64/uint8.
* @li weight: A 1D tensor with shape (C) or none. Indicating the scale weight of each class. Support
dtype:float32/bfloat16/float16. \n

* @par Attributes:
* @li reduction: An optional attribute. Specifies the reduction to be applied to the output.
*                Type is string. Defaults to "mean" .
* @li ignore_index: An optional attribute. Specifying a target that is ignored and does not affect the input gradient.
*                   Type is int. Defaults to -100 . \n

* @par Outputs:
* @li y: if reduction is "none", a 1D or 3D tensor with shape (N) or (N, H, W). When x is 2D(shape (N, C)), y should be
1D(shape(N)) or scalar, when x is 4D(shape(N, C, H, W)), y should be 3d(shape(N, H, W)). Otherwise, shape is (1).
Support dtype: float32/bfloat16/float16.
* @li total_weight: A 1D tensor with shape (1). Support dtype: float32/bfloat16/float16. \n

* @par Third-party framework compatibility
* Compatible with pytorch NLLLoss operator
*/
REG_OP(NLLLoss)
    .INPUT(x, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .INPUT(target, TensorType({DT_INT32, DT_INT64, DT_UINT8}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .OUTPUT(total_weight, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .ATTR(reduction, String, "mean")
    .ATTR(ignore_index, Int, -100)
    .OP_END_FACTORY_REG(NLLLoss)

/**
* @brief The negative log likelihood loss grad.

* @par Inputs:
* @li x: A 2D or 4D tensor dtype of float32 or bfloat16 or float16 with shape (N, C) or (N, C, H, W).
* @li y_grad: A 1D or 3D tensor dtype of float32 or bfloat16 or float16 with shape (N) or (N, H, W) if reduction is "none". When x is 2D(shape (N, C)), y_grad should be 1D(shape(N)) or scalar, when x is 4D(shape(N, C, H, W)), y_grad should be 3d(shape(N, H, W)). Otherwise, shape is (1).
* @li target: Indicates the real label. A 1D or 3D tensor dtype of int32, int64 or uint8 with shape (N) or (N, H, W). When x is 2D(shape (N, C)), target should be 1D(shape(N)) or scalar, when x is 4D(shape(N, C, H, W)), target should be 3d(shape(N, H, W)).
* @li weight: Indicates the weight of each class. A 1D tensor dtype of float32 or bfloat16 or float16 with shape (C).
* @li total_weight: A 1D tensor dtype of float32 or bfloat16 or float16 with shape (1).

* @par Attributes:
* @li reduction: Computation method of the loss function. An optional string. Defaults to "mean" .
* @li ignore_index: Specifies a target value that is ignored and does not affect the input gradient. An optional int. Defaults to -100.

* @par Outputs:
* x_grad: A tensor has the same shape as "x". Must be the following type: float32, bfloat16, float16.

* @par Third-party framework compatibility
* Compatible with pytorch NLLLossGrad operator
*/
REG_OP(NLLLossGrad)
    .INPUT(x, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .INPUT(y_grad, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .INPUT(target, TensorType({DT_INT32, DT_INT64, DT_UINT8}))
    .INPUT(weight, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .INPUT(total_weight, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .OUTPUT(x_grad, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .ATTR(reduction, String, "mean")
    .ATTR(ignore_index, Int, -100)
    .OP_END_FACTORY_REG(NLLLossGrad)

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
* @brief Fused Operator of AddLayerNorm and Quantize/DynamicQuant/AscendQuantV2. \n
*  calculating: x1, x2, gamma, beta, bias, scales1, scales2, zero_points1, zero_points2 \n
*  x = x1 + x2 + bias \n
*  rstd = 1 / (sqrt(Var(x) + eps)) \n
*  y = (x - E(x)) * rstd * gamma + beta \n
*  when quant_mode = "static" and div_mode=true, output out_scales1 and out_scales2 are invalid: \n
*  y1 = round(y / scales1) + zero_points1 \n
*  y2 = round(y / scales2) + zero_points2 \n
*  when quant_mode = "static" and div_mode=false, output out_scales1 and out_scales2 are invalid: \n
*  y1 = round(y * scales1) + zero_points1 \n
*  y2 = round(y * scales2) + zero_points2 \n
*  when quant_mode = "dynamic": \n
*  tmp1 = y * scales1 \n
*  tmp2 = y * scales2 \n
*  out_scales1 = reduce_max(abs(tmp1))/127 \n
*  out_scales2 = reduce_max(abs(tmp2))/127 \n
*  y1 = round(tmp1 / out_scales1) \n
*  y2 = round(tmp2 / out_scales2) \n

* @par Inputs
* @li x1: A tensor for add compute. Support dtype: float32, float16, bfloat16, support format: ND.
* @li x2: A tensor for add compute. Support dtype: float32, float16, bfloat16, support format: ND.
* @li gamma: A tensor for layer norm weight params. Support dtype: float32, float16, bfloat16, support format: ND.
* @li beta: A tensor for layer norm weight params. Support dtype: float32, float16, bfloat16, support format: ND.
* @li bias: An optional input tensor for add compute. Support dtype: float32, float16, bfloat16, support format: ND.
* @li scales1: An optional input tensor for one of quant scale. Support dtype: float32, float16, bfloat16, support
format: ND.
* @li scales2: An optional input tensor for another quant scale. Support dtype: float32, float16, bfloat16, support
format: ND.
* @li zero_points1: An optional input tensor for one of quant offset. Support dtype: float32, float16, bfloat16,
support format: ND.
* @li zero_points2: An optional input tensor for another quant offset. Support dtype: float32, float16, bfloat16,
support format: ND.

* @par Attributes
* @li quant_mode: An optional attribute utilized to select quant mode, can be "dynamic" or "static", the type is
string. Defaults to "dynamic".
* @li epsilon: An optional attribute for layer norm compute, the type is float. Defaults to 1e-5.
* @li additional_output: An optional attribute control whether output x valid or invalid, the type is bool. Defaults
to false, which means x output is invalid.
* @li div_mode: An optional attribute control static quant algorithm, the type is bool. Defaults
to true, which means scales while be divided by normlization output.

* @par Outputs
* @li y1: Quantize result 1.
*     A tensor. Support dtype: int8, support format: ND.
* @li y2: Quantize result 2.
*     A tensor. Support dtype: int8, support format: ND.
* @li x: Describing the result of x1 + x2 + bias.
*     A tensor. Support dtype: float32, float16, bfloat16, support format: ND.
* @li out_scales1: Describing the result of dynamic quantize scales computed via scales1.
*     A tensor. Support dtype: float32, support format: ND.
* @li out_scales2: Describing the result of dynamic quantize scales computed via scales2.
*     A tensor. Support dtype: float32, support format: ND.
*/
REG_OP(AddLayerNormQuant)
    .INPUT(x1, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(x2, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(gamma, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(beta, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(bias, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(scales1, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(scales2, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(zero_points1, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(zero_points2, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(y1, ge::TensorType({DT_INT8, DT_INT8, DT_INT8}))
    .OUTPUT(y2, ge::TensorType({DT_INT8, DT_INT8, DT_INT8}))
    .OUTPUT(x, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(out_scales1, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(out_scales2, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .ATTR(quant_mode, String, "dynamic")
    .ATTR(epsilon, Float, 1e-5f)
    .ATTR(additional_output, Bool, false)
    .ATTR(div_mode, Bool, true)
    .OP_END_FACTORY_REG(AddLayerNormQuant)

/**
* @brief AddRmsNorm operator interface implementation. \n
*  calculating: x1, x2, gamma \n
*  x = x1 + x2 \n
*  rstd = np.rsqrt(np.mean(np.power(x,2), reduce_axis, keepdims=True) + epsilon)) \n
*  y = gamma * (x * rstd)

* @par Inputs
* Three inputs, including:
* @li x1: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li x2: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li gamma: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].

* @par Attributes
* epsilon: Input eps in the formula, which is used to prevent division-by-zero errors.
* An optional attribute, the type is float. Must be greater than or equal to zero. Defaults to 1e-6.

* @par Outputs
* Three outputs, including:
* @li y: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li rstd: A Tensor. Support dtype: [float32], support format: [ND].
* @li x: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
*/
REG_OP(AddRmsNorm)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(rstd, TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(epsilon, Float, 1e-6f)
    .OP_END_FACTORY_REG(AddRmsNorm)

/**
* @brief AddRmsNormCast operator interface implementation. \n
*  calculating: x1, x2, gamma \n
*  x = x1 + x2 \n
*  rstd = np.rsqrt(np.mean(np.power(x,2), reduce_axis, keepdims=True) + epsilon)) \n
*  y2 = gamma * (x * rstd) \n
*  y1 = cast16232(y2) \n

* @par Inputs
* Three inputs, including:
* @li x1: A tensor. Support dtype: float16/bfloat16, support format: ND.
* @li x2: A tensor. Support dtype: float16/bfloat16, support format: ND.
* @li gamma: A tensor. Support dtype: float16/bfloat16, support format: ND.

* @par Attributes
* epsilon: Input eps in the formula, which is used to prevent division-by-zero errors.
* An optional attribute, the type is float. Defaults to 1e-6.

* @par Outputs
* Four outputs, including:
* @li y1: A tensor. Support dtype: float32, support format: ND.
* @li y2: A tensor. Support dtype: float16/bfloat16, support format: ND.
* @li rstd: A tensor. Describing the reciprocal of (x1 + x2)'s standard deviation.
*           Support dtype: float32, support format: ND.
* @li x: A tensor. Support dtype: float16/bfloat16, support format: ND.
*/
REG_OP(AddRmsNormCast)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y1, TensorType({DT_FLOAT}))
    .OUTPUT(y2, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(rstd, TensorType({DT_FLOAT}))
    .OUTPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .ATTR(epsilon, Float, 1e-6f)
    .OP_END_FACTORY_REG(AddRmsNormCast)

/**
* @brief Fused Operator of Add, RmsNorm and DynamicQuant.
* Calculating input: x1, x2, gamma, smooth_scale1, smooth_scale2 \n
* Calculating process: \n
*  x = x1 + x2 \n
*  rstd = np.rsqrt(np.mean(np.power(x, 2), reduce_axis, keepdims=True) + epsilon)) \n
*  rmsnorm_out = x * rstd * gamma \n
*  if smooth_scales1 exist: \n
*    scale1 = row_max(abs(rmsnorm_out * smooth_scale1)) / 127 \n
*  if smooth_scales1 not exist: \n
*    scale1 = row_max(abs(rmsnorm_out)) / 127 \n
*  y1 = round(rmsnorm_out / scale1) \n
*  if smooth_scales2 exist: \n
*    scale2 = row_max(abs(rmsnorm_out * smooth_scale2)) / 127 \n
*    y2 = round(rmsnorm_out / scale2) \n
*  if smooth_scales2 not exist:  \n
*    not calculate scale2 and y2. \n

* @par Inputs
* @li x1: A tensor. Input x1 for the add operation.
*         Support dtype: float16/bfloat16, support format: ND.
* @li x2: A tensor. Input x2 for the add operation.
*         Support dtype: float16/bfloat16, support format: ND.
* @li gamma: A tensor. Describing the weight of the rmsnorm operation.
*            Support dtype: float16/bfloat16, support format: ND.
* @li smooth_scale1: A tensor. Describing the weight of the first dynamic quantization.
*              Support dtype: float16/bfloat16, support format: ND.
* @li smooth_scale2: An optional input tensor. Describing the weight of the secend dynamic quantization.
*              Support dtype: float16/bfloat16, support format: ND.
* @li beta: An optional input tensor. Describing the offset value of dynamic quantization.
*              Support dtype: float16/bfloat16, support format: ND. Has the same dtype and shape as "gamma".
* @par Attributes
* @li epsilon: An optional attribute. Describing the epsilon of the rmsnorm operation.
*          The type is float. Defaults to 1e-6.
* @li dst_type: An optional int32. Output y data type enum value. Support DT_INT8, DT_INT4, DT_HIFLOAT8, DT_FLOAT8_E5M2,
*               DT_FLOAT8_E4M3FN. Defaults to DT_INT8.

* @par Outputs
* @li y1: A tensor. Describing the output of the first dynamic quantization.
*                   Support dtype: int8/hifloat8/float8e5m2/float8e4m3fn, support format: ND.
* @li y2: A tensor. Describing the output of the second dynamic quantization.
*                   Support dtype: int8/hifloat8/float8e5m2/float8e4m3fn, support format: ND.
* @li x: A tensor. Describing the output of the x1+x2 add operation.
*                  Support dtype: float16/bfloat16, support format: ND.
* @li scale1: A tensor. Describing of the factor for the first dynamic quantization.
*                  Support dtype: float32, support format: ND.
* @li scale2: A tensor. Describing of the factor for the second dynamic quantization.
*                  Support dtype: float32, support format: ND.
*/

REG_OP(AddRmsNormDynamicQuant)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(smooth_scale1, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(smooth_scale2, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(beta, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y1, TensorType({DT_INT8, DT_HIFLOAT8, DT_FP8_E5M2, DT_FP8_E4M3FN, DT_INT4}))
    .OUTPUT(y2, TensorType({DT_INT8, DT_HIFLOAT8, DT_FP8_E5M2, DT_FP8_E4M3FN, DT_INT4}))
    .OUTPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(scale1, TensorType({DT_FLOAT, DT_FLOAT}))
    .OUTPUT(scale2, TensorType({DT_FLOAT, DT_FLOAT}))
    .ATTR(epsilon, Float, 1e-6)
    .ATTR(output_mask, ListBool, {})
    .ATTR(dst_type, Int, DT_INT8)
    .OP_END_FACTORY_REG(AddRmsNormDynamicQuant)

/**
* @brief Fused Operator of AddRmsNorm and DynamicQuant . \n

* @par Inputs:
* @li x1: A tensor of type float16/bfloat16. Supported format "ND". \n
* @li x2: A tensor of type float16/bfloat16. Supported format "ND". \n
* @li gamma: A tensor of type float16/bfloat16. Supported format "ND". \n
* @li smooth_scale1: Optional Input. A tensor of type float16/bfloat16. Supported format "ND". \n
* @li smooth_scale2: Optional Input. A tensor of type float16/bfloat16. Supported format "ND". \n

* @par Attributes:
* epsilon: An optional float, default value is 1e-6.

* @par Outputs:
* @li y1: A tensor of type int8, quantize result for rmsnorm(x1+x2)*smooth1. Supported format "ND". \n
* @li y2: A tensor of type int8, quantize result for rmsnorm(x1+x2)*smooth2. Supported format "ND". \n
* @li y3: A tensor of type float32, cast result for rmsnorm(x1+x2). Supported format "ND". \n
* @li y4: A tensor of type float16/bfloat16, describe the result for rmsnorm(x1+x2). Supported format "ND". \n
* @li x: A tensor of type float16/bfloat16, describing the result of x1 + x2. Supported format "ND". \n
* @li scale1: A tensor of type float32, describing the result of dynamic quantize scales. Supported format "ND". \n
* @li scale2: A tensor of type float32, describing the result of dynamic quantize scales. Supported format "ND". \n
*/
REG_OP(AddRmsNormDynamicQuantV2)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(smooth_scale1, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(smooth_scale2, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y1, TensorType({DT_INT8}))
    .OUTPUT(y2, TensorType({DT_INT8}))
    .OUTPUT(y3, TensorType({DT_FP32}))
    .OUTPUT(y4, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(scale1, TensorType({DT_FLOAT}))
    .OUTPUT(scale2, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 1e-6)
    .OP_END_FACTORY_REG(AddRmsNormDynamicQuantV2)

/**
* @brief AddRmsNormQuant operator interface implementation.
* Calculating input: x1, x2, gamma, scales1, scales2, zero_points1, zero_points2 \n
* Calculating process: \n
*  x = x1 + x2 \n
*  rstd = np.rsqrt(np.mean(np.power(x, 2), reduce_axis, keepdims=True) + epsilon)) \n
*  rmsnorm_out = x * rstd * gamma \n
*  if div_mode is true: \n
*    y1 = round(rmsnorm_out / scales1 + zero_points1) \n
*    y2 = round(rmsnorm_out / scales2 + zero_points2) \n
*  if div_mode is false: \n
*    y1 = round(rmsnorm_out * scales1 + zero_points1) \n
*    y2 = round(rmsnorm_out * scales2 + zero_points2) \n

* @par Inputs
* @li x1: A tensor. Input x1 for the add operation.
*         Support dtype: float32/float16/bfloat16, support format: ND.
* @li x2: A tensor. Input x2 for the add operation.
*         Support dtype: float32/float16/bfloat16, support format: ND.
* @li gamma: A tensor. Describing the weight of the rmsnorm operation.
*            Support dtype: float32/float16/bfloat16, support format: ND.
* @li scales1: A tensor. Describing the weight of the first quant operation.
*              Support dtype: float32/float16/bfloat16, support format: ND.
* @li scales2: An optional input tensor. Describing the weight of the secend quant operation.
*              Support dtype: float32/float16/bfloat16, support format: ND.
* @li zero_points1: An optional input tensor. Describing the bias of the first quant operation.
*                   Support dtype: int32/float32/float16/bfloat16, support format: ND.
* @li zero_points2: An optional input tensor. Describing the bias of the secend quant operation.
*                   Support dtype: int32/float32/float16/bfloat16, support format: ND.

* @par Attributes
* @li axis: An optional attribute. Describing the axis of the quant operation, does not take effect now.
*           The type is int. Defaults to -1.
* @li epsilon: An optional attribute. Describing the epsilon of the rmsnorm operation.
*              The type is float. Defaults to 1e-6.
* @li div_mode: An optional attribute. When div_mode is true, the quant opertaion uses division, otherwise, uses
multiplication.
*               The type is bool. Defaults to true.
* @li dst_type: An optional int32. Output y data type enum value. Support DT_INT8, DT_HIFLOAT8, DT_FLOAT8_E5M2, 
*               DT_FLOAT8_E4M3FN. Defaults to DT_INT8.
* @par Outputs
* @li y1: A tensor. Describing the output of the first quant operation.
*                   Support dtype: int8/hifloat8/float8e5m2/float8e4m3fn, support format: ND.
* @li y2: A tensor. Describing the output of the second quant operation.
*                   Support dtype: int8/hifloat8/float8e5m2/float8e4m3fn, support format: ND.
* @li x: A tensor. Describing the output of the x1+x2 add operation.
*                  Support dtype: float32/float16/bfloat16, support format: ND.
*/

REG_OP(AddRmsNormQuant)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(scales1, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(scales2, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(zero_points1, TensorType({DT_INT32, DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(zero_points2, TensorType({DT_INT32, DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y1, TensorType({DT_INT8, DT_HIFLOAT8, DT_FP8_E5M2, DT_FP8_E4M3FN}))
    .OUTPUT(y2, TensorType({DT_INT8, DT_HIFLOAT8, DT_FP8_E5M2, DT_FP8_E4M3FN}))
    .OUTPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(axis, Int, -1)
    .ATTR(epsilon, Float, 1e-6f)
    .ATTR(div_mode, Bool, true)
    .ATTR(dst_type, Int, DT_INT8)
    .OP_END_FACTORY_REG(AddRmsNormQuant)

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
* @brief Batch normalization (also known as batch norm) is a method used to
* make training of artificial neural networks faster and more stable
* through normalization of the layers' inputs by re-centering and re-scaling.

* @par Inputs:
* @li x: A 4D/5D tensor of type float16/bfloat16/float32, with format NCHW/NHWC/NCDHW/NDHWC. describing the feature_map.
* @li weight: A 1D tensor with the shape is same as dim C of input x, support dtypes are related to input x dtype,
* the following combinations are supported: [x: float16, weight: float16/float32], [x: bfloat16, weight:
bfloat16/float32], [x: float32, weight: float32].
* describing the weight.
* @li bias: A 1D tensor of the same dtype and shape as input weight, describing the bias.
* @li running_mean: A 1D tensor of type float16/bfloat16/float32, the shape is same as dim C of input x,
* the following combinations are supported: [x: float16, running_mean: float16/float32], [x: bfloat16, running_mean:
bfloat16/float32], [x: float32, running_mean: float32].
* describing the running mean.
* @li running_var: A 1D tensor of the same dtype and shape as input running_mean, describing the running var. \n

* @par Attributes:
* @li epsilon: An optional float32, small value added to variance to avoid dividing by zero. Defaults to "1e-5".
* @li momentum: An optional float32, the value used for the running_mean and running_var computation. Defaults to "0.1".
* @li is_training: An optional bool, specifying if the operation is used for training or inference. Defaults to "True",
now only support "True". \n

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
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gx, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(beta, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(mean, TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(rstd, TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
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
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gx, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(mean, TensorType::({DT_FLOAT}))
    .INPUT(rstd, TensorType::({DT_FLOAT}))
    .OUTPUT(dx, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(dgx, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(dbeta, TensorType({DT_FLOAT}))
    .OUTPUT(dgamma, TensorType({DT_FLOAT}))
    .ATTR(alpha, Float, 0.3f)
    .OP_END_FACTORY_REG(DeepNormGrad);

/**
* @brief DuaQuantizeAddLayerNorm operator interface implementation.

* @par Inputs
* @li x1: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li x2: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li gamma: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li beta: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li bias: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li scales1: A Tensor. Support dtype: [float32, bfloat16], support format: [ND].
* @li scales2: A Tensor. Support dtype: [float32, bfloat16], support format: [ND].
* @li zero_points1: A optional Tensor. Support dtype: [int8, uint8, bfloat16, int32], support format: [ND].
* @li zero_points2: A optional Tensor. Support dtype: [int8, uint8, bfloat16, int32], support format: [ND].

* @par Attributes
* @li dtype: A required attribute, the type is int. No defaults value.
* @li axis: A optional attribute, the type is float. Defaults to -1.
* @li epsilon: A optional attribute, the type is float. Defaults to 1e-5.
* @li additional_output: A optional attribute, the type is bool. Defaults to false.

* @par Outputs
* @li y1: A Tensor. Support dtype: [int8, uint8, int32], support format: [ND].
* @li y2: A Tensor. Support dtype: [int8, uint8, int32], support format: [ND].
* @li x: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
*/
REG_OP(DuaQuantizeAddLayerNorm)
    .INPUT(x1, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(x2, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(gamma, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(beta, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(bias, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(scales1, ge::TensorType({DT_BF16, DT_FLOAT}))
    .INPUT(scales2, ge::TensorType({DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(zero_points1, ge::TensorType({DT_INT8, DT_UINT8, DT_BF16, DT_INT32}))
    .OPTIONAL_INPUT(zero_points2, ge::TensorType({DT_INT8, DT_UINT8, DT_BF16, DT_INT32}))
    .OUTPUT(y1, ge::TensorType({DT_INT8, DT_UINT8, DT_INT32}))
    .OUTPUT(y2, ge::TensorType({DT_INT8, DT_UINT8, DT_INT32}))
    .OUTPUT(x, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .REQUIRED_ATTR(dtype, Int)
    .ATTR(axis, Int, -1)
    .ATTR(epsilon, Float, 1e-5f)
    .ATTR(additional_output, Bool, false)
    .OP_END_FACTORY_REG(DuaQuantizeAddLayerNorm)

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
 * @brief backward operator for group normalization. \n
 * @par Inputs:
 * Five input, including:
 * @li dy: A tensor. Group grad. Datatype support float32, float16, bfloat16. Format support ND.
 * "dy" supports 2-8 dimensions (N, C, *), the calculation logic only cares about the first two dimensions (N and C),
 * and the rest can all be combined into one dimension.
 * @li mean: A tensor. Mean of each group. Datatype support float32, float16, bfloat16. Format support ND.
 * Must be 2D (N, num_groups).
 * @li rstd: A tensor. Reciprocal standard deviation of each group. Datatype support float32, float16, bfloat16. Format
 support ND.
 * Must be 2D (N, num_groups).
 * @li x: A Tensor. Specifies the offset. Datatype support float32, float16, bfloat16. Format support ND.
 * "x" supports 2-8 dimensions (N, C, *), the calculation logic only cares about the first two dimensions (N and C),
 * and the rest can all be combined into one dimension.
 * @li gamma: A tensor. Specifies the scaling factor. Datatype support float32, float16, bfloat16. Format support ND.
 * Must be 1D. The value of "gamma" needs to be consistent with the C-axis value of "x".

 * @par Attributes:
 * @li num_groups: Int. Number specifying the number of group.
 * @li data_format: An optional string. Defaults to NCHW.
 * @li dx_is_require: An optional bool, controls whether to return dx. Defaults to true.
 * @li dgamma_is_require: An optional bool, controls whether to return dgamma. Defaults to true.
 * @li dbeta_is_require: An optional bool, controls whether to return dbeta. Defaults to true.

 * @par Outputs:
 * Three output, including:
 * @li dx: A tensor. x factor grad. Datatype is the same as the input datatype. Has the same format and shape as "x".
 * @li dgamma: A tensor. Scale factor grad. Has the same datatype, format and shape as "gamma".
 * @li dbeta: A tensor. Offset factor grad. Has the same datatype, format and shape as "gamma".
 * @par Third-party framework compatibility
 * @li Compatible with the backward of PyTorch operator GroupNorm.
 */

REG_OP(GroupNormGrad)
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mean, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(rstd, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(dx, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(dgamma, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(dbeta, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(num_groups, Int)
    .ATTR(data_format, String, "NCHW")
    .ATTR(dx_is_require, Bool, true)
    .ATTR(dgamma_is_require, Bool, true)
    .ATTR(dbeta_is_require, Bool, true)
    .OP_END_FACTORY_REG(GroupNormGrad)

/**
* @brief Performs group normalization and silu.

* @par Inputs:
* Three inputs
* @li x: A ND Tensor of type bfloat16/float16/float32.
* @li gamma: A Tensor of type bfloat16/float16/float32.
* Must be 1D. Specifies the scaling factor.
* @li beta: A Tensor of type bfloat16/float16/float32.
* Must be 1D. Specifies the offset.

* @par Attributes:
* @li num_groups: An required int32/int64, specifying the number of group.
* @li eps: An optional float32, specifying the small value added to the
* denominator for numerical stability. Defaults to "0.00001".
* @li activate_silu: An optional bool.  Defaults to "true".

* @par Outputs:
* Three outputs
* @li y: A ND Tensor of type bfloat16/float16/float32 for the normalized "x".
* @li mean: A Tensor of type bfloat16/float16/float32.
* Must be 1D. Specifies the mean of "x".
* Ascend 950 AI Processor: The data type must be consistent with that of gamma and beta.
* @li rstd: A Tensor of type bfloat16/float16/float32.
* Must be 1D. Specifies the rstd of "x".
* Ascend 950 AI Processor: The data type must be consistent with that of gamma and beta.

* @par Third-party framework compatibility
* @li Compatible with the PyTorch operator GroupNorm and Silu.

*/
REG_OP(GroupNormSilu)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(gamma, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(beta, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(mean, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(rstd, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(num_groups, Int)
    .ATTR(eps, Float, 0.00001f)
    .ATTR(activate_silu, Bool, true)
    .OP_END_FACTORY_REG(GroupNormSilu)

/**
* @brief Performs group normalization and swish. \n

* @par Inputs:
* Three inputs
* @li x: A ND Tensor of type bfloat16/float16/float32.
* @li gamma: A Tensor of type bfloat16/float16/float32.
* Must be 1D. Specifies the scaling factor.
* @li beta: A Tensor of type bfloat16/float16/float32.
* Must be 1D. Specifies the offset. \n

* @par Attributes:
* @li num_groups: An required int32/int64, specifying the number of group.
* @li eps: An optional float32, specifying the small value added to the
* denominator for numerical stability. Defaults to "0.00001".
* @li data_format: An optional String, Defaults to NCHW.
* @li activate_swish: An optional bool.  Defaults to "true".
* @li swish_scale: An optional float.  Defaults to "1". \n

* @par Outputs:
* Three outputs
* @li y: A ND Tensor of type bfloat16/float16/float32 for the normalized "x".
* @li mean: A Tensor of type bfloat16/float16/float32.
* Must be 1D. Specifies the mean of "x".
* @li rstd: A Tensor of type bfloat16/float16/float32.
* Must be 1D. Specifies the rstd of "x". \n

* @par Third-party framework compatibility
* @li Compatible with the PyTorch operator GroupNorm and Swish.

*/
REG_OP(GroupNormSwish)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(gamma, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(beta, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(mean, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(rstd, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(num_groups, Int)
    .ATTR(data_format, String, "NCHW")
    .ATTR(eps, Float, 0.00001f)
    .ATTR(activate_swish, Bool, true)
    .ATTR(swish_scale, Float, 1.0)
    .OP_END_FACTORY_REG(GroupNormSwish)

/**
* @brief Performs the backward operation of group normalization and swish.

 * @par Inputs:
 * Six input, including:
 * @li dy: A Tensor. Group grad. Datatype support float32, float16, bfloat16. Format support ND.
 * @li mean: A Tensor. Mean of each group. Datatype support float32, float16, bfloat16. Format support ND.
 * @li rstd: A Tensor. Reciprocal standard deviation of each group. Datatype support float32, float16, bfloat16. Format
support ND.
 * @li x: A Tensor. Specifies the offset. Datatype support float32, float16, bfloat16. Format support ND. Same shape as
mean.
 * @li gamma: A Tensor. Specifies the scaling factor. Datatype support float32, float16, bfloat16. Format support ND.
Same shape as dy.
 * @li beta: A Tensor. Specifies the intercept. Datatype support float32, float16, bfloat16. Format support ND. Same
shape as gamma.

* @par Attributes:
* @li num_groups: Int. Number specifying the number of group.
* @li data_format: An optional String, Defaults to NCHW.
* @li swish_scale: An optional float. Defaults to "1.0".
* @li dgamma_is_require: An optional bool, controls whether to return weight.grad. Defaults to true.
* @li dbeta_is_require: An optional bool, controls whether to return beta.grad. Defaults to true.

 * @par Outputs:
 * Three output, including:
 * @li dx: A Tensor. x factor grad. Datatype is the same as the input Datatype. Format support ND.
 * @li dgamma: A Tensor. scale factor grad. Datatype is the same as the input Datatype. Format support ND.
 * @li dbeta: A Tensor. offset factor grad. Datatype is the same as the input Datatype. Format support ND.

* @par Third-party framework compatibility
* @li Compatible with the backward of PyTorch operator GroupNorm and Swish.

*/
REG_OP(GroupNormSwishGrad)
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mean, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(rstd, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(beta, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(dx, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(dgamma, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(dbeta, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(num_groups, Int)
    .ATTR(data_format, String, "NCHW")
    .ATTR(swish_scale, Float, 1.0)
    .ATTR(dgamma_is_require, Bool, true)
    .ATTR(dbeta_is_require, Bool, true)
    .OP_END_FACTORY_REG(GroupNormSwishGrad)

/**
* @brief Performs group normalization . \n

* @par Inputs:
* Three inputs
* @li x: A ND tensor of type bfloat16, float16, float32. The input feature map will be processed by group normalization.
* "x" supports 2-8 dimensions (N, C, *), the calculation logic only cares about the first two dimensions (N and C),
* and the rest can all be combined into one dimension. 
* The data type and shape of x must meet the following conditions:  
* - When the data type of x is float32, C/num_groups must be a multiple of 8.  
* - When the data type of x is float16 or bfloat16, C/num_groups must be a multiple of 16.
* @li gamma: A ND tensor of type bfloat16, float16, float32. Must be 1D. Specifies the scaling factor.
* The value of "gamma" needs to be consistent with the C-axis value of "x". Has the same dype as "x".
* @li beta: A ND tensor of type bfloat16, float16, float32. Must be 1D. Specifies the offset.
* The value of "beta" needs to be consistent with the C-axis value of "x". Has the same dype as "x". \n

* @par Attributes:
* @li num_groups: An required int32, specifying the number of group.
* @li eps: An optional float32, specifying the small value added to variance to avoid dividing by zero. Defaults to "0.00001".
* @li data_format: An optional string, specifying the format of "x". Defaults to "NHWC". This parameter is reserved and does not take effect.
* @li is_training: An optional bool, specifying if the operation is used for training or inference. Defaults to "True".

* - When set to true, it indicates training mode and uses the mean and variance of the current batch.
* - When set to false, it indicates inference mode and uses the mean and variance saved during training. \n

* @par Outputs:
* Three outputs
* @li y: A ND tensor of type bfloat16, float16, float32 for the normalized "x". Has the same type, format and shape as "x".  
* @li mean: A ND tensor of type bfloat16, float16, float32. Must be 2D (N, num_groups). Specifies the mean of "x".
* Has the same dype as "x".
* @li rstd: A ND tensor of type bfloat16, float16, float32. Must be 2D (N, num_groups). Specifies the rstd of "x".
* Has the same dype as "x". \n

* @par Third-party framework compatibility
* @li Compatible with the PyTorch operator GroupNorm.

*/
REG_OP(GroupNormV2)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(gamma, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(beta, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(mean, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(rstd, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(num_groups, Int)
    .ATTR(data_format, String, "NHWC")
    .ATTR(eps, Float, 0.00001f)
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(GroupNormV2)

/**
* @brief Fused Operator of Add and LayerNorm.

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
* @li x1: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li mean: A Tensor. Support dtype: [float32], support format: [ND].
* @li rstd: A Tensor. Support dtype: [float32], support format: [ND].
* @li x2: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
*/
REG_OP(InplaceAddLayerNorm)
    .INPUT(x1, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(x2, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(gamma, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(beta, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(bias, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(x1, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(mean, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(rstd, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(x2, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .ATTR(epsilon, Float, 1e-5f)
    .ATTR(additional_output, Bool, false)
    .OP_END_FACTORY_REG(InplaceAddLayerNorm)

/**
* @brief InplaceAddRmsNorm operator interface implementation. \n
*  calculating: x1, x2, gamma \n
*  x2 = x1 + x2 \n
*  rstd = np.rsqrt(np.mean(np.power(x,2), reduce_axis, keepdims=True) + epsilon)) \n
*  x1 = gamma * (x2 * rstd)

* @par Inputs
* Three inputs, including:
* @li x1: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li x2: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li gamma: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].

* @par Attributes:
* epsilon: A optional attribute, the type is float. Defaults to 1e-6.

* @par Outputs
* Three outputs, including:
* @li x1: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li rstd: A Tensor. Support dtype: [float32], support format: [ND].
* @li x2: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
*/
REG_OP(InplaceAddRmsNorm)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(rstd, TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(epsilon, Float, 1e-6f)
    .OP_END_FACTORY_REG(InplaceAddRmsNorm)

/**
* @brief Instance normalization (also known as instance norm) is a method used to
* make training of artificial neural networks faster and more stable
* through normalization of the layers' inputs by re-centering and re-scaling.

* @par Inputs:
* Three inputs, including:
* @li x: Empty tensors are supported, but only allows the reduction axis to have shape 0, the dim N and dim C must not be empty. 
         Must be one of the following types: bfloat16, float16, float32.
         4D/5D with format NCHW/NHWC/NCDHW/NDHWC.
         2D~8D with format ND, the second dim is fixed as the dim C.
* @li gamma: Empty tensors are not supported. An ND Tensor. 
             The data type is same as input x, if not, explicitly set to float32.
             The shape is same as dim C of input x.
* @li beta: Empty tensors are not supported. An ND tensor of the same dtype and shape as input gamma.

* @par Attributes:
* Two attributes, including:
* @li data_format: A optional attribute, the type is string. Defaults to "NDHWC".
* @li epsilon: A optional attribute, the type is float. Defaults to "1e-6".

* @par Outputs:
* Three outputs, including:
* @li y: Empty tensors are supported. The shape, data type and format are the same as the input x.
* @li mean: Empty tensors are not supported. An ND tensor of the same dtype as input gamma, 
            the number of dim is same as input x, 
            the shape size of the non-reduction axis is same as input x, 
            the reduction axis is 1.
* @li variance: Empty tensors are not supported. An ND tensor of the same dtype and shape as output mean.
*/

REG_OP(InstanceNorm)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(gamma, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(beta, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(mean, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(variance, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .ATTR(data_format, String, "NDHWC")
    .ATTR(epsilon, Float, 1e-6f)
    .OP_END_FACTORY_REG(InstanceNorm)

/**
* @brief LayerNormGrad operator interface implementation \n
*  calculating: dy, x, variance, mean, gamma \n
*  rstd = 1.0 / sqrt(variance + eps) \n
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
* @li variance: A tensor. Third output of forward propagation, indicating the variance value of input.
* Must be one of the following types: float16, float32, bfloat16. The format must be ND.
* Has the same shape as mean, which is [A1,...,Ai,1,...,1], where there are j 1s after Ai,
* and j is the length of the axis that requires normalization.
* Has the same type and format as mean.
* @li mean: A tensor. Second output of forward propagation, indicating the mean value of input.
* Must be one of the following types: float16, float32, bfloat16. The format must be ND.
* Has the same shape as variance, which is [A1,...,Ai,1,...,1], where there are j 1s after Ai,
* and j is the length of the axis that requires normalization.
* Has the same type and format as variance.
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
* calculating: x, gamma, beta \n
* mean  = np.mean(x, reduce_axis, keepdims=True) \n
* rstd = np.rsqrt(np.mean(np.power((x - mean),2), reduce_axis, keepdims=True) + epsilon)) \n
* y = gamma*((x - mean) * rstd) + beta

* @par Inputs:
* Four inputs, including:
* @li x: A ND Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li normalized_shape: A ND Tensor. Must be one of the following types: int32, int64
* @li gamma: A ND Tensor. Must be one of the following types: float16, float32, bfloat16. Shape is normalized_shape.
* @li beta: A ND Tensor. Must be one of the following types: float16, float32, bfloat16. Shape is normalized_shape.\n

* @par Attributes:
* @li epsilon: An optional attribute, the type is float32. Defaults to 1e-5 . \n

* @par Outputs:
* Three outputs, including:
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
* @brief QuantizeAddLayerNorm operator interface implementation.

* @par Inputs
* @li x1: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li x2: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li gamma: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li beta: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li bias: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li scales: A Tensor. Support dtype: [float32, bfloat16], support format: [ND].
* @li zero_points: A optional Tensor. Support dtype: [float32, bfloat16], support format: [ND].

* @par Attributes
* @li dtype: A required attribute, the type is int. No defaults value.
* @li axis: A optional attribute, the type is int. Defaults to -1.
* @li epsilon: A optional attribute, the type is float. Defaults to 1e-5.
* @li additional_output: A optional attribute, the type is bool. Defaults to false.

* @par Outputs
* @li y: A Tensor. Support dtype: [int8], support format: [ND].
* @li x: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
*/
REG_OP(QuantizeAddLayerNorm)
    .INPUT(x1, ge::TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x2, ge::TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(gamma, ge::TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(beta, ge::TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(bias, ge::TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(scales, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(zero_points, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, ge::TensorType({DT_INT8, DT_INT8, DT_INT8}))
    .OUTPUT(x, ge::TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(dtype, Int)
    .ATTR(axis, Int, -1)
    .ATTR(epsilon, Float, 1e-5f)
    .ATTR(additional_output, Bool, false)
    .OP_END_FACTORY_REG(QuantizeAddLayerNorm)

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
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(rstd, TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(dx, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(dgamma, TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OP_END_FACTORY_REG(RmsNormGrad)

/**
* @brief RmsNormQuantV2 operator interface implementation.
* Calculating input: x, gamma, scales1, scales2, zero_points1, zero_points2 \n
* Calculating process: \n
*  rstd = np.rsqrt(np.mean(np.power(x, 2), reduce_axis, keepdims=True) + epsilon)) \n
*  rmsnorm_out = x * rstd * gamma \n
*  if div_mode is true: \n
*    y1 = round(rmsnorm_out / scales1 + zero_points1) \n
*    y2 = round(rmsnorm_out / scales2 + zero_points2) \n
*  if div_mode is false: \n
*    y1 = round(rmsnorm_out * scales1 + zero_points1) \n
*    y2 = round(rmsnorm_out * scales2 + zero_points2) \n

* @par Inputs
* @li x: A tensor. Input x for the add operation.
*         Support dtype: float32/float16/bfloat16, support format: ND.
* @li gamma: A tensor. Describing the weight of the rmsnorm operation.
*            Support dtype: float32/float16/bfloat16, support format: ND.
* @li scales1: A tensor. Describing the weight of the first quant operation.
*              Support dtype: float32/float16/bfloat16, support format: ND.
* @li scales2: An optional input tensor. Describing the weight of the secend quant operation.
*              Support dtype: float32/float16/bfloat16, support format: ND.
* @li zero_points1: An optional input tensor. Describing the bias of the first quant operation.
*                   Support dtype: int32/int8/float32/float16/bfloat16, support format: ND.
* @li zero_points2: An optional input tensor. Describing the bias of the secend quant operation.
*                   Support dtype: int32/int8/float32/float16/bfloat16, support format: ND.

* @par Attributes
* @li epsilon: An optional attribute. Describing the epsilon of the rmsnorm operation.
*              The type is float. Defaults to 1e-6.
* @li div_mode: An optional attribute. When div_mode is true, the quant opertaion uses division, otherwise, uses
multiplication.
*               The type is bool. Defaults to true.
* @li dst_type: An optional int32. Output y data type enum value. Support DT_INT8, DT_INT4, DT_HIFLOAT8, DT_FP8_E5M2, 
*               DT_FP8_E4M3FN. Defaults to DT_INT8.
* @par Outputs
* @li y1: A tensor. Describing the output of the first quant operation.
*                   Support dtype: int8/hifloat8/float8e5m2/float8e4m3fn, support format: ND.
* @li y2: A tensor. Describing the output of the second quant operation.
*                   Support dtype: int8/hifloat8/float8e5m2/float8e4m3fn, support format: ND.
*/

REG_OP(RmsNormQuantV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(scales1, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(scales2, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(zero_points1, TensorType({DT_INT32, DT_INT8, DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(zero_points2, TensorType({DT_INT32, DT_INT8, DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(beta, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y1, TensorType({DT_INT8, DT_INT4, DT_HIFLOAT8, DT_FP8_E5M2, DT_FP8_E4M3FN}))
    .OUTPUT(y2, TensorType({DT_INT8, DT_INT4, DT_HIFLOAT8, DT_FP8_E5M2, DT_FP8_E4M3FN}))
    .ATTR(epsilon, Float, 1e-6f)
    .ATTR(div_mode, Bool, true)
    .ATTR(dst_type, Int, DT_INT8)
    .OP_END_FACTORY_REG(RmsNormQuantV2)

/**
*@brief A fusion operator for bert lamb.

*@par Inputs:
* Ten inputs, including:
*@li input0: A ND Tensor specifying gradient. Support float16, float32, bfloat16.
*@li input1: A ND Tensor specifying second moment. Values must be greater than or equal to 0. Support float16, float32,
bfloat16.
*@li input2: A ND Tensor specifying first moment. Support float16, float32, bfloat16.
*@li input3: A ND Tensor specifying parameters to be updated. Support float16, float32, bfloat16.
*@li input4: A ND Tensor specifying the learning rate. Support float16, float32, bfloat16.
*@li mul0_x: A ND Tensor specifying beta1. Values must range from 0 to 1. Support float16, float32, bfloat16.
*@li mul1_x: A ND Tensor specifying 1 - beta1. Values must range from 0 to 1. Support float16, float32, bfloat16.
*@li mul2_x: A ND Tensor specifying beta2. Values must range from 0 to 1. Support float16, float32, bfloat16.
*@li mul3_x: A ND Tensor specifying 1 - beta2. Values must range from 0 to 1. Support float16, float32, bfloat16.
*@li add2_y: A ND Tensor ued for improving numerical stability. Support float16, float32, bfloat16. \n

*@par Outputs:
* Three outputs, including:
*@li output0: A ND Tensor specifying updated second moment. Support float16, float32, bfloat16.
*@li output1: A ND Tensor specifying updated first moment. Support float16, float32, bfloat16.
*@li output2: A ND Tensor specifying updated parameters. Support float16, float32, bfloat16. \n
*\n
* output0 = input0^2 * mul3_x + input1 * mul2_x
*\n
* output1 = input2 * mul0_x + input0 * mul1_x
*\n
* output2 = input3 - (output1 / sqrt(output0) + add2_y) * input4
*\n
* @par attention Constraints:
* @li All input shapes must can be broadcasted to the output shapes.
* @li All outputs must have the same shape and type as inputs.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(AdamApplyOne)
    .INPUT(input0, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(input1, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(input2, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(input3, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(input4, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mul0_x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mul1_x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mul2_x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mul3_x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(add2_y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(output0, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(output1, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(output2, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(AdamApplyOne)

/**
*@brief A fusion operator for bert lamb.

*@par Inputs:
*Eleven inputs, including:
* @li input0: A ND tensor specifying gradient of the input. Support float16, float32, bfloat16.
* @li input1: A ND tensor specifying second moment. Support float16, float32, bfloat16.
* @li input2: A ND tensor specifying first moment. Support float16, float32, bfloat16.
* @li input3: A ND tensor specifying parameters to be updated. Support float16, float32, bfloat16.
* @li input4: A ND tensor specifying the learning rate. Values must be greater than 0. Support float16, float32,
bfloat16.
* @li mul0_x: A ND tensor specifying beta1. Values must be greater than 0. Support float16, float32, bfloat16.
* @li mul1_x: A ND tensor specifying 1 - beta1. Values must range from 0 to 1. Support float16, float32, bfloat16.
* @li mul2_x: A ND tensor specifying beta2. Values must be greater than 0. Support float16, float32, bfloat16.
* @li mul3_x: A ND tensor specifying 1 - beta2. Values must range from 0 to 1. Support float16, float32, bfloat16.
* @li mul4_x: A ND tensor specifying weight. Values must be greater than or equal to 0. Support float16, float32,
bfloat16.
* @li add2_y: A ND tensor ued for improving numerical stability. Values must be greater than 0. Support float16,
float32, bfloat16. \n

*@par Outputs:
*Three outputs, including:
* @li output0: A ND tensor specifying updated second moment. Support float16, float32, bfloat16.
* @li output1: A ND tensor specifying updated first moment. Support float16, float32, bfloat16.
* @li output2: A ND tensor specifying updated parameters. Support float16, float32, bfloat16. \n
*\n
* output0 = input0^2 * mul3_x + input1 * mul2_x
*\n
* output1 = input2 * mul0_x + input0 * mul1_x
*\n
* output2 = input3 - (output1 / sqrt(output0) + add2_y + input3 * mul4_x) * input4

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use. \n

*@par attention Constraints:
*@li The empty tensor is not allowed in the shape of all inputs. \n
*@li The first four input shapes must satisfy the broadcast relationship.
*@li All output shapes must be the same with the input broadcasted shape. \n
*@li All input and output data types must be consistent.
*/
REG_OP(AdamApplyOneWithDecay)
    .INPUT(input0, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(input1, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(input2, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(input3, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(input4, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mul0_x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mul1_x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mul2_x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mul3_x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mul4_x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(add2_y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(output0, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(output1, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(output2, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(AdamApplyOneWithDecay)

/**
*@brief A fusion operator for bert lamb.

*@par Inputs:
*Eleven inputs, including:
* @li input0: A ND tensor specifying gradient of the input. Support float16, float32, bfloat16.
* @li input1: A ND tensor specifying second moment. Support float16, float32, bfloat16.
* @li input2: A ND tensor specifying first moment. Support float16, float32, bfloat16.
* @li input3: A ND tensor specifying parameters to be updated. Support float16, float32, bfloat16.
* @li input4: A ND tensor specifying the learning rate. Values must be greater than 0. Shape size must be equal to 1.
Support float16, float32, bfloat16.
* @li mul0_x: A ND tensor specifying beta1. Values must be greater than 0. Shape size must be equal to 1. Support
float16, float32, bfloat16.
* @li mul1_x: A ND tensor specifying 1 - beta1. Values must range from 0 to 1. Shape size must be equal to 1. Support
float16, float32, bfloat16.
* @li mul2_x: A ND tensor specifying beta2. Values must be greater than 0. Shape size must be equal to 1. Support
float16, float32, bfloat16.
* @li mul3_x: A ND tensor specifying 1 - beta2. Values must range from 0 to 1. Shape size must be equal to 1. Support
float16, float32, bfloat16.
* @li mul4_x: A ND tensor specifying weight. Values must be greater than or equal to 0. Support float16, float32,
bfloat16.
* @li add2_y: A ND tensor ued for improving numerical stability. Values must be greater than 0. Support float16,
float32, bfloat16. \n

*@par Outputs:
*Three outputs, including:
* @li input1: A ND tensor specifying updated second moment. Support float16, float32, bfloat16.
* @li input2: A ND tensor specifying updated first moment. Support float16, float32, bfloat16.
* @li input3: A ND tensor specifying updated parameters. Support float16, float32, bfloat16. \n
*\n
* output0 = input0^2 * mul3_x + input1 * mul2_x
*\n
* output1 = input2 * mul0_x + input0 * mul1_x
*\n
* output2 = input3 - (output1 / sqrt(output0) + add2_y + input3 * mul4_x) * input4

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(AdamApplyOneWithDecayAssign)
    .INPUT(input0, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(input1, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(input2, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(input3, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(input4, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mul0_x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mul1_x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mul2_x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mul3_x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mul4_x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(add2_y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(input1, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(input2, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(input3, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(AdamApplyOneWithDecayAssign)

/**
*@brief The main function of the advcance_step operator is to advance the inference step in vLLM, that is,
* update the model status and generate new inputTokens, inputPostions, seqLen, and slotMapping in each generation step.
* Improves the efficiency of vLLM inference. \n

*@par Inputs:
* Six inputs, including:
*@li input_tokens: A 1-D input tensor. When spec_token and accepted_num are None, length equal to num_seqs. When spec_token and accepted_num are NOT None, length equal to num_seqs * (spec_num + 1). Must be int64 type. Must be int64 type. Format is ND.
*@li sampled_token_ids: A 2-D input tensor. When spec_token and accepted_num are None, the first dim equal to num_queries and the second dim equal to one. When spec_token and accepted_num are NOT None, the first dim equal to num_seqs and the second dim equal to spec_num+1. Must be int64 type.
* Must be int64 type. Format is ND.
*@li input_positions: A 1-D input tensor. When spec_token and accepted_num are None, length equal to num_seqs. When spec_token and accepted_num are NOT None, length equal to num_seqs * (spec_num + 1). Must be int64 type. Format is ND.
*@li seq_lens: A 1-D input tensor. When spec_token and accepted_num are None, length equal to num_seqs. When spec_token and accepted_num are NOT None, length equal to num_seqs * (spec_num + 1). Must be int64 type. Must be int64 type. Format is ND.
*@li slot_mapping: A 1-D input tensor. When spec_token and accepted_num are None, length equal to num_seqs. When spec_token and accepted_num are NOT None, length equal to num_seqs * (spec_num + 1). Must be int64 type. Must be int64 type. Format is ND.
*@li block_tables: When spec_token and accepted_num are None, A 1-D input tensor, and length equal to num_seqs. When spec_token and accepted_num are NOT None, A 2-D input tensor, the first dim equal to num_seqs and the second dim equal to spec_num+1. Must be int64 type. Format is ND.
*@li spec_token: A 2-D optional input tensor, which the first dim equal to num_seqs and the second dim equal to spec_num. Must be int64 type. Format is ND.
*@li accepted_num: A 1-D optional input tensor, and length equal to num_seqs. Must be int64 type. Format is ND. \n

*@par Attributes:
*@li num_seqs: A required Int, which equal to the length of input_tokens, input_positions, seq_lens,
* slot_mapping and block_tables. The value of it must bigger than 0.
*@li num_queries: A required Int, which equal to the length of sampled_token_ids's first dim.
* The value of it must bigger than 0.
*@li block_size: A required Int, which means the basic block length of each block. The value of it must bigger than 0.
\n

*@par Outputs:
*@li input_tokens: A 1-D output tensor. The input tensor input_tokens will self-updating and save as itself.
* Must be int64 type. Format is ND.
*@li input_positions: A 1-D output tensor. The input tensor input_positions will self-updating and save as itself.
* Must be int64 type. Format is ND.
*@li seq_lens: A 1-D output tensor. The input tensor seq_lens will self-updating and save as itself.
* Must be int64 type. Format is ND.
*@li slot_mapping: A 1-D output tensor. The input tensor slot_mapping will self-updating and save as itself.
* Must be int64 type. Format is ND. \n

*/

REG_OP(AdvanceStep)
    .INPUT(input_tokens, TensorType({DT_INT64}))
    .INPUT(sampled_token_ids, TensorType({DT_INT64}))
    .INPUT(input_positions, TensorType({DT_INT64}))
    .INPUT(seq_lens, TensorType({DT_INT64}))
    .INPUT(slot_mapping, TensorType({DT_INT64}))
    .INPUT(block_tables, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(spec_token, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(accepted_num, TensorType({DT_INT64}))
    .OUTPUT(input_tokens, TensorType({DT_INT64}))
    .OUTPUT(input_positions, TensorType({DT_INT64}))
    .OUTPUT(seq_lens, TensorType({DT_INT64}))
    .OUTPUT(slot_mapping, TensorType({DT_INT64}))
    .REQUIRED_ATTR(num_seqs, Int)
    .REQUIRED_ATTR(num_queries, Int)
    .REQUIRED_ATTR(block_size, Int)
    .OP_END_FACTORY_REG(AdvanceStep)

/**
*@brief Updates "var" according to the adagrad scheme.
*   accum += grad * grad
*   var -= lr * grad * (1 / sqrt(accum))
*
*@attention Constraints:
*@li The input and output tensors must have the same shape.
*
*@par Inputs:
*@li var: A mutable tensor. Should be from a Variable(). Support float16, bfloat16 and float32.
*@li accum: A mutable tensor. Has the same type as "var".
*     Should be from a Variable().
*@li lr: A scalar. Has the same type as "var".
*@li grad: A tensor for the gradient. Has the same type as "var".
*
*@par Attributes:
*@li update_slots: An optional bool. Defaults to "True". If "True", the accum tensor will be updated.
*@li use_locking: An optional bool. Defaults to "False".
*     If "True", updating of the "var", "ms", and "mom" tensors is protected
*     by a lock; otherwise the behavior is undefined, but may exhibit less
*     contention.
*
*@par Outputs:
*@li var: A mutable tensor. Has the same type as input "var".
*@li accum: A mutable tensor. Has the same type as input "var".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ApplyAdagrad.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ApplyAdagrad instead.
*/
REG_OP(ApplyAdagradD)
    .INPUT(var, TensorType::NumberType())
    .INPUT(accum, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .OUTPUT(accum, TensorType::NumberType())
    .ATTR(update_slots, Bool, true)
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyAdagradD)

/**
* @brief Updates "var" according to the AdamW algorithm. For clarity, the output variable is suffixed with "_out" in the fomulas below (e.g. m_out).
* @code{.c}
*  if maximize:
*       gt = -grad
*   else:
*       gt = grad
*   m_out = m * beta1 - (beta1 - 1) * gt
*   v_out = v * beta2 - (beta2 - 1) * gt * gt
*   var_t = var * (1 + (-lr * weight_decay))
*   beta1_power_out = beta1_power * beta1
*   beta2_power_out = beta2_power * beta2
*   if amsgrad:
*       max_grad_norm_out = max(max_grad_norm, v_out)
*       denom = sqrt(-max_grad_norm_out / (beta2_power_out -1)) + epsilon
*   else:
*       denom = sqrt(-v_out / (beta2_power_out -1)) + epsilon
*   var_out = var_t + ((lr / (beta1_power_out -1)) * (m_out / denom))
* @endcode
*
* @par Inputs:
* @li var: A tensor of ND, dtype is float16 bfloat16 or float32.
*     Specifying parameters to be updated.
*     Should be from a Variable().
* @li m: A tensor of ND, dtype is float16 bfloat16 or float32.
*     Specifying first moment.
*     Should be from a Variable().
* @li v: A tensor of ND, dtype is float16 bfloat16 or float32.
*     Specifying second moment. Values must be greater than or equal to 0.
*     Should be from a Variable().
* @li beta1_power: A scalar of the same type as "var", value is beta1^(step-1), between 0 and 1.
* @li beta2_power: A scalar of the same type as "var", value is beta2^(step-1), between 0 and 1.
* @li lr: Specifying learning_rate. A scalar of the same type as "var", value is between 0 and 1 generally (e.g. 1e-3).
* @li weight_decay: Specifying weight decay. A scalar of the same type as "var", value is between 0 and 1 generally (e.g. 1e-2).
* @li beta1: Specifying beta1. A scalar of the same type as "var", value is between 0 and 1 (e.g. 0.9).
* @li beta2: Specifying beta2. A scalar of the same type as "var", value is between 0 and 1 (e.g. 0.999).
* @li epsilon: A scalar of the same type as "var", used to improve numerical stability.
*     Should be a minimum value greater than 0 (e.g. 1e-8).
* @li grad: A tensor of the same type as "var", dtype is float16 bfloat16 or float32.
*     Specifying gradient.
* @li max_grad_norm: A mutable tensor of the same type as "var", an optional input,
*     dtype is float16 bfloat16 or float32.
*     Should be from a Variable().
*
* @par Attributes:
* @li amsgrad: An optional bool. Specifying using max_grad_norm in the calculation. Defaults to "False", only support "False".
* @li maximize: An optional bool. Specifying maximize the objective with respect to the params, instead of minimizing. Defaults to "False".
*
* @par Outputs:
* @li var: A mutable tensor. Has the same shape and type as input "var".
* @li m: A mutable tensor. Has the same shape and type as input "m".
* @li v: A mutable tensor. Has the same shape and type as input "v".
*
* @par attention Constraints:
* @li The input tensors must have the same shape.
* @li Ensure the argument for the square root is non-negative.
*/
REG_OP(ApplyAdamW)
    .INPUT(var, TensorType::NumberType())
    .INPUT(m, TensorType::NumberType())
    .INPUT(v, TensorType::NumberType())
    .INPUT(beta1_power, TensorType::NumberType())
    .INPUT(beta2_power, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(weight_decay, TensorType::NumberType())
    .INPUT(beta1, TensorType::NumberType())
    .INPUT(beta2, TensorType::NumberType())
    .INPUT(epsilon, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OPTIONAL_INPUT(max_grad_norm, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .OUTPUT(m, TensorType::NumberType())
    .OUTPUT(v, TensorType::NumberType())
    .ATTR(amsgrad, Bool, false)
    .ATTR(maximize, Bool, false)
    .OP_END_FACTORY_REG(ApplyAdamW)

/**
* @brief Updates "var" "m" "v" and "max_grad_norm" according to the AdamWV2 algorithm.
*
* @attention Constraints:
*  The input tensors must have the same shape, except for the step. The shape of step must be (1,).
*  When the data types of the input tensors var,m,v,grad,max_grad_norm are the same, the data type can be
*  float16,bfloat16 or float32.
*  The data types of the input tensors var,m and v must be the same.For example,if var tensor is float16,
*  the data types of m and v must also be float16.
*  The data tytpes of the input tensors grad and max_grad_norm must be the same.For example,if grad tensor is float16,
*  the data types of max_grad_norm must also be float16.
*  When data type of the input tensor var,m and v are different with input tensor grad and max_grad_norm,
*  the data types of var,m and v can only be float32,and the data type of grad and max_grad_norm tensor
*  can only be float16 or bfloat16.
*
* @par Inputs:
* @li var: A Tensor, dtype is float16 bfloat16 or float32, default is ND.
* @li m: A Tensor of the same type as "var", default is ND.
* @li v: A Tensor of the same type as "var", default is ND.
* @li grad: A Tensor, dtype is float16 bfloat16 or float32, for the gradient, default is ND.
* @li step: A Tensor, dtype is float32 or int64.
* @li max_grad_norm: An optional Tensor of the same type as "grad", default is ND.
*
* @par Attributes:
* @li lr: A optional float, default is 0.1
* @li beta1: A optional float, default is 0.1
* @li beta2: A optional float,default is 0.1
* @li weight_decay: A optional float, default is 0.1
* @li eps: A optional float, default is 0.1
* @li amsgrad: A optional bool, default is false.
* @li maximize: A optional bool, default is false.\n
*/
REG_OP(ApplyAdamWV2)
    .INPUT(var, TensorType::FLOAT())
    .INPUT(m, TensorType::FLOAT())
    .INPUT(v, TensorType::FLOAT())
    .INPUT(grad, TensorType::FLOAT())
    .INPUT(step, TensorType({DT_FLOAT, DT_INT64}))
    .OPTIONAL_INPUT(max_grad_norm, TensorType::FLOAT())
    .ATTR(lr, Float, 0.1f)
    .ATTR(beta1, Float, 0.1f)
    .ATTR(beta2, Float, 0.1f)
    .ATTR(weight_decay, Float, 0.1f)
    .ATTR(eps, Float, 0.1f)
    .ATTR(amsgrad, Bool, false)
    .ATTR(maximize, Bool, false)
    .OP_END_FACTORY_REG(ApplyAdamWV2)

/**
*@brief Updates "var" according to the Ftrl-proximal scheme.
*@code{.c}
* accum_new = accum + grad * grad
* linear += grad - (accum_new^(-lr_power) - accum^(-lr_power)) * var / lr
* quadratic = accum_new^(-lr_power) / lr + 2 * l2
* var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
* accum = accum_new
*@endcode
*
*@par Inputs:
* @li var: A tensor of ND. Must be of dtype float16, float32 or bfloat16.
*     Specifying parameters to be updated.
*     Should be from a Variable().
* @li accum: A tensor of ND. Must be of the same shape and dtype as "var".
*     Specifying gradient accumulation.
*     Should be from a Variable().
* @li linear: A tensor of ND. Must be of the same shape and dtype as "var".
*     Specifying correction item.
*     Should be from a Variable().
* @li grad: A tensor of ND. Must be of the same shape and dtype as "var", specifying gradient.
* @li lr: A scalar of the same dtype as "var", specifying the scaling factor.
* @li l1: A scalar of the same dtype as "var", specifying L1 regulariation.
* @li l2: A scalar of the same dtype as "var", specifying L2 regulariation.
* @li lr_power: A scalar of the same dtype as "var", for the scaling factor.
*
*@par Attributes:
*use_locking: An optional bool. Defaults to "False". Only support "False".
*     If "True", updating of the "var" tensors will be protected by a lock;
*     otherwise the behavior is undefined, but may exhibit less contention.
*
*@par Outputs:
*var: A mutable Tensor. Must be of the same shape and dtype as input "var".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ApplyFtrl.
*/
REG_OP(ApplyFtrl)
    .INPUT(var, TensorType::NumberType())
    .INPUT(accum, TensorType::NumberType())
    .INPUT(linear, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(l1, TensorType::NumberType())
    .INPUT(l2, TensorType::NumberType())
    .INPUT(lr_power, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyFtrl)

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
* @li bias_correction: A bool. Set to "true" for bias correction and "false" for no correction. Optional and defaults to
"true".
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
*@brief Updates "var" by subtracting 'alpha' * 'delta' from it.
*   var -= delta * alpha
*
*@attention Constraints:
*  the input tensors must have the same shape.
*
*@par Inputs:
*@li var: A mutable tensor. Should be from a Variable().
*@li alpha: A scalar. Has the same type as "var".
*@li delta: A tensor for the change. Has the same type as "var".
*
*@par Attributes:
* use_locking: An optional bool. Defaults to "False".
*     If "True", updating of the "var" tensors is protected
*     by a lock; otherwise the behavior is undefined, but may exhibit less
*     contention.
*
*@par Outputs:
* var: A mutable tensor. Has the same type as input "var".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ApplyGradientDescent.
*
*/
REG_OP(ApplyGradientDescent)
    .INPUT(var, TensorType::NumberType())
    .INPUT(alpha, TensorType::NumberType())
    .INPUT(delta, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyGradientDescent)

/**
*@brief Updates "var" according to the momentum scheme. Set use_nesterov = True if you
*  want to use Nesterov momentum. \n
*  computing process:
*@code{.c}
*  accum = accum * momentum + grad
*  if use_nesterov {
*      var -= grad * lr + accum * momentum * lr
*  } else {
*      var -= lr * accum
*  }
*@endcode
*
*@par Inputs:
*@li var: A mutable tensor of ND. Must be of dtype float16, float32 or bfloat16.
*    Specifying parameters to be updated.
*    Should be from a Variable().
*@li accum: A mutable tensor of ND. Must be of the same shape and dtype as "var".
*    Specifying gradient accumulation.
*    Should be from a Variable().
*@li lr: A scalar. Must be of the same type as "var". Specifying the learning rate.
*@li grad: A tensor of ND. Must be of the same shape and type as "var". Specifying the gradient.
*@li momentum: A scalar. Must be of the same type as "var". Specifying the momentum.

*@par Attributes:
*@li use_nesterov: An optional bool. Defaults to "False".
*    If "True", the tensor var will be updated as Nesterov momentum.
*
*@li use_locking: An optional bool. Defaults to "False".
*    If "True", updating of the "var" and "accum" tensors is protected by a lock;
*    otherwise the behavior is undefined, but may exhibit less contention.
*
*@par Outputs:
* var: A mutable tensor. Has the same shape, dtype and format as input "var".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ApplyMomentum.
*
*/

REG_OP(ApplyMomentum)
    .INPUT(var, TensorType::NumberType())
    .INPUT(accum, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .INPUT(momentum, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .ATTR(use_nesterov, Bool, false)
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyMomentum)

/**
* @brief Applies a 3D adaptive average pooling over
* an input signal composed of several input planes.

* @par Inputs:
* One input, including:
* @li x: A Tensor. Must be one of the following data types:
*     float16, bfloat16, float32. \n
* For Ascend 950PR/Ascend 950DT: Support Format: [NDHWC, NCDHW].

* @par Attributes:
* @li output_size: A required list of 3 ints
* specifying the size (D,H,W) of the output tensor. \n
* @li data_format: An optional string, Specify the data format of the input and
* output data. either "NDHWC"(default), or "NCDHW" . \n
* For Ascend 950PR/Ascend 950DT: Supported "NDHWC", and "NCDHW".

* @par Outputs:
* @li y: A Tensor. Has the same data type and the same format as "x" \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator AdaptiveAvgPool3d.
*/
REG_OP(AdaptiveAvgPool3d)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(output_size, ListInt)
    .ATTR(data_format, String, "NDHWC")
    .OP_END_FACTORY_REG(AdaptiveAvgPool3d)

/**
* @brief Applies a 3D adaptive max pooling over
* an input signal composed of several input planes.
* The supported data formats are NCDHW and NCHW.

* @par Inputs:
* One input, including:
* x: A Tensor. Must be one of the following data types:
*     float16, bfloat16, float32. \n

* @par Attributes:
* @li output_size: A required list of 3 ints,
* specifying the shape of the output tensor. \n
* @li indices_dtype: An optional int, default value is 3.
* (3 is int32, 9 is int64) \n

* @par Outputs:
* @li y: A Tensor. Has the same data type as "x". \n
* @li indices: A Tensor. Has the same shape as "x". \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator AdaptiveMaxPool3d.
*/
REG_OP(AdaptiveMaxPool3d)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(indices, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(output_size, ListInt)
    .ATTR(indices_dtype, Int, 3)
    .OP_END_FACTORY_REG(AdaptiveMaxPool3d)

/**
* @brief Performs average pooling on the input.

* @par Inputs:
* x: A 5D tensor. Supported type:float16. Additional support for float32, bfloat16 in Ascend 950 AI Processor.
* Support format: NDHWC. Additional support for NCDHW in Ascend 950 AI Processor. \n

* @par Attributes:
* @li ksize: List of ints that has length 1, 3 or 5. The size of the window
* for each dimension of the input tensor.
* @li strides: List of ints that has length 1, 3 or 5. The stride of the sliding
* window for each dimension of the input tensor.
* @li pads: List of ints that has length 1, 3 or 6, implicit zero paddings on both sides of each dimension
* of the input tensor.
* @li ceil_mode: When true, will use ceil instead of floor in the formula to
* compute the output shape. Default value is false.
* @li count_include_pad: When true, will include the zero-padding in the
* averaging calculation. Default value is true.
* @li divisor_override: if specified, it will be used as divisor, otherwise
* size of the pooling region will be used. Default value is 0.
* @li data_format: An optional string, format of input data. It supports "NDHWC"(default).
* Additional support for "NCDHW" in Ascend 950 AI Processor. \n

* @par Outputs:
* y: The average pooled output tensor. \n

* @attention Constraints:
* @li The ksize of the D, H and W dimensions should be greater than 0.
* For Atlas Training Series Product, Atlas A2 Training Series Product/Atlas 800I A2 Inference Product,
* Atlas A3 Training Series Product: The produce of the ksize in D, H and W dimensions
* should be greater than 0 and less than or equal to 255.  \n
* @li For Atlas Training Series Product, Atlas A2 Training Series Product/Atlas 800I A2 Inference Product,
* Atlas A3 Training Series Product: The stride of the D, H and W dimensions should be greater than 0 and
* smaller than 64.  \n
* For Ascend 950 AI Processor: The stride of the D, H and W dimensions should be greater than 0.
* @li The ouput "y" shape at the N and C dimensions should be equal with input "x" shape at same dimensions. The output
* shape at the D, H and W dimensions is calculated by below formula: \n
* @code{.c}
    if "ceil_mode" is False:
            out_depth = (in_depth + pad_front + pad_backend - ksize_d) / stride_d + 1
            out_height = (in_height + pad_top + pad_bottom - ksize_h) / stride_h + 1
            out_width = (in_width + pad_left + pad_right - ksize_w) / stride_w + 1
    else :
            out_depth = (in_depth + pad_front + pad_backend - ksize_d + stride_d - 1) / stride_d + 1
            out_height = (in_height + pad_top + pad_bottom - ksize_h + stride_h - 1) / stride_h + 1
            out_width = (in_width + in_width + pad_right - ksize_w + stride_w - 1) / stride_w + 1
            if  (out_depth - 1) * stride_d >= in_depth + pad_front :
                    out_depth = out_depth - 1
            if  (out_height - 1) * stride_h >= in_height + pad_top :
                    out_height = out_height - 1
            if  (out_width - 1) * stride_w >= in_width + in_width :
                    out_width = out_width - 1
  It not support out_depth < 0, out_height < 0 or out_width < 0.
* @endcode

* @par Third-party framework compatibility
* Compatible with the TensorFlow/Pytorch operator AvgPool3D.
*/
REG_OP(AvgPool3D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(ceil_mode, Bool, false)
    .ATTR(count_include_pad, Bool, true)
    .ATTR(divisor_override, Int, 0)
    .ATTR(data_format, String, "NDHWC")
    .OP_END_FACTORY_REG(AvgPool3D)

/**
* @brief Computes average pooling3d backwards gradients.

* @par Inputs:
* @li orig_input_shape: An one-dim tensor of type int32, which describes the
* original input shape [N,C,D,H,W] of forward AvgPool3D.
* @li grads: A 5D Tensor of backwards gradient. With the format "NDHWC",
* grads supports data type float16/float32/bfloat16.

* @par Attributes:
* @li ksize: List of ints. Describes the size of the window for each dimension of the input tensor.
* Restriction: "ksize" value is in the range [1, 255] and <= "orig_input_shape" for DHW dimensions. \n
* For Atlas Inference Series Product: "ksize" length is 5. \n
* For Atlas Training Series Product: "ksize" length is 5. \n
* For Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component: "ksize"
length is 3. \n
* For Atlas A3 Training Series Product/Atlas A3 Inference Series Product: "ksize" length is 3. \n
* For Ascend 950 AI Processor: "ksize" length is 3, without the limit of [1, 255] and must be greater than 0. \n
* @li strides: List of ints. The stride of the sliding window for each dimension of the input tensor.
* Restriction: "strides" value is in the range [1, 63]. \n
* For Atlas Inference Series Product: "strides" length is 5. \n
* For Atlas Training Series Product: "strides" length is 5. \n
* For Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component: "strides"
length is 3. \n
* For Atlas A3 Training Series Product/Atlas A3 Inference Series Product: "strides" length is 3. \n
* For Ascend 950 AI Processor: "strides" length is 3, without the limit of [1, 63] and must be greater than 0. \n
* @li pads: List of ints, implicit zero paddings on both sides of the input.
* Restriction: "pads" is in the range [0, ksize/2]. \n
* For Atlas Inference Series Product: "pads" length is 6. \n
* For Atlas Training Series Product: "pads" length is 6. \n
* For Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component: "pads"
length is 3. \n
* For Atlas A3 Training Series Product/Atlas A3 Inference Series Product: "pads" length is 3. \n
* For Ascend 950 AI Processor: "pads" length is 3. \n
* @li ceil_mode: An optional bool. When true, will use ceil instead of floor in the formula to
* compute the output shape. Default value false.
* @li count_include_pad: An optional bool. When true, will include the zero-padding in the
* averaging calculation. Otherwise, not include paddings in averaging. Default value true.
* @li divisor_override: An optional int, if specified, it will be used as divisor, otherwise
* size of the pooling region will be used. Default value 0, which means this attribute does not take effect.
* @li data_format: An optional string, the format of the input "grads". Defaults to "NDHWC".
* For Atlas Inference Series Product: support "NDHWC" and "NCDHW". \n
* For Atlas Training Series Product: support "NDHWC" and "NCDHW". \n
* For Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component: only
support "NDHWC". \n
* For Atlas A3 Training Series Product/Atlas A3 Inference Series Product: only support "NDHWC". \n
* For Ascend 950 AI Processor: only support "NDHWC". \n

* @par Outputs:
* output: A mutable tensor with the same shape as "orig_input_shape" and same type as "grads".

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator AvgPoolGrad.
*/

REG_OP(AvgPool3DGrad)
    .INPUT(orig_input_shape, TensorType({DT_INT32}))
    .INPUT(grads, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(output, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(ceil_mode, Bool, false)
    .ATTR(count_include_pad, Bool, true)
    .ATTR(divisor_override, Int, 0)
    .ATTR(data_format, String, "NDHWC")
    .OP_END_FACTORY_REG(AvgPool3DGrad)

/**
* @brief Performs max pooling on the input and outputs both max values and indices.

* @par Inputs:
* One input:
* x: A tensor of type bfloat16, float16, float32, the shape is [batch, channels, depth_in, height_in, width_in] or
  [batch, depth_in, height_in, width_in, channels].

* @par Attributes:
* @li ksize: A required list of int64 values,
* specifying the size of the window for each dimension of the input tensor.
* A list that has length 3.
* @li strides: A required list of int64 values,
* specifying the stride of the sliding window for each dimension of the input tensor.
* A list that has length 3.
* @li pads: A required list of int64 values,
* specifying the pad of the input feature map.
* A list that has length 3:
* 0 <= pads[0] <= (ksize[0]//2), 0 <= pads[1] <= (ksize[1]//2), 0 <= pads[2] <= (ksize[2]//2).
* @li dilation: A list that has length 3, default value is {1,1,1}.
* @li ceil_mode: When true, will use ceil instead of floor to compute the output shape, defaults to false.
* @li data_format: The value can be "NCDHW" or "NDHWC", defaults to "NCDHW".
* @li dtype: An optional int, default value is 3.  (3 is int32, 9 is int64)

* @par Outputs:
* @li y: A tensor has the same type and format as input "x", the shape is [batch, channels, depth_out, height_out, width_out] or
  [batch, depth_out, height_out, width_out, channels].
* @li argmax:  A tensor of type is int64 or int32, the shape is [batch, channels, depth_out, height_out, width_out] or
  [batch, depth_out, height_out, width_out, channels].

* @par Third-party framework compatibility
* Compatible with the PyTorch operator max_pool3d_with_indices.
*/

REG_OP(MaxPool3DWithArgmaxV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OUTPUT(argmax, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilation, ListInt, {1, 1, 1})
    .ATTR(ceil_mode, Bool, false)
    .ATTR(data_format, String, "NCDHW")
    .ATTR(dtype, Int, 3)
    .OP_END_FACTORY_REG(MaxPool3DWithArgmaxV2)

/**
* @brief Quantizes the input.

* @par Inputs:
* x: A tensor of type float16 or float32, specifying the input.
* The format must be NC1HWC0, FRACTAL_NZ, NDC1HWC0 or ND. Shape supports 1D ~ 8D.
* If "dst_type" is 29, the last dimension of the shape must be divisible by 2. \n

* @par Attributes:
* @li scale: A required float32, specifying the scaling ratio.
* @li offset: A required float32, specifying the offset.
* @li sqrt_mode: An optional bool, specifying whether to perform square on "scale", either "True" or "False".
* Defaults to "False".
* @li round_mode: An optional string, specifying the cast mode.
* The value range is [Round, Floor, Ceil, Trunc, Hybrid]. Defaults to "Round".
* @li dst_type: An optional int32, specifying the output data type.
* Defaults to "2", represents dtype "DT_INT8". "29" represents dtype "DT_INT4", "34" represents dtype "DT_HIFLOAT8",
"35" represents dtype "DT_FLOAT8_E5M2", "36" represents dtype "DT_FLOAT8_E4M3FN". \n

* @par Outputs:
* y: The quantized output tensor of type int8, int4, hifloat8, float8_e5m2 or float8_e4m3fn.
* The format must be NC1HWC0, FRACTAL_NZ, NDC1HWC0 or ND. Shape supports 1D ~ 8D.
* Has the same format and shape as input "x". \n

* @attention Constraints:
* @li round_mode value range is [Round, Floor, Ceil, Trunc, Hybrid]. \n
* Round: round to nearest, tie to even(c language rint). \n
* Floor: round to minus infinity(c language floor). \n
* Ceil: round to positive infinity(c language ceil). \n
* Trunc: round to zero(c language trunc). \n
* Hybrid: only valid when output dtype is hifloat8. \n
* The following constraints apply to products other than Ascend 950 AI Processor: \n
* @li When format is FRACTAL_NZ, shape supports 4D ~ 8D.
* @li When "x" is dynamic shape, shape [-2] is not supported.
* @li When "x" is dynamic shape, the data type of output "y" does not support int4.
* @li When the format of "x" is ND, the data type of output "y" does not support int4. \n

* @par Third-party framework compatibility
* It is a custom operator. It has no corresponding operator in Caffe.
*/
REG_OP(AscendQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .OUTPUT(y, TensorType({DT_INT8, DT_INT4, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .REQUIRED_ATTR(scale, Float)
    .REQUIRED_ATTR(offset, Float)
    .ATTR(sqrt_mode, Bool, false)
    .ATTR(round_mode, String, "Round")
    .ATTR(dst_type, Int, DT_INT8)
    .OP_END_FACTORY_REG(AscendQuant)

/**
* @brief Quantizes the input, scale/offset support broadcasting operations. \n

* @par Inputs:
* @li x: A required Tensor. Must be one of the following types: float16,
* float32, bfloat16. The format support ND. Shape support 1D ~ 8D. Specifying the input.
* @li scale: A required Tensor. Must be one of the following types: float16,
* float32, bfloat16. The format support ND. Shape support 1D ~ 8D. Specifying the scaling ratio.
* @li offset: An optional Tensor. Must be one of the following types: float16,
* float32, bfloat16. The format support ND. Shape support 1D ~ 8D. Shape is same as "scale". Specifying the offset. \n

* @par Attributes:
* @li sqrt_mode: An optional bool, specifying whether to perform square
* on "scale", either "True" or "False". Defaults to "False".
* @li round_mode: An optional string, specifying the cast type.
* The value range is ["round", "floor", "ceil", "trunc", "hybrid"]. Defaults to "round".
* @li dst_type: An optional int32, specifying the output data type.
* The value range is [DT_INT8, DT_INT4, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN]. Defaults to DT_INT8.
* @li axis: An optional int32, describes which dimension of the input tensor x to be processed. Default is -1.
* Suppose x's dim number is D, axis should be in [-D, D - 1].
* if D > 2, axis should be -1, -2, D-2 or D-1. \n

* @attention Constraints:
* @li When dst_type is DT_FLOAT8_E5M2 or DT_FLOAT8_E4M3FN, round_mode only supports "round".
* @li When dst_type is DT_HIFLOAT8, round_mode supports "round" and "hybrid".
* @li When dst_type is other value, round_mode supports "round", "floor", "ceil" and "trunc".
* @li If dst_type is DT_INT4, the last dimension of the shape must be divisible by 2.
* @li The dtype of x, scale and offset must be same.
* @li scale and offset's shape should be the same.
* @li scale and offset's dimension number should be 1 or the same with x.
* @li When scale's dimension number is 1, it should be 1 or the same with the dimension specified by attributes[axis] of
input x.
* @li When scale's dimension number is the same with x, the dimension specified by attributes[axis] shoule be 1 or the
same with x's, others should be 1. \n

* @par Outputs:
* y: The quantized output tensor of type int8, int4, hifloat8, float8_e5m2 or float8_e4m3fn. The format support ND.
* Shape support 1D ~ 8D. Has the same shape as input "x". Dtype should be the same as the attribute dst_type. \n
*/
REG_OP(AscendQuantV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(scale, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_INT8, DT_INT4, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .ATTR(sqrt_mode, Bool, false)
    .ATTR(round_mode, String, "round")
    .ATTR(dst_type, Int, DT_INT8)
    .ATTR(axis, Int, -1)
    .OP_END_FACTORY_REG(AscendQuantV2)

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
* @brief Combine Dequant + Swiglu + Quant.

* @par Inputs:
* Seven inputs, including:
* @li x: A tensor. Shape is (X..., H), dim must > 2, and H must be even. Type is int32, float16, bfloat16.
* @li weight_scale: Dequantization scale of weight. An optional tensor. Type is float32. Shape is (1..., H).
* @li activation_scale: Dequantization scale of activation. An optional tensor. Type is float32. Shape is  (X..., 1).
* @li bias: Bias for matmul. An optional tensor. Type is float16/bfloat16/int32/float32. Shape is (X..., H).
* @li quant_scale: Quantized scale. An optional tensor. Type is float16/bfloat16/float32. Shape is (1..., H).
* @li quant_offset: Quantized offset. An optional tensor. Type is float16/bfloat16/float32. Shape is (1..., H).
* @li group_index: Mean group index. An optional tensor. Type is int32/int64. Shape is (1,). \n

* @par Outputs:
* @li y: A tensor. Type is int8/fp8_e5m2/fp8_e4m3fn/fp4x2_e2m1/fp4x2_e1m2/hifloat8.
* @li scale: A tensor. Type is float32.

* @par Attributes:
* @li activate_left: Type is bool.
* The swi activate_left algorithm to use:
*     'false'(activate right) or 'true'(activate left), defalut is 'false'(activate right).
* @li quant_mode: Type is string. The quant mode to use: 'static' or 'dynamic', defalut is 'static'.
* @li dst_type: Type is Int32. Declare the output y dtype. Support 2:int8, 35:fp8_e5m2, 36:fp8_e4m3fn, 40:fp4x2_e2m1, 41:fp4x2_e1m2, Defaults to 2, only used for Ascend 950 AI Processors.
* @li round_mode: Type is String. The round mode to use: 'rint', 'round, 'floor', 'ceil', 'trunc', defalut is 'rint', only used for Ascend 950 AI Processors.
* @li activate_dim: Type is Int32. Describing the split dimension in Glu algorithm: value in [-xDim, xDim-1], defalut is -1, only used for Ascend 950 AI Processors.
* @li swiglu_mode: Type is int. Optional parameter, default is 0. The SWIGLU computation mode to use:
*     '0' (default) for standard SWIGLU, '1' for a variant using odd-even blocking, which requires support for clamp_limit, activation coefficient, and bias. This attribute is not supported in Ascend 950 AI Processors.
* @li clamp_limit: Type is float. Optional parameter, default is 7.0. The threshold limit for SWIGLU input. This attribute is not supported in Ascend 950 AI Processors.
* @li glu_alpha: Type is float. Optional parameter, default is 1.702. The activation coefficient for the GLU activation function. This attribute is not supported in Ascend 950 AI Processors.
* @li glu_bias: Type is float. Optional parameter, default is 1.0. The bias applied during SWIGLU linear computation. This attribute is not supported in Ascend 950 AI Processors.

* @attention Constraints:
* @li When the type of x is int32, weight_scale must be input.
* @li When the type of x is float16, bfloat16, weight_scale, activation_scale and bias must be None.
* @li When dst_type is int8, fp8_e5m2, fp8_e4m3fn, round_mode only supports 'rint'.
* @li When dst_type is fp4x2_e2m1 or fp4x2_e1m2, round_mode supports 'rint', 'round, 'floor', 'ceil' and 'trunc'.
* @li When dst_type is hifloat8, round_mode supports 'round'.
* @li The shape of activate_dim corresponding to x must be divisible by 2.
* @li When activate_dim is not the last dimension of x, group_index must be None.
* @li The input quant_offset is not supported in Ascend 950 AI Processors only.
* @li The type of output y is fp8_e5m2, fp8_e4m3fn, fp4x2_e2m1 and fp4x2_e1m2 only supported in Ascend 950 AI Processors.
* @li The attribute quant_mode is only supported 'dynamic' in Ascend 950 AI Processors.
* @li The attribute dst_type is only supported in Ascend 950 AI Processors.
* @li The attribute round_mode is only supported in Ascend 950 AI Processors.
* @li The attribute activate_dim is only supported in Ascend 950 AI Processors.
* @li The attribute swiglu_mode is not supported in Ascend 950 AI Processors.
* @li The attribute clamp_limit is not supported in Ascend 950 AI Processors.
* @li The attribute glu_alpha is not supported in Ascend 950 AI Processors.
* @li The attribute glu_bias is not supported in Ascend 950 AI Processors.

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(DequantSwigluQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16, DT_INT32}))
    .OPTIONAL_INPUT(weight_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(activation_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_BF16, DT_INT32, DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(quant_offset, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(group_index, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT8, DT_FP8_E4M3FN, DT_FP8_E5M2, DT_FP4X2_E2M1, DT_FP4X2_E1M2, DT_HIFLOAT8}))
    .OUTPUT(scale, TensorType({DT_FLOAT}))
    .ATTR(activate_left, Bool, false)
    .ATTR(quant_mode, String, "static")
    .ATTR(dst_type, Int, DT_INT8)
    .ATTR(round_mode, String, "rint")
    .ATTR(activate_dim, Int, -1)
    .ATTR(swiglu_mode, Int, 0)
    .ATTR(clamp_limit, Float, 7.0)
    .ATTR(glu_alpha, Float, 1.702)
    .ATTR(glu_bias, Float, 1.0)
    .OP_END_FACTORY_REG(DequantSwigluQuant)

/**
* @brief Online quantizes the input tensor per block.

* @par Inputs:
- x: A tensor of type float16 or bfloat16. Shape must be 2-dimensional.

* @par Attributes:
- min_scale: (Optional) Minimum scale value for quantization. Must be a positive float.
*   Defaults to 0.0.
- round_mode: (Optional) Quantization rounding mode. Valid values:
*   - "rint": Supported for FLOAT8_E5M2/FLOAT8_E4M3FN
*   - "round": Supported for HIFLOAT8 only
*   Defaults to "rint".
- dst_type: (Optional) Target data type enum value:
*   - 34: HIFLOAT8
*   - 35: FLOAT8_E5M2
*   - 36: FLOAT8_E4M3FN
*   Defaults to 35 (FLOAT8_E5M2).
- row_block_size: (Optional) Number of elements per block in row dimension.
* Only support 1, 128, 256, 512. Defaults to 1.
- col_block_size: (Optional) Number of elements per block in column dimension.
* Only support 64, 128, 192, 256. Defaults to 128.

* @par Outputs:
- y: Quantized tensor with same shape as input x. Data type depends on dst_type.
- scale: Scale tensor of type float. Shape is [ceil(x.rows/row_block_size), ceil(x.cols/col_block_size)].

* @par Third-party framework compatibility:
* Custom operator with no direct mapping in Caffe/ONNX/TensorFlow/PyTorch.
*/
REG_OP(DynamicBlockQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2}))
    .OUTPUT(scale, TensorType({DT_FLOAT}))
    .ATTR(min_scale, Float, 0.0)
    .ATTR(round_mode, String, "rint")
    .ATTR(dst_type, Int, DT_FLOAT8_E5M2)
    .ATTR(row_block_size, Int, 1)
    .ATTR(col_block_size, Int, 128)
    .OP_END_FACTORY_REG(DynamicBlockQuant)

/**
 * @brief Dynamic Quant. Performs pre-token symmetric dynamic quantization on input tensors.
 * @par Inputs:
 * @li x: A Tensor. Type is:DT_FLOAT16 or DT_BF16. For Atlas A2 Training Series Product/Atlas 800I A2 Inference
 * Product/A200I A2 Box Heterogeneous Component and Atlas A3 Training Series Product/Atlas A3 Inference Series Product.
 * Whose shape must be greater than 1. The data format support ND.
 * @li smooth_scales: An optional Tensor.
 * When group_index is null and smooth_scales is not null, shape is 1 Dims. Dim[0] is the last dimension of x.
 * When group_index is not null, smooth_scales can not be null, shape is 2 Dims. Dim[0] is the expert num(E). E must be not greater than 1024. Dim[1] is the last dimension of x.
 * The data type can be FLOAT16 or BFLOAT16. The data type must be the same as that of x.
 * The data format support ND.
 * @li group_index: An optional Tensor. Specifying the index of group. 1-D with shape
 * [E, ], the first dim of scale shape is same as the first dim of smooth_scales shape.
 * Must be one of the following types: int32. The format support ND. \n
 * @par Attributes:
 * dst_type: An optional int32. Output y data type enum value.
 * Support DT_INT4, DT_INT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8.
 * Defaults to DT_INT8. \n
 * @par Outputs:
 * @li y: A Tensor. Quantized output tensor, Shape is same as input x. If y dtype is int4, x last dim must be divisible by 2.
 * The format support ND. Type specified by dst_type, support INT4, INT8,
 * FLOAT8_E5M2, FLOAT8_E4M3FN, HIFLOAT8.
 * @li scale: A Tensor. Scale used for quantization.
 * Type is DT_FLOAT32. The format support ND.
 */
REG_OP(DynamicQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(smooth_scales, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(group_index, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT8, DT_INT4, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .OUTPUT(scale, TensorType({DT_FLOAT}))
    .ATTR(dst_type, Int, DT_INT8)
    .OP_END_FACTORY_REG(DynamicQuant)

/**
* @brief Multiplies dynamic quantize and sparse updates into a variable reference .

* @par Inputs:
* Five inputs, including:
* @li var: An ND Tensor.
* Must be one of the following types: int8
* @li var_scale: An ND Tensor.
* Must be one of the following types: float
* @li indices: An ND Tensor.
* Must be one of the following types: int32，int64
* @li updates: An ND Tensor .
* Must be one of the following types: bfloat16，float16
* @li smooth_scales: An ND optional Tensor .
* Must be one of the following types: bfloat16，float16 \n

* @par Attributes:
* @li axis: An optional attribute. Defaults to 0, not support -1.
* @li reduce: A required attribute, can be "update". \n

* @par Outputs:
* var: A Tensor. Has the same type and format as input "var" .
* var_scale: A Tensor. Has the same type and format as input "var_scale" . \n

* @par Third-party framework compatibility
* Compatible with the Mindspore operator Scatter.
*/
REG_OP(DynamicQuantUpdateScatter)
    .INPUT(var, TensorType({DT_INT8}))
    .INPUT(var_scale, TensorType({DT_FLOAT}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_BF16, DT_FLOAT16}))
    .OPTIONAL_INPUT(smooth_scales, TensorType({DT_BF16, DT_FLOAT16}))
    .OUTPUT(var, TensorType({DT_INT8}))
    .OUTPUT(var_scale, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(reduce, String)
    .ATTR(axis, Int, 0)
    .OP_END_FACTORY_REG(DynamicQuantUpdateScatter)

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
 * @brief Dynamic Quant V2. Performs pre-token/per-tensor asymmetric dynamic quantization on input tensors.
 * @par Inputs:
 * @li x: A tensor. Type is:DT_FLOAT16 or DT_BF16. For Atlas A2 Training Series Product/Atlas 800I A2 Inference
 * Product/A200I A2 Box Heterogeneous Component and Atlas A3 Training Series Product/Atlas A3 Inference Series Product.
 * Whose shape must be greater than 1. The data format support ND.
 * @li smooth_scales: An optional tensor.
 * When group_index is null, shape is 1 Dims. Dim[0] is the last dimension of x.
 * When group_index is not null, shape is 2 Dims. Dim[0] is the expert num(E). E must be not greater than 1024. Dim[1]
 * is the last dimension of x. The data type can be FLOAT16 or BFLOAT16. The data type must be the same as that of x.
 * The data format support ND.
 * @li group_index: An optional tensor. Specifying the index of group. 1-D with shape
 * [E, ].
 * The first dim of group_index shape is same as the first dim of smooth_scales shape.
 * Must be one of the following types: int32. The format support ND.
 * If group_index is not null, smooth_scales must be not null. \n
 * @par Attributes:
 * @li dst_type: An optional attribute of type int. Declare the output dtype.
 * Support DT_INT4, DT_INT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8.
 * Defaults to DT_INT8.
 * @li is_symmetrical: An optional attribute of type bool. Select whether to be symmetrical.
 * Defaults to false.
 * @li quant_mode: An optional attribute of type string. Specifies the mode of quantization.
 * Support "pertoken", "pertensor", "perchannel". Defaults to "pertoken". \n
 * @par Outputs:
 * @li y: A tensor. Quantized output tensor, Shape is same as input x. If y dtype is int4, x last dim must be divisible
 * by 2. The format support ND. Type specified by dst_type, support INT4, INT8, FLOAT8_E5M2, FLOAT8_E4M3FN, HIFLOAT8.
 * @li scale: A tensor. Scale used for quantization.
 * When quant_mode is "pertoken", shape is the same as the shape of x after removing the last dimension.
 * When quant_mode is "pertensor", shape is (1,).
 * When quant_mode is "perchannel", shape is the same as the shape of x after removing the second last dimension.
 * Type is DT_FLOAT32. The format support ND.
 * @li offset: A tensor. Offset used for quantization. Shape is the same as the shape of scale.
 * Type is DT_FLOAT32. The format support ND. Shape is same as scale. \n
 * @attention Constraints:
 * Warning: E should to be be not greater than multiplication result of x Dims after removing the last dimension(S).
 * The value of group_index should to be increasing, ranging from 0 to S. The last value is should to be S. Otherwise
 * the result is meaningless.
 */
REG_OP(DynamicQuantV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(smooth_scales, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(group_index, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT8, DT_INT4, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .OUTPUT(scale, TensorType({DT_FLOAT}))
    .OUTPUT(offset, TensorType({DT_FLOAT}))
    .ATTR(dst_type, Int, DT_INT8)
    .ATTR(is_symmetrical, Bool, false)
    .ATTR(quant_mode, String, "pertoken")
    .OP_END_FACTORY_REG(DynamicQuantV2)

/**
*@brief Fake-quantize the data of 'x' tensor with scale, zero_point, quant_min and quant_max. \n

*@par Inputs:
*Three inputs, including:
*@li x: A Tensor. Must be one of the following types: float16, float32.
*@li scale: A Tensor of type float32 or float16. Has the same type and format as "x".
*@li zero_point: A Tensor of type int32, float16 or float32.\n

*@par Attributes:
*@li axis: An required attribute of type int64.
*@li quant_min: An required attribute of type int64.
*@li quant_max: An required attribute of type int64.\n

*@par Outputs:
*y: A Tensor of type float32 or float16.
*mask: A Tensor of type bool. \n

*@par Third-party framework compatibility
* Compatible with Pytorch operator FakeQuantAffineCachemask.
*/

REG_OP(FakeQuantAffineCachemask)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(scale, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(zero_point, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(mask, TensorType({DT_BOOL}))
    .REQUIRED_ATTR(axis, Int)
    .REQUIRED_ATTR(quant_min, Int)
    .REQUIRED_ATTR(quant_max, Int)
    .OP_END_FACTORY_REG(FakeQuantAffineCachemask)

/**
 * @brief Flatness matters for LLM quantization. \n
 *
 * @par Inputs:
 * @li x: A tensor. The original data input. 3-D with shape [K, M, N]. Must be one of the following types:
 * float16, bfloat16. The format support ND.
 * @li kronecker_p1: A tensor. Input calculation matrix 1. 2-D with shape [M, M]. The value of M is same as input "x".
 * Must be one of the following types: float16, bfloat16. Has the same type as input "x".
 * The format support ND.
 * @li kronecker_p2: A tensor. Input calculation matrix 2. 2-D with shape [N, N]. The value of N is same as input "x".
 * Must be one of the following types: float16, bfloat16. Has the same type as input "x".
 * The format support ND. \n
 *
 * @par Outputs:
 * @li out: When the output data type is int4,the shape is consistent with the input parameter x.When the output data
 type is int32,the shape is [K,M,N/8].When the output data type is float4_e2m1,the shape is [K,M*N].
 * @li quant_scale: When the output data type is int4 or int32,the shape is [K].When the output data type is
 float4_e2m1,the shape is [K,ceilDiv(M*N,64),2].
 * The values of K,M,and N are same as that of "x". \n

 * @par Attributes:
 * clip_ratio: An optional float. Used to control the quantization cropping ratio. Defaults to 1. \n
 * dst_dtype: An optional int. Used to control the quantization dst_type. Defaults to 29. \n
 */
REG_OP(FlatQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(kronecker_p1, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(kronecker_p2, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(out, TensorType({DT_INT4, DT_FLOAT4_E2M1}))
    .OUTPUT(quant_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .ATTR(clip_ratio, Float, 1)
    .ATTR(dst_dtype, Int, DT_INT4)
    .OP_END_FACTORY_REG(FlatQuant)

/**
 * @brief Quantize feature map by group.
 * @par Inputs:
 * @li x: A Tensor. 2-D with shape [S, H]. Must be one of the following types:
 * float32, float16, bfloat16. The format support ND.
 * @li scale: A Tensor. Specifying the quantitation scale of x. 2-D with shape
 * [E, H], the second dim of scale shape is same as the second dim of x shape.
 * Must be one of the following types: float32, float16, bfloat16.
 * The format support ND.
 * @li group_index: A Tensor. Specifying the index of group. 1-D with shape
 * [E, ], the first dim of scale shape is same as the first dim of scale shape.
 * Must be one of the following types: int32, int64. The format support ND.
 * @li offset: A Tensor. Optional. Specifying the quantitation offset of x. 1-D
 * with shape [1, ] or 0-D with shape []. Must be one of the following types:
 * float32, float16, bfloat16. The dtype of offset should be same as scale.
 * The format support ND.
 * @par Outputs:
 * y: A 2-D Tensor. Shape is same as input x. The format support ND.
 * Must be one of the following types: int4, int8.
 * @par Attributes:
 * dst_type: An optional attribute of type int. Declare the output dtype.
 * Support DT_INT4, DT_INT8. Defaults to DT_INT8.
 * @attention Constraints:
 * @li If output y data type is INT4, the last dim of y shape should be
 * an even number.
 * @li Input group_index value should be in the range of [0, S] and be an
 * non-decreasing sequence. The last value of input group_index must be the
 * same as the first dim of x shape.
 */
REG_OP(GroupQuant)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(scale, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(group_index, TensorType({DT_INT32, DT_INT64}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_INT4, DT_INT8}))
    .ATTR(dst_type, Int, DT_INT8)
    .OP_END_FACTORY_REG(GroupQuant)

/**
* @brief IFMR(Input Feature Map Reconstruction).

* @par Inputs:
* @li data: A Tensor of feature map.
* @li data_min: A Tensor of min value of feature map.
* @li data_max: A Tensor of max value of feature map.
* @li cumsum: A Tensor of cumsum bin of data . \n

* @par Attributes:
* @li min_percentile: Min init percentile.
* @li max_percentile: Max init percentile.
* @li search_range: Search range.
* @li search_step: Step size of searching.
* @li with_offset: Whether using offset.
* @li quant_bits: Bits of quant, an optional attr, default value is 8. \n

* @par Outputs:
* @li scale: Optimal scale.
* @li offset: Optimal offset. \n

* @par Third-party framework compatibility
* Compatible with mindspore.
*/

REG_OP(IFMR)
    .INPUT(data, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(data_min, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(data_max, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(cumsum, TensorType({DT_INT32}))
    .OUTPUT(scale, TensorType({DT_FLOAT}))
    .OUTPUT(offset, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(min_percentile, Float)
    .REQUIRED_ATTR(max_percentile, Float)
    .REQUIRED_ATTR(search_range, ListFloat)
    .REQUIRED_ATTR(search_step, Float)
    .REQUIRED_ATTR(with_offset, Bool)
    .ATTR(quant_bits, Int, 8)
    .OP_END_FACTORY_REG(IFMR)

/**
* @brief Dequantizes the input tensor into a float tensor.
* [min_range, max_range] are float32 tensors that specify the range
* for "y".
* The "mode" attribute controls exactly which calculations are used to convert
* the float values to their quantized equivalents.
* @par Inputs:
* @li x: A Tensor. Must be one of the following types: qint8, quint8, qint32, quint16, qint16.
* Shape suport 1D ~ 8D. The format support ND or NC1HWC0.
* @li min_range: A Tensor of type float32.
* Specifies the minimum scalar value possibly produced for the input. Shape suport 1D ~ 8D.
* The format support ND or NC1HWC0. Has the same format as "x".
* @li max_range: A Tensor of type float32.
* Specifies the maximum scalar value possibly produced for the input. The format support ND or NC1HWC0.
* Shape suport 1D ~ 8D. "max_range" has the same shape as "min_range". Has the same format as "x". \n

* @par Attributes:
* mode: An optional string from: "MIN_COMBINED", "MIN_FIRST", and "SCALED".
* Defaults to "MIN_COMBINED" . \n

* @par Outputs:
* y: A dictionary of type float32. The format support ND or NC1HWC0.
* "y" has the same shape and format as "x". \n

* @attention Constraints:
* @li "min_range" and "max_range" have the same shapes.
* @li "x" and "y" have the same shapes. \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Dequantize.
*/
REG_OP(Dequantize)
    .INPUT(x, TensorType(DT_QINT8, DT_QUINT8, DT_QINT32, DT_QINT16, DT_QUINT16))
    .INPUT(min_range, TensorType{DT_FLOAT})
    .INPUT(max_range, TensorType{DT_FLOAT})
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(mode, String, "MIN_COMBINED")
    .OP_END_FACTORY_REG(Dequantize)

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
* @brief Transfer quant param from float32 to uint64.

* @par Inputs:
* @li scale: A quantization parameter tensor. Must be one of the following types: float32.
             The format support ND. The shape support 1D and 2D.
* @li offset: An optional quantization parameter tensor. Must be one of the following types: float32.
              The format support ND. The shape support 1D and 2D. \n

* @par Attributes:
* @li round_mode: fp32 filing to fp19 mode; 0: truncation and filing; 1: r_int mode;

* @par Outputs:
* @li y: output tensor. Must be one of the following types: uint64. The format support ND.
         The shape support 1D and 2D. \n

* @attention Constraints:
* @li The passed scale, y cannot be a null pointer.

* - Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component or
* Atlas A3 Training Series Product/Atlas A3 Inference Series Product or Atlas Inference Series Product or
* Ascend 950 AI Processor: \n
* This operator supports use with the matmul operator, such as QuantBatchMatmulV3. \n
* - Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component or
* Atlas A3 Training Series Product/Atlas A3 Inference Series Product or Atlas Inference Series Product: \n
* This operator does not supports use with the grouped matmul operator, such as GroupedMatmul. \n
* @li When there is no offset, the y shape is consistent with the scale shape: \n
* - If y is used as matmul input(e.g., QuantBatchMatmulV3), the shape support 1D (t,), with t equal to 1 or n, or 2D(1,
n),
* where n is the same as that of the right matrix in the matmul calculation. \n
* - If y is used as grouped matmul input(e.g., GroupedMatmul), the shape support 1D (g,), or 2D(g, n), (g, 1),
* where n is the same as that of the right matrix in the grouped matmul calculation, g is the same as that of
* the number of group num(i.e., group_list parameter's shape). \n
* @li When there is an offset, use it only as matmul input (e.g., QuantBatchMatmulV3): \n
* The shape of scale, offet and y support 1D (t,), with t equal to 1 or n, or 2D(1, n), where n is the same as
* that of the right matrix in the matmul calculation.
*/
REG_OP(TransQuantParamV2)
    .INPUT(scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_UINT64}))
    .ATTR(round_mode, Int, 0)
    .OP_END_FACTORY_REG(TransQuantParamV2)

/**
* @brief: DynamicRNN calculation.
* @par Inputs:
* ten inputs:
* @li x:A required 3D Tensor. Must be one of the following types: float16, float32.
* The format must be ND.
* @li w:A required 2D Tensor. Must be one of the following types: float16, float32.
* The format must be ND.
* @li b:A required 1D Tensor. Must be one of the following types: float16, float32.
* The format must be ND.
* @li seq_length:A optional 3D Tensor. Must be one of the following types: float16, int32, float32.
* The format must be ND.
* @li init_h:A optional Tensor. Must be one of the following types: float16, float32.
* The format must be ND.
* @li init_c:A optional Tensor. Must be one of the following types: float16, float32.
* The format must be ND.
* @li wci:A optional reserved Tensor. Must be one of the following types: float16, float32. The format must be ND.
* @li wcf:A optional reserved Tensor. Must be one of the following types: float16, float32. The format must be ND.
* @li wco:A optional reserved Tensor. Must be one of the following types: float16, float32. The format must be ND.
* @li mask:A optional reserved Tensor. Must be one of the following types: uint8. The format must be ND. \n

* @par Attributes:
* @li cell_type:An string identifying the cell type in the op.
* Default to "LSTM". Only LSTM is currently supported.
* @li direction:An string identifying the direction in the op.
* Default to "UNIDIRECTIONAL". Only UNIDIRECTIONAL is currently supported.
* @li cell_depth:An integer identifying the cell depth in the op.
* Default to 1. Only 1 is currently supported.
* @li use_peephole:An bool identifying if use peephole in the op.
* Default to false. Only false is currently supported.
* @li keep_prob:An float identifying the keep prob in the op. Default to 1.0.
* @li cell_clip:An float identifying the cell clip in the op. Default to -1.0.
* @li num_proj:An integer identifying the num projection in the op.
* Default to 0. Only 0 is currently supported.
* @li time_major:An bool identifying the time major in the op. Default to true.
* @li activation:An string identifying the type of activation function in the op.
* Default to "tanh". Only "tanh" is currently supported.
* @li forget_bias:An float identifying the forget bias in the op. Default to 0.
* @li gate_order:An string identifying the type of gate order in the op.
* Support "ijfo" and "ifjo". Default to "ijfo".
* @li is_training:An bool identifying is training in the op. Default to true . \n

* @par Outputs:
* eight outputs:
* @li y:A 3D Tensor. Must be one of the following types: float16, float32. The format must be ND.
* @li output_h:A 3D Tensor. Must be one of the following types: float16, float32. The format must be ND.
* @li output_c:A 3D Tensor. Must be one of the following types: float16, float32. The format must be ND.
* @li i:A 3D Tensor. Must be one of the following types: float16, float32. The format must be ND.
* @li j:A 3D Tensor. Must be one of the following types: float16, float32. The format must be ND.
* @li f:A 3D Tensor. Must be one of the following types: float16, float32. The format must be ND.
* @li o:A 3D Tensor. Must be one of the following types: float16, float32. The format must be ND.
* @li tanhc:A 3D Tensor. Must be one of the following types: float16, float32. The format must be ND.
* @par Third-party framework compatibility:
* Compatible with the TF operator LSTM.
*/
REG_OP(DynamicRNN)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(w, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(b, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(seq_length, TensorType({DT_INT32, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(init_h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(init_c, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(wci, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(wcf, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(wco, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(mask, TensorType({DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(output_h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(output_c, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(i, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(j, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(f, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(o, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(tanhc, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(cell_type, String, "LSTM")
    .ATTR(direction, String, "UNIDIRECTIONAL")
    .ATTR(cell_depth, Int, 1)
    .ATTR(use_peephole, Bool, false)
    .ATTR(keep_prob, Float, 1.0)
    .ATTR(cell_clip, Float, -1.0)
    .ATTR(num_proj, Int, 0)
    .ATTR(time_major, Bool, true)
    .ATTR(activation, String, "tanh")
    .ATTR(forget_bias, Float, 0.0)
    .ATTR(gate_order, String, "ijfo")
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(DynamicRNN)

/**
* @brief: DynamicRNNV2 calculation.
* @par Inputs:
* ten inputs:
* @li x:A required Tensor. Must be one of the following types: float16, float32.
* @li weight_input:A required Tensor. Must be one of the following types: float16, float32.
* @li weight_hidden:A required Tensor. Must be one of the following types: float16, float32.
* @li b:A required Tensor. Must be one of the following types: float16, float32. The format must be ND.
* @li seq_length:A optional Tensor. Must be one of the following types: float16, int32.
* @li init_h:A optional Tensor. Must be one of the following types: float16, float32.
* @li init_c:A optional Tensor. Must be one of the following types: float16, float32.
* @li wci:A optional Tensor. Must be one of the following types: float16, float32.
* @li wcf:A optional Tensor. Must be one of the following types: float16, float32.
* @li wco:A optional Tensor. Must be one of the following types: float16, float32.
* @li mask:A optional Tensor. Must be one of the following types: uint8. The format must be ND . \n

* @par Attributes:
* @li cell_type:An string identifying the cell type in the op.
* Default to "LSTM". Only LSTM is currently supported.
* @li direction:An string identifying the direction in the op.
* Default to "UNIDIRECTIONAL". Support "UNIDIRECTIONAL" and "REDIRECTIONAL".
* @li cell_depth:An integer identifying the cell depth in the op.
* Default to 1. Only 1 is currently supported.
* @li use_peephole:An bool identifying if use peephole in the op.
* Default to false. Only false is currently supported.
* @li keep_prob:An float identifying the keep prob in the op.
* Default to 1. Only 1.0 is currently supported.
* @li cell_clip:An float identifying the cell clip in the op.
* Default to -1. Only -1.0 is currently supported.
* @li num_proj:An integer identifying the num projection in the op.
* Default to 0. Only 0 is currently supported.
* @li time_major:An bool identifying the time major in the op. Default to true.
* @li activation:An string identifying the type of activation function in
* the op. Default to "tanh". Only "tanh" is currently supported.
* @li recurrent_activation:An string identifying the type of activation
* function in the op. Default to "sigmoid". Only support "sigmoid" in
* Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component
* and Atlas A3 Training Series Product/Atlas A3 Inference Series Product.
* Support "sigmoid" and "hard_sigmoid"
* in other series produces. In general, set "hard_sigmoid" for TF Keras LSTM.
* @li forget_bias:An float identifying the forget bias in the op. Default to 0.
* @li gate_order:An string identifying the type of gate order in the op.
* Support "ijfo" and "ifco". Default to "ijfo".
* Set "ijfo" for TF operator LSTM, Set "ifco" for TF Keras LSTM.
* @li stateful: An bool identifying the type of stateful in the op.
* Default to fasle. Only false is currently supported.
* @li merge_mode: An string identifying the type of merge_modein the op.
* Default to "concat". Only "concat" is currently supported
* @li is_training:An bool identifying is training in the op. Default to true . \n

* @par Outputs:
* eight outputs:
* @li y:A Tensor. Must be one of the following types: float16, float32.
* @li output_h:A Tensor. Must be one of the following types: float16, float32.
* Return the last output_h.
* @li output_c:A Tensor. Must be one of the following types: float16, float32.
* Return the last output_c.
* @li i:A Tensor. Must be one of the following types: float16, float32.
* @li j:A Tensor. Must be one of the following types: float16, float32.
* @li f:A Tensor. Must be one of the following types: float16, float32.
* @li o:A Tensor. Must be one of the following types: float16, float32.
* @li tanhc:A Tensor. Must be one of the following types: float16, float32.
* @par Third-party framework compatibility:
* Compatible with the TF operator LSTM or TF keras operator LSTM or the Pytorch operator LSTM.
*/

REG_OP(DynamicRNNV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(weight_input, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(weight_hidden, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(b, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(seq_length, TensorType({DT_INT32, DT_FLOAT16}))
    .OPTIONAL_INPUT(init_h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(init_c, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(wci, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(wcf, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(wco, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(mask, TensorType({DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(output_h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(output_c, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(i, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(j, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(f, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(o, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(tanhc, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(cell_type, String, "LSTM")
    .ATTR(direction, String, "UNIDIRECTIONAL")
    .ATTR(cell_depth, Int, 1)
    .ATTR(use_peephole, Bool, false)
    .ATTR(keep_prob, Float, 1.0)
    .ATTR(cell_clip, Float, -1.0)
    .ATTR(num_proj, Int, 0)
    .ATTR(time_major, Bool, true)
    .ATTR(activation, String, "tanh")
    .ATTR(recurrent_activation, String, "sigmoid")
    .ATTR(forget_bias, Float, 0.0)
    .ATTR(gate_order, String, "ijfo")
    .ATTR(stateful, Bool, false)
    .ATTR(merge_mode, String, "concat")
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(DynamicRNNV2)

/**
*@brief Asserts that the given condition is true. \n

*@par Inputs:
*If input_condition evaluates to false, print the list of tensors in data.
*Inputs include:
*@li input_condition: The condition to evaluate.
*@li input_data: The tensors to print out when condition is false.
 It's a dynamic input.  \n

*@par Attributes:
*summarize: An optional int. Defaults to 3. Print this many entries of each tensor. \n

*@par Third-party framework compatibility
*Compatible with tensorflow Assert operator. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(Assert)
  .INPUT(input_condition, TensorType{DT_BOOL})
  .DYNAMIC_INPUT(input_data, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8,
      DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
      DT_UINT64, DT_BOOL, DT_DOUBLE, DT_STRING}))
  .ATTR(summarize, Int, 3)
  .OP_END_FACTORY_REG(Assert)

/**
*@brief Return a tensor with the same shape and contents as input. \n

*@par Inputs:
*x: A tensor. Must be one of the following types: float32、float16、int8、
int16、uint16、uint8、int32、int64、uint32、uint64、bool、double、string、bfloat16. \n

*@par Outputs:
*y: A tensor with the same shape、data type and contents as input. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Identity.
*/
REG_OP(Identity)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE, DT_STRING, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE, DT_STRING, DT_BF16}))
    .OP_END_FACTORY_REG(Identity)

/**
*@brief Returns a list of tensors with the same shapes and contents as the input tensors. \n

*@par Inputs:
*x: A list of input tensors. It's a dynamic input. Must be one of the following types:
float32、float16、int8、int16、uint16、uint8、int32、int64、uint32、uint64、bool、double、string.\n

*@par Outputs:
*y: A list of Tensor objects, with the same length、shape、data type and contents as the input tensor list.
It's a dynamic output. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator IdentityN.
*/
REG_OP(IdentityN)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE, DT_STRING}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE, DT_STRING}))
    .OP_END_FACTORY_REG(IdentityN)

/**
*@brief Returns an integer representing the rank of input tensor. The rank of a tensor is the number of indices required to uniquely select each element of the tensor, that is, the dimension size of the tensor. \n

*@par Inputs:
*x: A Tensor of type float32, float16, int8, int16, uint16, uint8, int32, int64, uint32, uint64, bool, double, string. \n

*@par Outputs:
*y: A tensor. The rank of input tensor. Type is int32. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Rank.
*/
REG_OP(Rank)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE, DT_STRING}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(Rank)

/**
*@brief Returns the shape of a tensor. \n

*@par Inputs:
*x: A tensor. Must be one of the following types: float32、float16、int8、
int16、uint16、uint8、int32、int64、uint32、uint64、bool、double、string、bfloat16. \n

*@par Attributes:
*dtype: An optional int32 or int64. The output data type. Defaults to int32. \n

*@par Outputs:
*y: A tensor. The shape of the input tensor. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Size.
*/
REG_OP(Shape)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE, DT_STRING, DT_BF16}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(dtype, Int, DT_INT32)
    .OP_END_FACTORY_REG(Shape)

/**
*@brief Returns shape of tensors. \n

*@par Inputs:
*x: A list of input tensors. It's a dynamic input. \n

*@par Attributes:
*dtype: An optional int32 or int64. The output data type. Defaults to "int32". \n

*@par Outputs:
*y: A list of tensors with the same length as the input list of tensors.
It's a dynamic output. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ShapeN.
*/
REG_OP(ShapeN)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE, DT_STRING}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(dtype, Int, DT_INT32)
    .OP_END_FACTORY_REG(ShapeN)

}  // namespace ge

#endif // OP_NN_PROTO_H_
