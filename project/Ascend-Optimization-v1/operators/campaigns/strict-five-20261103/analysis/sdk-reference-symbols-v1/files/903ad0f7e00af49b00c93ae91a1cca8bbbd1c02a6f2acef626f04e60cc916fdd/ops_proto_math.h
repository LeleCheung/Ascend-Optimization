#ifndef OP_MATH_PROTO_H_
#define OP_MATH_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge{

REG_OP(Abs)
    .INPUT(x, "T")
    .OUTPUT(y, "T")
    .DATATYPE(T, TensorType({DT_UINT8, DT_BOOL, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT16,
                             DT_INT32, DT_INT64, DT_COMPLEX32, DT_COMPLEX64}))
    .OP_END_FACTORY_REG(Abs)

REG_OP(Acos)
    .INPUT(x, "T")
    .OUTPUT(y, "T")
    .DATATYPE(T, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
                             DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Acos)

REG_OP(Add)
    .INPUT(
        x1, TensorType(
                {DT_BOOL, DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
                 DT_COMPLEX128, DT_COMPLEX64, DT_STRING, DT_COMPLEX32}))
    .INPUT(
        x2, TensorType(
                {DT_BOOL, DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
                 DT_COMPLEX128, DT_COMPLEX64, DT_STRING, DT_COMPLEX32}))
    .OUTPUT(
        y, TensorType(
               {DT_BOOL, DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
                DT_COMPLEX128, DT_COMPLEX64, DT_STRING, DT_COMPLEX32}))
    .OP_END_FACTORY_REG(Add)

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

REG_OP(AddN)
    .DYNAMIC_INPUT(x, TensorType({NumberType(), DT_VARIANT}))
    .OUTPUT(y, TensorType({NumberType(), DT_VARIANT}))
    .REQUIRED_ATTR(N, Int)
    .OP_END_FACTORY_REG(AddN)

REG_OP(Addcdiv)
    .INPUT(input_data, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_DOUBLE, DT_INT64}))
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_DOUBLE, DT_INT64}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_DOUBLE, DT_INT64}))
    .INPUT(value, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT32, DT_DOUBLE, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_DOUBLE, DT_INT64}))
    .OP_END_FACTORY_REG(Addcdiv)

REG_OP(Addcmul)
    .INPUT(input_data, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8, DT_DOUBLE, DT_INT64}))
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8, DT_DOUBLE, DT_INT64}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8, DT_DOUBLE, DT_INT64}))
    .INPUT(value, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8, DT_DOUBLE, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8, DT_DOUBLE, DT_INT64}))
    .OP_END_FACTORY_REG(Addcmul)

REG_OP(Addr)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_UINT8, DT_BOOL}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_UINT8, DT_BOOL}))
    .INPUT(x3, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_UINT8, DT_BOOL}))
    .INPUT(beta, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_UINT8, DT_BOOL}))
    .INPUT(alpha, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_UINT8, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_UINT8, DT_BOOL}))
    .OP_END_FACTORY_REG(Addr)

REG_OP(AdjacentDifference)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT8, DT_INT16, DT_INT32, DT_INT64,
                          DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(y_dtype, Int, DT_INT32)
    .OP_END_FACTORY_REG(AdjacentDifference)

REG_OP(AngleV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_COMPLEX64, DT_BOOL, DT_UINT8,
                          DT_INT8, DT_INT16, DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(AngleV2)

REG_OP(ArgMaxV2)
    .INPUT(x, TensorType::NumberType())
    .INPUT(dimension, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(dtype, Type, DT_INT64)
    .OP_END_FACTORY_REG(ArgMaxV2)

REG_OP(ArgMaxWithValue)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT64, DT_INT32}))
    .OUTPUT(indice, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(values, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT64, DT_INT32}))
    .REQUIRED_ATTR(dimension, Int)
    .ATTR(keep_dims, Bool, false)
    .ATTR(indice_dtype, Type, DT_INT32)
    .OP_END_FACTORY_REG(ArgMaxWithValue)

REG_OP(ArgMin)
    .INPUT(x, TensorType::NumberType())
    .INPUT(dimension, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(dtype, Type, DT_INT64)
    .OP_END_FACTORY_REG(ArgMin)

REG_OP(ArgMinWithValue)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT64, DT_INT32}))
    .OUTPUT(indice, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(values, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT64, DT_INT32}))
    .REQUIRED_ATTR(dimension, Int)
    .ATTR(keep_dims, Bool, false)
    .ATTR(indice_dtype, Type, DT_INT32)
    .OP_END_FACTORY_REG(ArgMinWithValue)

REG_OP(AssignAdd)
    .INPUT(ref, TensorType::BasicType())
    .INPUT(value, TensorType::BasicType())
    .OUTPUT(ref, TensorType::BasicType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(AssignAdd)

REG_OP(AssignSub)
    .INPUT(var, TensorType::NumberType())
    .INPUT(value, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(AssignSub)

REG_OP(Atan)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Atan)

REG_OP(Axpy)
    .INPUT(x1, TensorType({DT_FLOAT, DT_INT32, DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_INT32, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(alpha, Float)
    .OP_END_FACTORY_REG(Axpy)

REG_OP(AxpyV2)
    .INPUT(x1, "T1")
    .INPUT(x2, "T2")
    .INPUT(alpha, "T3")
    .OUTPUT(y, "T4")
    .DATATYPE(T1, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_BF16, DT_INT64, DT_UINT8, DT_INT8, DT_BOOL}))
    .DATATYPE(T2, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_BF16, DT_INT64, DT_UINT8, DT_INT8, DT_BOOL}))
    .DATATYPE(T3, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_BF16, DT_INT64, DT_UINT8, DT_INT8, DT_BOOL}))
    .DATATYPE(T4, Promote({"T1", "T2", "T3"}))
    .OP_END_FACTORY_REG(AxpyV2)

REG_OP(BiasAdd)
    .INPUT(x, TensorType::NumberType())
    .INPUT(bias, TensorType::NumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(BiasAdd)

REG_OP(BiasAddGrad)
    .INPUT(x, TensorType::NumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(BiasAddGrad)

REG_OP(Bincount)
    .INPUT(array, TensorType(DT_INT32))
    .INPUT(size, TensorType(DT_INT32))
    .INPUT(weights, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_DOUBLE}))
    .OUTPUT(bins, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_DOUBLE}))
    .OP_END_FACTORY_REG(Bincount)

REG_OP(BitwiseAnd)
    .INPUT(x1, TensorType::IntegerDataType())
    .INPUT(x2, TensorType::IntegerDataType())
    .OUTPUT(y, TensorType::IntegerDataType())
    .OP_END_FACTORY_REG(BitwiseAnd)

REG_OP(BitwiseOr)
    .INPUT(x1, TensorType::IntegerDataType())
    .INPUT(x2, TensorType::IntegerDataType())
    .OUTPUT(y, TensorType::IntegerDataType())
    .OP_END_FACTORY_REG(BitwiseOr)

REG_OP(BitwiseXor)
    .INPUT(x1, TensorType::IntegerDataType())
    .INPUT(x2, TensorType::IntegerDataType())
    .OUTPUT(y, TensorType::IntegerDataType())
    .OP_END_FACTORY_REG(BitwiseXor)

REG_OP(Cast)
    .INPUT(x, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT32, DT_UINT8,
                          DT_INT64, DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_COMPLEX64,
                          DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32, DT_BF16, DT_UINT1,
                          DT_COMPLEX32, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN,
                          DT_FLOAT4_E1M2, DT_FLOAT4_E2M1}))
    .OUTPUT(y, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT32, DT_UINT8,
                           DT_INT64, DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_COMPLEX64,
                           DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32,
                           DT_BF16, DT_COMPLEX32, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN,
                           DT_FLOAT4_E1M2, DT_FLOAT4_E2M1, DT_INT4}))
    .REQUIRED_ATTR(dst_type, Int)
    .OP_END_FACTORY_REG(Cast)

REG_OP(Cdist)
    .INPUT(x1, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(x2, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .ATTR(p, Float, 2.0)
    .OP_END_FACTORY_REG(Cdist)

REG_OP(Ceil)
    .INPUT(x, TensorType({FloatingDataType, DT_BF16}))
    .OUTPUT(y, TensorType({FloatingDataType, DT_BF16}))
    .OP_END_FACTORY_REG(Ceil)

REG_OP(Cos)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Cos)

REG_OP(Cross)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8,
                           DT_INT16, DT_DOUBLE, DT_INT64, DT_UINT16, DT_UINT32,
                           DT_UINT64, DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8,
                           DT_INT16, DT_DOUBLE, DT_INT64, DT_UINT16, DT_UINT32,
                           DT_UINT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8,
                           DT_INT16, DT_DOUBLE, DT_INT64, DT_UINT16, DT_UINT32,
                           DT_UINT64, DT_COMPLEX64, DT_COMPLEX128}))
    .ATTR(dim, Int, -65530)
    .OP_END_FACTORY_REG(Cross)

REG_OP(Cumsum)
    .INPUT(x, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128, DT_BF16}))
    .INPUT(axis, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128, DT_BF16}))
    .ATTR(exclusive, Bool, false)
    .ATTR(reverse, Bool, false)
    .OP_END_FACTORY_REG(Cumsum)

REG_OP(DiagPart)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT64, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT64, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(DiagPart)

REG_OP(Div)
    .INPUT(
        x1, TensorType(
                {DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT32, DT_DOUBLE, DT_INT64, DT_UINT16, DT_INT16,
                 DT_COMPLEX64, DT_COMPLEX128, DT_BF16, DT_COMPLEX32}))
    .INPUT(
        x2, TensorType(
                {DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT32, DT_DOUBLE, DT_INT64, DT_UINT16, DT_INT16,
                 DT_COMPLEX64, DT_COMPLEX128, DT_BF16, DT_COMPLEX32}))
    .OUTPUT(
        y, TensorType(
               {DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT32, DT_DOUBLE, DT_INT64, DT_UINT16, DT_INT16,
                DT_COMPLEX64, DT_COMPLEX128, DT_BF16, DT_COMPLEX32}))
    .OP_END_FACTORY_REG(Div)

REG_OP(DivNoNan)
    .INPUT(x1, TensorType({DT_FLOAT, DT_UINT8, DT_INT8, DT_INT32, DT_FLOAT16, DT_DOUBLE, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_UINT8, DT_INT8, DT_INT32, DT_FLOAT16, DT_DOUBLE, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_UINT8, DT_INT8, DT_INT32, DT_FLOAT16, DT_DOUBLE, DT_BF16}))
    .OP_END_FACTORY_REG(DivNoNan)

REG_OP(Dot)
    .INPUT(input_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_UINT8, DT_INT8, DT_INT32}))
    .INPUT(input_y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_UINT8, DT_INT8, DT_INT32}))
    .OUTPUT(output, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_UINT8, DT_INT8, DT_INT32}))
    .OP_END_FACTORY_REG(Dot)

REG_OP(Equal)
    .INPUT(
        x1, TensorType(
                {DT_FLOAT, DT_BF16, DT_FLOAT16, DT_INT32, DT_INT8, DT_UINT8, DT_DOUBLE, DT_INT16, DT_INT64,
                 DT_COMPLEX64, DT_COMPLEX128, DT_QUINT8, DT_QINT8, DT_QINT32, DT_STRING, DT_BOOL}))
    .INPUT(
        x2, TensorType(
                {DT_FLOAT, DT_BF16, DT_FLOAT16, DT_INT32, DT_INT8, DT_UINT8, DT_DOUBLE, DT_INT16, DT_INT64,
                 DT_COMPLEX64, DT_COMPLEX128, DT_QUINT8, DT_QINT8, DT_QINT32, DT_STRING, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(Equal)

REG_OP(Exp)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .ATTR(base, Float, -1.0)
    .ATTR(scale, Float, 1.0)
    .ATTR(shift, Float, 0.0)
    .OP_END_FACTORY_REG(Exp)

REG_OP(Expand)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32,DT_INT64, DT_INT8, DT_UINT8, DT_BOOL, DT_BF16}))
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32,DT_INT64, DT_INT8, DT_UINT8, DT_BOOL, DT_BF16}))
    .OP_END_FACTORY_REG(Expand)

REG_OP(Eye)
    .OUTPUT(y, TensorType({TensorType::BasicType(), DT_BOOL}))    /* "Result, has targeted element type" */
    .REQUIRED_ATTR(num_rows, Int)
    .ATTR(num_columns, Int, 0)
    .ATTR(batch_shape, ListInt, {})
    .ATTR(dtype, Int, 0)
    .OP_END_FACTORY_REG(Eye)

REG_OP(Floor)
    .INPUT(x, TensorType({FloatingDataType, DT_BF16}))
    .OUTPUT(y, TensorType({FloatingDataType, DT_BF16}))
    .OP_END_FACTORY_REG(Floor)

REG_OP(FloorDiv)
    .INPUT(
        x1, TensorType(
                {DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8, DT_INT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_BF16}))
    .INPUT(
        x2, TensorType(
                {DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8, DT_INT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_BF16}))
    .OUTPUT(
        y, TensorType(
               {DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8, DT_INT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_BF16}))
    .OP_END_FACTORY_REG(FloorDiv)

REG_OP(FloorMod)
    .INPUT(x1, TensorType({DT_INT32, DT_INT64, DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_BF16}))
    .INPUT(x2, TensorType({DT_INT32, DT_INT64, DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_BF16}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64, DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_BF16}))
    .OP_END_FACTORY_REG(FloorMod)

REG_OP(FusedMulAddN)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT16, DT_BF16}))
    .INPUT(x3, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT16, DT_BF16}))
    .OP_END_FACTORY_REG(FusedMulAddN)

REG_OP(Gcd)
    .INPUT(x1, "T")
    .INPUT(x2, "T")
    .OUTPUT(y, "T")
    .DATATYPE(T, TensorType({DT_UINT8, DT_INT8, DT_INT16, DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(Gcd)

REG_OP(Ger)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(Ger)

REG_OP(Greater)
    .INPUT(x1, TensorType::RealNumberType())
    .INPUT(x2, TensorType::RealNumberType())
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(Greater)

REG_OP(GreaterEqual)
    .INPUT(x1, TensorType::RealNumberType())
    .INPUT(x2, TensorType::RealNumberType())
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(GreaterEqual)

REG_OP(GroupedBiasAddGrad)
    .INPUT(grad_y, "T")
    .OPTIONAL_INPUT(group_idx, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(grad_bias, "T")
    .ATTR(group_idx_type, Int, 0)
    .DATATYPE(T, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OP_END_FACTORY_REG(GroupedBiasAddGrad)

REG_OP(HansDecode)
    .INPUT(mantissa, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(fixed, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(var, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(pdf, TensorType({DT_INT32}))
    .OUTPUT(output, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .ATTR(reshuff, Bool, false)
    .OP_END_FACTORY_REG(HansDecode)

REG_OP(HansEncode)
    .INPUT(input_tensor, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(pdf, TensorType({DT_INT32}))
    .OUTPUT(pdf, TensorType({DT_INT32}))
    .OUTPUT(mantissa, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(fixed, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(var, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .ATTR(statistic, Bool, false)
    .ATTR(reshuff, Bool, false)
    .OP_END_FACTORY_REG(HansEncode)

REG_OP(HistogramV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT64, DT_INT32, DT_INT16, DT_INT8, DT_UINT8}))
    .INPUT(min, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT64, DT_INT32, DT_INT16, DT_INT8, DT_UINT8}))
    .INPUT(max, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT64, DT_INT32, DT_INT16, DT_INT8, DT_UINT8}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .ATTR(bins, Int, 100)
    .OP_END_FACTORY_REG(HistogramV2)

REG_OP(InvGrad)
    .INPUT(x, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16, DT_INT32, DT_INT8}))
    .INPUT(grad, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16, DT_INT32, DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16, DT_INT32, DT_INT8}))
    .OP_END_FACTORY_REG(InvGrad)

REG_OP(Invert)
    .INPUT(x, TensorType::IntegerDataType())
    .OUTPUT(y, TensorType::IntegerDataType())
    .OP_END_FACTORY_REG(Invert)

REG_OP(IsClose)
    .INPUT(x1, TensorType({BasicType(), DT_BOOL, DT_BF16}))
    .INPUT(x2, TensorType({BasicType(), DT_BOOL, DT_BF16}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .ATTR(rtol, Float, 1e-05f)
    .ATTR(atol, Float, 1e-08f)
    .ATTR(equal_nan, Bool, false)
    .OP_END_FACTORY_REG(IsClose)

REG_OP(IsFinite)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BOOL, DT_UINT8, DT_INT8, DT_UINT16,
                          DT_INT16, DT_INT32, DT_UINT32, DT_UINT64, DT_INT64}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(IsFinite)

REG_OP(IsInf)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BF16}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(IsInf)

REG_OP(IsNan)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(IsNan)

REG_OP(IsNegInf)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BFLOAT16}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(IsNegInf)

REG_OP(IsPosInf)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BFLOAT16}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(IsPosInf)

REG_OP(KLDivV2)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(target, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .ATTR(reduction, String, "mean")
    .ATTR(log_target, Bool, false)
    .OP_END_FACTORY_REG(KLDivV2)

REG_OP(Lerp)
    .INPUT(start, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(end, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(weight, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(Lerp)

REG_OP(Less)
    .INPUT(x1, TensorType::RealNumberType())
    .INPUT(x2, TensorType::RealNumberType())
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(Less)

REG_OP(LessEqual)
    .INPUT(x1, TensorType::RealNumberType())
    .INPUT(x2, TensorType::RealNumberType())
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(LessEqual)

REG_OP(LinSpace)
    .INPUT(start, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT8, DT_UINT8, DT_INT32, DT_INT16, DT_FLOAT16, DT_BF16}))
    .INPUT(stop, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT8, DT_UINT8, DT_INT32, DT_INT16, DT_FLOAT16, DT_BF16}))
    .INPUT(num, TensorType::IndexNumberType())
    .OUTPUT(output, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT8, DT_UINT8, DT_INT32, DT_INT16, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(LinSpace)

REG_OP(Log)
    .INPUT(
        x, TensorType(
               {DT_UINT8, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT, DT_DOUBLE, DT_FLOAT16, DT_BF16, DT_BOOL,
                DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType::UnaryDataType())
    .ATTR(base, Float, -1.0)
    .ATTR(scale, Float, 1.0)
    .ATTR(shift, Float, 0.0)
    .OP_END_FACTORY_REG(Log)

REG_OP(Log1p)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Log1p)

REG_OP(LogicalAnd)
    .INPUT(x1, TensorType({DT_BOOL}))
    .INPUT(x2, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(LogicalAnd)

REG_OP(LogicalNot)
    .INPUT(x, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(LogicalNot)

REG_OP(LogicalOr)
    .INPUT(x1, TensorType({DT_BOOL}))
    .INPUT(x2, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(LogicalOr)

REG_OP(MaskedScale)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .INPUT(mask, TensorType({DT_INT8, DT_FLOAT16, DT_FLOAT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .REQUIRED_ATTR(value, Float)
    .OP_END_FACTORY_REG(MaskedScale)

REG_OP(Maximum)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_BF16, DT_INT8, DT_UINT8}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_BF16, DT_INT8, DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_BF16, DT_INT8, DT_UINT8}))
    .OP_END_FACTORY_REG(Maximum)

REG_OP(Minimum)
    .INPUT(x1, TensorType({DT_BF16, DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT32, DT_INT64, DT_INT8, DT_UINT8}))
    .INPUT(x2, TensorType({DT_BF16, DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT32, DT_INT64, DT_INT8, DT_UINT8}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT32, DT_INT64, DT_INT8, DT_UINT8}))
    .OP_END_FACTORY_REG(Minimum)

REG_OP(Mod)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8,
                           DT_INT64, DT_DOUBLE, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8,
                           DT_INT64, DT_DOUBLE, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8,
                           DT_INT64, DT_DOUBLE, DT_BF16}))
    .OP_END_FACTORY_REG(Mod)

REG_OP(Mul)
    .INPUT(x1, "T1")
    .INPUT(x2, "T2")
    .OUTPUT(y, "T3")
    .DATATYPE(T1, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8,
                              DT_UINT16, DT_INT16, DT_INT32, DT_INT64, DT_BF16,
                              DT_COMPLEX64, DT_COMPLEX128, DT_COMPLEX32}))
    .DATATYPE(T2, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8,
                              DT_UINT16, DT_INT16, DT_INT32, DT_INT64, DT_BF16,
                              DT_COMPLEX64, DT_COMPLEX128, DT_COMPLEX32}))
    .DATATYPE(T3, Promote({"T1", "T2"}))
    .OP_END_FACTORY_REG(Mul)

REG_OP(Muls)
     .INPUT(x, TensorType({DT_FLOAT, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_COMPLEX32, DT_COMPLEX64}))
     .OUTPUT(y, TensorType({DT_FLOAT, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_COMPLEX32, DT_COMPLEX64}))
     .REQUIRED_ATTR(value, Float)
     .OP_END_FACTORY_REG(Muls)

REG_OP(Neg)
    .INPUT(
        x, TensorType(
               {DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(
        y, TensorType(
               {DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Neg)

REG_OP(NonFiniteCheck)
    .DYNAMIC_INPUT(tensor_list, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(found_flag, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(NonFiniteCheck)

REG_OP(NotEqual)
    .INPUT(x1, TensorType::RealNumberType())
    .INPUT(x2, TensorType::RealNumberType())
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(NotEqual)

REG_OP(OneHot)
    .INPUT(x, TensorType({DT_UINT8, DT_INT32, DT_INT64}))
    .INPUT(depth, TensorType({DT_INT32, DT_INT64}))
    .INPUT(on_value, TensorType::BasicType())
    .INPUT(off_value, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .ATTR(axis, Int, -1)
    .OP_END_FACTORY_REG(OneHot)

REG_OP(OnesLike)
    .INPUT(
        x, TensorType(
               {DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_UINT8, DT_INT16, DI_UINT16, DT_INT32, DT_INT64,
                DT_COMPLEX128, DT_BOOL, DT_BF16}))
    .OUTPUT(
        y, TensorType(
               {DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_UINT8, DT_INT16, DI_UINT16, DT_INT32, DT_INT64,
                DT_COMPLEX128, DT_BOOL, DT_BF16}))
    .OP_END_FACTORY_REG(OnesLike)

REG_OP(Pow)
    .INPUT(x1, "T1")
    .INPUT(x2, "T2")
    .OUTPUT(y, "T3")
    .DATATYPE(T1, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64, DT_INT8, DT_INT16,
                              DT_UINT8, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .DATATYPE(T2, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64, DT_INT8, DT_INT16,
                              DT_UINT8, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .DATATYPE(T3, Promote({"T1", "T2"}))
    .OP_END_FACTORY_REG(Pow)

REG_OP(Range)
    .INPUT(start, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_DOUBLE, DT_INT64, DT_BF16}))
    .INPUT(limit, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_DOUBLE, DT_INT64, DT_BF16}))
    .INPUT(delta, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_DOUBLE, DT_INT64, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_DOUBLE, DT_INT64, DT_BF16}))
    .ATTR(is_closed, Bool, false)
    .OP_END_FACTORY_REG(Range)

REG_OP(RealDiv)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_DOUBLE, DT_UINT8, DT_INT8,
                           DT_UINT16, DT_INT16, DT_INT32, DT_INT64, DT_BOOL,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_DOUBLE, DT_UINT8, DT_INT8,
                           DT_UINT16, DT_INT16, DT_INT32, DT_INT64, DT_BOOL,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_DOUBLE, DT_UINT8, DT_INT8,
                           DT_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(RealDiv)

REG_OP(Reciprocal)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE, DT_FLOAT16,
                          DT_COMPLEX64, DT_COMPLEX128, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_FLOAT16,
                           DT_COMPLEX64, DT_COMPLEX128, DT_BF16}))
    .OP_END_FACTORY_REG(Reciprocal)

REG_OP(ReduceAll)
    .INPUT(x, TensorType({DT_BOOL}))
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_BOOL}))
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceAll)

REG_OP(ReduceAny)
    .INPUT(x, TensorType({DT_BOOL}))
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_BOOL}))
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceAny)

REG_OP(ReduceLogSumExp)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(axes, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceLogSumExp)

REG_OP(ReduceMax)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .ATTR(noop_with_empty_axes, Bool, true)
    .OP_END_FACTORY_REG(ReduceMax)

REG_OP(ReduceMean)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .ATTR(noop_with_empty_axes, Bool, true)
    .OP_END_FACTORY_REG(ReduceMean)

REG_OP(ReduceMin)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .ATTR(noop_with_empty_axes, Bool, true)
    .OP_END_FACTORY_REG(ReduceMin)

REG_OP(ReduceProd)
    .INPUT(x,TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y,TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .ATTR(noop_with_empty_axes, Bool, true)
    .OP_END_FACTORY_REG(ReduceProd)

REG_OP(ReduceStdV2)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(std, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(dim, ListInt, {})
    .ATTR(correction, Int, 1)
    .ATTR(keepdim, Bool, false)
    .ATTR(is_mean_out, Bool, true)
    .OP_END_FACTORY_REG(ReduceStdV2)

REG_OP(ReduceSum)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .ATTR(noop_with_empty_axes, Bool, true)
    .OP_END_FACTORY_REG(ReduceSum)

REG_OP(ReduceVar)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(var, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(dim, ListInt, {})
    .ATTR(correction, Int, 1)
    .ATTR(keepdim, Bool, false)
    .ATTR(is_mean_out, Bool, true)
    .OP_END_FACTORY_REG(ReduceVar)

REG_OP(RightShift)
    .INPUT(x, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, \
           DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .INPUT(y, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, \
           DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .OUTPUT(z, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, \
            DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .OP_END_FACTORY_REG(RightShift)

REG_OP(Round)
    .INPUT(x, TensorType(DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT32, DT_INT64,
                         DT_DOUBLE))
    .OUTPUT(y, TensorType(DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT32, DT_INT64,
                          DT_DOUBLE))
    .ATTR(decimals, Int, 0)
    .OP_END_FACTORY_REG(Round)

REG_OP(Rsqrt)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Rsqrt)

REG_OP(RsqrtGrad)
    .INPUT(y, TensorType({UnaryDataType, DT_INT32, DT_INT8, DT_BF16}))
    .INPUT(dy, TensorType({UnaryDataType, DT_INT32, DT_INT8, DT_BF16}))
    .OUTPUT(z, TensorType({UnaryDataType, DT_INT32, DT_INT8, DT_BF16}))
    .OP_END_FACTORY_REG(RsqrtGrad)

REG_OP(SearchSorted)
    .INPUT(sorted_sequence, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT16, DT_INT8,
                                DT_UINT8, DT_INT32, DT_INT64}))
    .INPUT(values, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT16, DT_INT8,
                                DT_UINT8, DT_INT32, DT_INT64}))
    .OPTIONAL_INPUT(sorter, TensorType({DT_INT64}))
    .OUTPUT(out, TensorType(DT_INT32, DT_INT64))
    .ATTR(dtype, Type, DT_INT64)
    .ATTR(right, Bool, false)
    .OP_END_FACTORY_REG(SearchSorted)

REG_OP(Select)
    .INPUT(condition, TensorType({DT_BOOL}))
    .INPUT(x1, TensorType({BasicType(), DT_BOOL, DT_STRING}))
    .INPUT(x2, TensorType({BasicType(), DT_BOOL, DT_STRING}))
    .OUTPUT(y, TensorType({BasicType(), DT_BOOL, DT_STRING}))
    .OP_END_FACTORY_REG(Select)

REG_OP(SelectV2)
    .INPUT(condition, TensorType({DT_BOOL}))
    .INPUT(then,TensorType({DT_COMPLEX128,DT_COMPLEX64,DT_DOUBLE,DT_FLOAT,DT_FLOAT16,DT_INT16,DT_INT32,DT_INT64,DT_INT8,DT_UINT16,DT_UINT32,DT_UINT64,DT_UINT8,DT_BOOL,DT_BF16}))
    .INPUT(else,TensorType({DT_COMPLEX128,DT_COMPLEX64,DT_DOUBLE,DT_FLOAT,DT_FLOAT16,DT_INT16,DT_INT32,DT_INT64,DT_INT8,DT_UINT16,DT_UINT32,DT_UINT64,DT_UINT8,DT_BOOL,DT_BF16}))
    .OUTPUT(result,TensorType({DT_COMPLEX128,DT_COMPLEX64,DT_DOUBLE,DT_FLOAT,DT_FLOAT16,DT_INT16,DT_INT32,DT_INT64,DT_INT8,DT_UINT16,DT_UINT32,DT_UINT64,DT_UINT8,DT_BOOL,DT_BF16}))
    .OP_END_FACTORY_REG(SelectV2)

REG_OP(Sign)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Sign)

REG_OP(Sin)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Sin)

REG_OP(Sinh)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Sinh)

REG_OP(Sort)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT16, DT_INT8,
                          DT_UINT8, DT_INT32, DT_INT64, DT_BF16, 
                          DT_UINT32, DT_UINT16, DT_UINT64}))
    .OUTPUT(y1, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT16, DT_INT8,
                            DT_UINT8, DT_INT32, DT_INT64, DT_BF16,
                            DT_UINT32, DT_UINT16, DT_UINT64}))
    .OUTPUT(y2, TensorType({DT_INT32, DT_INT64}))
    .ATTR(axis, Int, -1)
    .ATTR(descending, Bool, false)
    .ATTR(stable, Bool, false)
    .ATTR(y2_dtype, Type, DT_INT32)
    .OP_END_FACTORY_REG(Sort)

REG_OP(SortWithIndex)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT8, DT_INT16, DT_INT32, DT_INT64,
                          DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .INPUT(index, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT8, DT_INT16, DT_INT32, DT_INT64,
                          DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .OUTPUT(sorted_index, TensorType({DT_INT32, DT_INT64}))
    .ATTR(axis, Int, -1)
    .ATTR(descending, Bool, false)
    .ATTR(stable, Bool, false)
    .OP_END_FACTORY_REG(SortWithIndex)

REG_OP(Sqrt)
    .INPUT(x, TensorType{(DT_BF16, DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128)})
    .OUTPUT(y, TensorType{(DT_BF16, DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128)})
    .OP_END_FACTORY_REG(Sqrt)

REG_OP(SqrtGrad)
    .INPUT(y, TensorType(UnaryDataType))
    .INPUT(dy, TensorType(UnaryDataType))
    .OUTPUT(z, TensorType(UnaryDataType))
    .OP_END_FACTORY_REG(SqrtGrad)

REG_OP(Square)
    .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Square)

REG_OP(SquareSumV1)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(axis, ListInt)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(SquareSumV1)

REG_OP(SquaredDifference)
    .INPUT(x1, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x2, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(SquaredDifference)

REG_OP(STFT)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .OPTIONAL_INPUT(window, TensorType({DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .ATTR(hop_length, Int, 0)
    .ATTR(win_length, Int, 0)
    .ATTR(normalized, Bool, false)
    .ATTR(onesided, Bool, true)
    .ATTR(return_complex, Bool, true)
    .REQUIRED_ATTR(n_fft, Int)
    .OP_END_FACTORY_REG(STFT)

REG_OP(Sub)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_UINT8, DT_INT8,
                           DT_UINT16, DT_INT16, DT_INT32, DT_INT64, DT_BOOL,
                           DT_COMPLEX64, DT_COMPLEX128, DT_BF16, DT_COMPLEX32}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_UINT8, DT_INT8,
                           DT_UINT16, DT_INT16, DT_INT32, DT_INT64, DT_BOOL,
                           DT_COMPLEX64, DT_COMPLEX128, DT_BF16, DT_COMPLEX32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_UINT8, DT_INT8,
                           DT_UINT16, DT_INT16, DT_INT32, DT_INT64, DT_BOOL,
                           DT_COMPLEX64, DT_COMPLEX128, DT_BF16, DT_COMPLEX32}))
    .OP_END_FACTORY_REG(Sub)

REG_OP(Tanh)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Tanh)

REG_OP(TanhGrad)
    .INPUT(y, TensorType::UnaryDataType())
    .INPUT(dy, TensorType::UnaryDataType())
    .ATTR(complex_conj, Bool, false)
    .OUTPUT(z, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(TanhGrad)

REG_OP(TensorEqual)
    .INPUT(input_x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT64, DT_INT32, DT_INT8, DT_UINT8, DT_BOOL, DT_BF16, DT_IN16, DT_UINT16, DT_UINT32, DT_UINT64}))
    .INPUT(input_y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT64, DT_INT32, DT_INT8, DT_UINT8, DT_BOOL, DT_BF16, DT_IN16, DT_UINT16, DT_UINT32, DT_UINT64}))
    .OUTPUT(output_z, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(TensorEqual)

REG_OP(Tile)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128,
                          DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64,
                          DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32, DT_BF16, DT_BOOL, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .INPUT(multiples, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128,
                           DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64,
                           DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32, DT_BF16, DT_BOOL, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .OP_END_FACTORY_REG(Tile)

REG_OP(TopKV2)
    .INPUT(x, TensorType::RealNumberType())
    .INPUT(k, TensorType::IndexNumberType())
    .OUTPUT(values, TensorType::RealNumberType())
    .OUTPUT(indices, TensorType::IndexNumberType())
    .ATTR(sorted, Bool, true)
    .ATTR(dim, Int, -1)
    .ATTR(largest, Bool, true)
    .ATTR(indices_dtype, Int, DT_INT32)
    .OP_END_FACTORY_REG(TopKV2)

REG_OP(TransformBiasRescaleQkv)
    .INPUT(qkv, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(qkv_bias, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(q, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(k, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(v, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(num_heads, Int)
    .OP_END_FACTORY_REG(TransformBiasRescaleQkv)

REG_OP(Trunc)
    .INPUT(input_x, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8}))
    .OUTPUT(output_y, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8}))
    .OP_END_FACTORY_REG(Trunc)

REG_OP(ZerosLike)
    .INPUT(x, TensorType({BasicType(), DT_VARIANT, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8, DT_FLOAT4_E1M2, DT_FLOAT4_E2M1}))
    .OUTPUT(y, TensorType({BasicType(), DT_VARIANT, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8, DT_FLOAT4_E1M2, DT_FLOAT4_E2M1}))
    .OP_END_FACTORY_REG(ZerosLike)

REG_OP(AsStrided)
    .INPUT(x, TensorType({BasicType(), DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_BOOL}))
    .INPUT(size, TensorType::IndexNumberType())
    .INPUT(stride, TensorType::IndexNumberType())
    .INPUT(storage_offset, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({BasicType(), DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_BOOL}))
    .OP_END_FACTORY_REG(AsStrided)

REG_OP(Assign)
    .INPUT(ref, TensorType({BasicType(), DT_BOOL, DT_STRING}))
    .INPUT(value, TensorType({BasicType(), DT_BOOL, DT_STRING}))
    .OUTPUT(ref, TensorType({BasicType(), DT_BOOL, DT_STRING}))
    .ATTR(validate_shape, Bool, true)
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(Assign)

REG_OP(BroadcastTo)
    .INPUT(x, TensorType({BasicType(), DT_BOOL, DT_STRING, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({BasicType(), DT_BOOL, DT_STRING, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .OP_END_FACTORY_REG(BroadcastTo)

REG_OP(ChunkCat)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(dim, Int)
    .REQUIRED_ATTR(num_chunks, Int)
    .OP_END_FACTORY_REG(ChunkCat)

REG_OP(CircularPad)
    .INPUT(x, TensorType({DT_INT8, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT32}))
    .INPUT(paddings, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT8, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT32}))
    .OP_END_FACTORY_REG(CircularPad)

REG_OP(ClipByValue)
    .INPUT(
        x, TensorType(
               {DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8,
                DT_QINT32, DT_QINT8, DT_QUINT8, DT_UINT16, DT_UINT8, DT_BF16, DT_COMPLEX32}))
    .INPUT(
        clip_value_min, TensorType(
                            {DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64,
                             DT_INT8, DT_QINT32, DT_QINT8, DT_QUINT8, DT_UINT16, DT_UINT8, DT_BF16, DT_COMPLEX32}))
    .INPUT(
        clip_value_max, TensorType(
                            {DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64,
                             DT_INT8, DT_QINT32, DT_QINT8, DT_QUINT8, DT_UINT16, DT_UINT8, DT_BF16, DT_COMPLEX32}))
    .OUTPUT(
        y, TensorType(
               {DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8,
                DT_QINT32, DT_QINT8, DT_QUINT8, DT_UINT16, DT_UINT8, DT_BF16, DT_COMPLEX32}))
    .OP_END_FACTORY_REG(ClipByValue)

REG_OP(ClipByValueV2)
    .INPUT(x, TensorType::NumberType())
    .INPUT(clip_value_min, TensorType::NumberType())
    .INPUT(clip_value_max, TensorType::NumberType())
    .OUTPUT(y, TensorType::NumberType())
    .OP_END_FACTORY_REG(ClipByValueV2)

REG_OP(Concat)
    .INPUT(concat_dim, TensorType::IndexNumberType())
    .DYNAMIC_INPUT(x, TensorType({BasicType(), DT_BOOL, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_HIFLOAT8, DT_FLOAT8_E8M0}))
    .OUTPUT(y, TensorType({BasicType(), DT_BOOL, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_HIFLOAT8, DT_FLOAT8_E8M0}))
    .ATTR(N, Int, 1)
    .OP_END_FACTORY_REG(Concat)

REG_OP(ConcatD)
    .DYNAMIC_INPUT(x, TensorType({DT_BF16, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_BOOL, DT_INT32, DT_INT64,
                                  DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_DOUBLE, DT_COMPLEX64, DT_FLOAT8_E4M3FN,
                                  DT_FLOAT8_E5M2, DT_HIFLOAT8, DT_FLOAT8_E8M0}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_BOOL, DT_INT32, DT_INT64,
                           DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_DOUBLE, DT_COMPLEX64, DT_FLOAT8_E4M3FN,
                           DT_FLOAT8_E5M2, DT_HIFLOAT8, DT_FLOAT8_E8M0}))
    .REQUIRED_ATTR(concat_dim, Int)
    .ATTR(N, Int, 1)
    .OP_END_FACTORY_REG(ConcatD)

REG_OP(ConcatV2)
    .DYNAMIC_INPUT(x, TensorType({BasicType(), DT_BOOL, DT_STRING, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_HIFLOAT8, DT_FLOAT8_E8M0}))
    .INPUT(concat_dim, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({BasicType(), DT_BOOL, DT_STRING, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_HIFLOAT8, DT_FLOAT8_E8M0}))
    .ATTR(N, Int, 1)
    .OP_END_FACTORY_REG(ConcatV2)

REG_OP(DepthToSpace)
  .INPUT(x, TensorType::BasicType())
  .OUTPUT(y, TensorType::BasicType())
  .REQUIRED_ATTR(block_size, Int)
  .ATTR(mode, String, "DCR")
  .ATTR(data_format, String, "NHWC")
  .OP_END_FACTORY_REG(DepthToSpace)

REG_OP(DiagV2)
    .INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64,
                          DT_FLOAT, DT_FLOAT16, DT_BF16, DT_DOUBLE, DT_BOOL,
                          DT_COMPLEX32, DT_COMPLEX128, DT_COMPLEX64}))
    .ATTR(diagonal, Int, 0)
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64,
                           DT_FLOAT, DT_FLOAT16, DT_BF16, DT_DOUBLE, DT_BOOL,
                           DT_COMPLEX32, DT_COMPLEX128, DT_COMPLEX64}))
    .OP_END_FACTORY_REG(DiagV2)

REG_OP(DynamicPartition)
    .INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_BOOL, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_DOUBLE,\
        DT_COMPLEX64, DT_COMPLEX128, DT_RESOURCE, DT_STRING, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .INPUT(partitions, TensorType({DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_BOOL, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_DOUBLE,\
        DT_COMPLEX64, DT_COMPLEX128, DT_RESOURCE, DT_STRING, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .ATTR(num_partitions, Int, 1)
    .OP_END_FACTORY_REG(DynamicPartition)

REG_OP(FeedsRepeat)
    .INPUT(feeds, TensorType::BasicType())
    .INPUT(feeds_repeat_times, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(output_feeds_size, Int)
    .OP_END_FACTORY_REG(FeedsRepeat)

REG_OP(Fill)
    .INPUT(dims, TensorType::IndexNumberType())
    .INPUT(value, "T")
    .OUTPUT(y, "T")
    .DATATYPE(T, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16,
                              DT_INT8, DT_COMPLEX64, DT_INT64, DT_BOOL, DT_QINT8,
                              DT_QUINT8, DT_QINT32, DT_QINT16, DT_QUINT16, DT_UINT16,
                              DT_COMPLEX128, DT_FLOAT16, DT_BF16, DT_UINT32, DT_UINT64, DT_STRING}))
    .OP_END_FACTORY_REG(Fill)

REG_OP(FillDiagonalV2)
    .INPUT(x, TensorType({BasicType(), DT_BOOL, DT_BF16}))
    .INPUT(fill_value, TensorType({BasicType(), DT_BOOL, DT_BF16}))
    .OUTPUT(x, TensorType({BasicType(), DT_BOOL, DT_BF16}))
    .ATTR(wrap, Bool, false)
    .OP_END_FACTORY_REG(FillDiagonalV2)

REG_OP(Fills)
     .INPUT(x, TensorType({BasicType(), DT_BOOL}))
     .OUTPUT(y, TensorType({BasicType(), DT_BOOL}))
     .REQUIRED_ATTR(value, Float)
     .OP_END_FACTORY_REG(Fills)

REG_OP(Im2col)
    .INPUT(x, TensorType({RealNumberType(), DT_BOOL, DT_COMPLEX32, DT_COMPLEX64}))
    .OUTPUT(y, TensorType({RealNumberType(), DT_BOOL, DT_COMPLEX32, DT_COMPLEX64}))
    .REQUIRED_ATTR(ksizes, ListInt)
    .ATTR(strides, ListInt, {1})
    .ATTR(dilations, ListInt, {1})
    .ATTR(padding_mode, String, "CALCULATED")
    .ATTR(pads, ListInt, {0})
    .OP_END_FACTORY_REG(Im2col)

REG_OP(InvertPermutation)
    .INPUT(x, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(InvertPermutation)

REG_OP(MaskedFill)
    .INPUT(x, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16, DT_INT8, DT_INT32, DT_INT64, DT_BOOL}))
    .INPUT(mask, TensorType({DT_BOOL}))
    .INPUT(value, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16, DT_INT8, DT_INT32, DT_INT64, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16, DT_INT8, DT_INT32, DT_INT64, DT_BOOL}))
    .OP_END_FACTORY_REG(MaskedFill)

REG_OP(MatrixDiag)
    .INPUT(x, TensorType({BasicType(), DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_BOOL}))
    .OUTPUT(y, TensorType({BasicType(), DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_BOOL}))
    .OP_END_FACTORY_REG(MatrixDiag)

REG_OP(MatrixSetDiag)
    .INPUT(x, TensorType::BasicType())
    .INPUT(diagonal, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(MatrixSetDiag)

REG_OP(MemSet)
    .REQUIRED_ATTR(sizes, ListInt)
    .ATTR(dtypes, ListType, {})
    .ATTR(values_int, ListInt, {})
    .ATTR(values_float, ListFloat, {})
    .OP_END_FACTORY_REG(MemSet)

REG_OP(MemSetV2)
    .DYNAMIC_INPUT(
        x, TensorType(
               {DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
                DT_UINT64, DT_BOOL}))
    .DYNAMIC_OUTPUT(
        x, TensorType(
               {DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
                DT_UINT64, DT_BOOL}))
    .ATTR(values_int, ListInt, {})
    .ATTR(values_float, ListFloat, {})
    .OP_END_FACTORY_REG(MemSetV2)

REG_OP(MirrorPad)
    .INPUT(
        x, TensorType(
               {DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_BF16, DT_FLOAT16,
                DT_FLOAT, DT_DOUBLE, DT_BOOL, DT_COMPLEX64, DT_COMPLEX128, DT_HIFLOAT8, DT_FLOAT8_E5M2,
                DT_FLOAT8_E4M3FN, DT_FLOAT8_E8M0}))
    .INPUT(paddings, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(
        y, TensorType(
               {DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_BF16, DT_FLOAT16,
                DT_FLOAT, DT_DOUBLE, DT_BOOL, DT_COMPLEX64, DT_COMPLEX128, DT_HIFLOAT8, DT_FLOAT8_E5M2,
                DT_FLOAT8_E4M3FN, DT_FLOAT8_E8M0}))
    .REQUIRED_ATTR(mode, String)
    .OP_END_FACTORY_REG(MirrorPad)

REG_OP(Pack)
    .DYNAMIC_INPUT(x, TensorType({BasicType(), DT_BOOL, DT_STRING}))
    .OUTPUT(y, TensorType({BasicType(), DT_BOOL, DT_STRING}))
    .ATTR(axis, Int, 0)
    .ATTR(N, Int, 1)
    .OP_END_FACTORY_REG(Pack)

REG_OP(Pad)
    .INPUT(x, TensorType({TensorType::BasicType(), DT_BOOL, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E8M0}))
    .INPUT(paddings, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({TensorType::BasicType(), DT_BOOL, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E8M0}))
    .OP_END_FACTORY_REG(Pad)

REG_OP(PadV3)
    .INPUT(x, TensorType({TensorType::BasicType(), DT_BOOL, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E8M0}))
    .INPUT(paddings, TensorType::IndexNumberType())
    .OPTIONAL_INPUT(constant_values, TensorType({TensorType::BasicType(), DT_BOOL, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E8M0}))
    .OUTPUT(y, TensorType({TensorType::BasicType(), DT_BOOL, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E8M0}))
    .ATTR(mode, String, "constant")
    .ATTR(paddings_contiguous, Bool, true)
    .OP_END_FACTORY_REG(PadV3)

REG_OP(Roll)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT32, DT_INT8, DT_UINT8, DT_BF16, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT32, DT_INT8, DT_UINT8, DT_BF16, DT_INT64}))
    .REQUIRED_ATTR(shifts, ListInt)
    .ATTR(dims, ListInt, {})
    .OP_END_FACTORY_REG(Roll)

REG_OP(Slice)
    .INPUT(x, TensorType({BasicType(), DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E8M0, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2}))
    .INPUT(offsets, TensorType::IndexNumberType())
    .INPUT(size, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({BasicType(), DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E8M0, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2}))
    .OP_END_FACTORY_REG(Slice)

REG_OP(SpaceToDepth)
  .INPUT(x, TensorType::BasicType())
  .OUTPUT(y, TensorType::BasicType())
  .REQUIRED_ATTR(block_size, Int)
  .ATTR(data_format, String, "NHWC")
  .OP_END_FACTORY_REG(SpaceToDepth)

REG_OP(Split)
    .INPUT(split_dim, TensorType({DT_INT32}))
    .INPUT(x, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT,  DT_FLOAT16, DT_INT16,
                          DT_INT32,      DT_INT64,     DT_INT8,   DT_QINT16, DT_QINT32,  DT_QINT8,
                          DT_QUINT16,    DT_QUINT8,    DT_UINT16, DT_UINT32, DT_UINT64,  DT_UINT8,
                          DT_BF16,       DT_BOOL}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT,  DT_FLOAT16, DT_INT16,
                                   DT_INT32,      DT_INT64,     DT_INT8,   DT_QINT16, DT_QINT32,  DT_QINT8,
                                   DT_QUINT16,    DT_QUINT8,    DT_UINT16, DT_UINT32, DT_UINT64,  DT_UINT8,
                                   DT_BF16,       DT_BOOL}))
    .REQUIRED_ATTR(num_split, Int)
    .OP_END_FACTORY_REG(Split)

REG_OP(SplitV)
    .INPUT(x, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16,
                          DT_INT32, DT_INT64, DT_INT8, DT_QINT16, DT_QINT32, DT_QINT8,
                          DT_QUINT16, DT_QUINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8,
                          DT_BF16, DT_BOOL, DT_STRING}))
    .INPUT(size_splits, TensorType::IndexNumberType())
    .INPUT(split_dim, TensorType({DT_INT32, DT_INT64}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16,
                                   DT_INT32, DT_INT64, DT_INT8, DT_QINT16, DT_QINT32, DT_QINT8,
                                   DT_QUINT16, DT_QUINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8,
                                   DT_BF16, DT_BOOL, DT_STRING}))
    .REQUIRED_ATTR(num_split, Int)
    .OP_END_FACTORY_REG(SplitV)

REG_OP(Squeeze)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(axis, ListInt, {})
    .OP_END_FACTORY_REG(Squeeze)

REG_OP(SqueezeV2)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(axis, ListInt, {})
    .OP_END_FACTORY_REG(SqueezeV2)

REG_OP(SqueezeV3)
    .INPUT(x, TensorType::ALL())
    .OPTIONAL_INPUT(axes, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType::ALL())
    .OP_END_FACTORY_REG(SqueezeV3)

REG_OP(StackBallQuery)
    .INPUT(xyz, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(center_xyz, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(xyz_batch_cnt, TensorType({DT_INT32, DT_INT64}))
    .INPUT(center_xyz_batch_cnt, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(idx, TensorType({DT_INT32}))
    .REQUIRED_ATTR(max_radius, Float)
    .REQUIRED_ATTR(sample_num, Int)
    .OP_END_FACTORY_REG(StackBallQuery)

REG_OP(StridedSlice)
    .INPUT(x, TensorType({BasicType(), DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .INPUT(begin, TensorType::IndexNumberType())
    .INPUT(end, TensorType::IndexNumberType())
    .INPUT(strides, TensorType::IndexNumberType())
    .ATTR(begin_mask, Int, 0)
    .ATTR(end_mask, Int, 0)
    .ATTR(ellipsis_mask, Int, 0)
    .ATTR(new_axis_mask, Int, 0)
    .ATTR(shrink_axis_mask, Int, 0)
    .OUTPUT(y, TensorType({BasicType(), DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .OP_END_FACTORY_REG(StridedSlice)

REG_OP(StridedSliceAssignV2)
    .INPUT(var, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT32, DT_INT64, DT_DOUBLE, DT_INT8}))
    .INPUT(input_value, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT32, DT_INT64, DT_DOUBLE, DT_INT8}))
    .INPUT(begin, TensorType::IndexNumberType())
    .INPUT(end, TensorType::IndexNumberType())
    .INPUT(strides, TensorType::IndexNumberType())
    .OPTIONAL_INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(var, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT32, DT_INT64, DT_DOUBLE, DT_INT8}))
    .OP_END_FACTORY_REG(StridedSliceAssignV2)

REG_OP(StridedSliceGrad)
    .INPUT(shape, TensorType::IndexNumberType())
    .INPUT(begin, TensorType::IndexNumberType())
    .INPUT(end, TensorType::IndexNumberType())
    .INPUT(strides, TensorType::IndexNumberType())
    .INPUT(dy, TensorType::BasicType())
    .OUTPUT(output, TensorType::BasicType())
    .ATTR(begin_mask, Int, 0)
    .ATTR(end_mask, Int, 0)
    .ATTR(ellipsis_mask, Int, 0)
    .ATTR(new_axis_mask, Int, 0)
    .ATTR(shrink_axis_mask, Int, 0)
    .OP_END_FACTORY_REG(StridedSliceGrad)

REG_OP(StridedSliceV2)
    .INPUT(x, TensorType({BasicType(), DT_BOOL, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E8M0}))
    .INPUT(begin, TensorType::IndexNumberType())
    .INPUT(end, TensorType::IndexNumberType())
    .OPTIONAL_INPUT(axes, TensorType::IndexNumberType())
    .OPTIONAL_INPUT(strides, TensorType::IndexNumberType())
    .ATTR(begin_mask, Int, 0)
    .ATTR(end_mask, Int, 0)
    .ATTR(ellipsis_mask, Int, 0)
    .ATTR(new_axis_mask, Int, 0)
    .ATTR(shrink_axis_mask, Int, 0)
    .OUTPUT(y, TensorType({BasicType(), DT_BOOL, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E8M0}))
    .OP_END_FACTORY_REG(StridedSliceV2)

REG_OP(StridedSliceV3)
    .INPUT(x, TensorType({BasicType(), DT_BOOL, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E8M0}))
    .INPUT(begin, TensorType::IndexNumberType())
    .INPUT(end, TensorType::IndexNumberType())
    .OPTIONAL_INPUT(axes, TensorType::IndexNumberType())
    .OPTIONAL_INPUT(strides, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({BasicType(), DT_BOOL, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E8M0}))
    .OP_END_FACTORY_REG(StridedSliceV3)

REG_OP(TensorMove)
    .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT32, DT_INT16, DT_UINT16, DT_INT8, DT_UINT8,
                          DT_UINT64, DT_INT64, DT_BOOL, DT_BF16, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_COMPLEX32, DT_COMPLEX64}))
    .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT32, DT_INT16, DT_UINT16, DT_INT8, DT_UINT8,
                           DT_UINT64, DT_INT64, DT_BOOL, DT_BF16, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_COMPLEX32, DT_COMPLEX64}))
    .OP_END_FACTORY_REG(TensorMove)

REG_OP(TransData)
    .INPUT(src, TensorType::BasicType())
    .OUTPUT(dst, TensorType::BasicType())
    .REQUIRED_ATTR(src_format, String)
    .REQUIRED_ATTR(dst_format, String)
    .ATTR(src_subformat, Int, 0)
    .ATTR(dst_subformat, Int, 0)
    .ATTR(groups, Int, 1)
    .OP_END_FACTORY_REG(TransData)

REG_OP(Transpose)
    .INPUT(x, TensorType({DT_BF16,      DT_FLOAT16,   DT_FLOAT,      DT_DOUBLE,   DT_INT64,       DT_INT32,
                          DT_UINT8,     DT_UINT16,    DT_UINT32,     DT_UINT64,   DT_INT8,        DT_INT16,
                          DT_COMPLEX32, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8,    DT_QUINT8,      DT_QINT16,
                          DT_QUINT16,   DT_QINT32,    DT_BOOL,       DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .INPUT(perm, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_BF16,      DT_FLOAT16,   DT_FLOAT,      DT_DOUBLE,   DT_INT64,       DT_INT32,
                           DT_UINT8,     DT_UINT16,    DT_UINT32,     DT_UINT64,   DT_INT8,        DT_INT16,
                           DT_COMPLEX32, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8,    DT_QUINT8,      DT_QINT16,
                           DT_QUINT16,   DT_QINT32,    DT_BOOL,       DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .OP_END_FACTORY_REG(Transpose)

REG_OP(Tril)
    .INPUT(
        x, TensorType(
               {DT_FLOAT16, DT_BF16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16, DT_INT8, DT_INT64, DT_QINT8,
                DT_QUINT8, DT_QINT32, DT_QUINT16, DT_QINT16, DT_UINT16, DT_UINT32, DT_UINT64, DT_BOOL}))
    .ATTR(diagonal, Int, 0)
    .OUTPUT(
        y, TensorType(
               {DT_FLOAT16, DT_BF16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16, DT_INT8, DT_INT64, DT_QINT8,
                DT_QUINT8, DT_QINT32, DT_QUINT16, DT_QINT16, DT_UINT16, DT_UINT32, DT_UINT64, DT_BOOL}))
    .OP_END_FACTORY_REG(Tril)

REG_OP(Triu)
    .INPUT(x, TensorType({TensorType::BasicType(), DT_BOOL}))
    .ATTR(diagonal, Int, 0)
    .OUTPUT(y, TensorType({TensorType::BasicType(), DT_BOOL}))
    .OP_END_FACTORY_REG(Triu)

REG_OP(Unpack)
    .INPUT(x, TensorType::BasicType())
    .DYNAMIC_OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(num, Int)
    .ATTR(axis, Int, 0)
    .OP_END_FACTORY_REG(Unpack)

REG_OP(Unsqueeze)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(axes, ListInt, {})
    .OP_END_FACTORY_REG(Unsqueeze)

REG_OP(UnsqueezeV2)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(axis, ListInt, {})
    .OP_END_FACTORY_REG(UnsqueezeV2)

REG_OP(UnsqueezeV3)
    .INPUT(x, TensorType::ALL())
    .INPUT(axes, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType::ALL())
    .OP_END_FACTORY_REG(UnsqueezeV3)

REG_OP(ViewCopy)
    .INPUT(dst, TensorType({BasicType(), DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .INPUT(dst_size, TensorType::IndexNumberType())
    .INPUT(dst_stride, TensorType::IndexNumberType())
    .INPUT(dst_storage_offset, TensorType::IndexNumberType())
    .INPUT(src, TensorType({BasicType(), DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .INPUT(src_size, TensorType::IndexNumberType())
    .INPUT(src_stride, TensorType::IndexNumberType())
    .INPUT(src_storage_offset, TensorType::IndexNumberType())
    .OUTPUT(dst, TensorType({BasicType(), DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .OP_END_FACTORY_REG(ViewCopy)

REG_OP(DropOutDoMask)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(mask, TensorType({DT_UINT8, DT_UINT1}))
    .INPUT(keep_prob, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(DropOutDoMask)

REG_OP(DropOutDoMaskV3)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(mask, TensorType({DT_UINT8, DT_BOOL}))
    .INPUT(keep_prob, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(DropOutDoMaskV3)

REG_OP(DropOutDoMaskV3D)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(mask, TensorType({DT_UINT8, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(keep_prob, Float)
    .OP_END_FACTORY_REG(DropOutDoMaskV3D)

REG_OP(DropOutV3)
    .INPUT(x, "T")
    .OPTIONAL_INPUT(noise_shape, TensorType({DT_INT64}))
    .INPUT(p, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(seed, TensorType({DT_INT32, DT_INT64}))
    .INPUT(offset, TensorType({DT_INT64}))
    .OUTPUT(y, "T")
    .OUTPUT(mask, TensorType({DT_UINT8}))
    .DATATYPE(T, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(DropOutV3)

REG_OP(RandomStandardNormalV2)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(offset, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(offset, TensorType({DT_INT64}))
    .REQUIRED_ATTR(dtype, Int)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(RandomStandardNormalV2)

REG_OP(RandomUniformIntV2)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(min, TensorType({DT_INT32, DT_INT64}))
    .INPUT(max, TensorType({DT_INT32, DT_INT64}))
    .INPUT(offset, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(offset, TensorType({DT_INT64}))
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(RandomUniformIntV2)

REG_OP(RandomUniformV2)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(offset, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(offset, TensorType({DT_INT64}))
    .REQUIRED_ATTR(dtype, Int)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(RandomUniformV2)

REG_OP(StatelessBernoulli)
    .INPUT(shape, TensorType({ DT_INT32, DT_INT64}))
    .INPUT(prob, TensorType({ DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BF16}))
    .INPUT(seed, TensorType({ DT_INT64 }))
    .INPUT(offset, TensorType({ DT_INT64 }))
    .OUTPUT(y, TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32,
                            DT_INT64, DT_UINT64, DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BF16}))
    .ATTR(dtype, Type, DT_FLOAT)
    .OP_END_FACTORY_REG(StatelessBernoulli)

REG_OP(StatelessDropOutGenMask)
    .INPUT(shape, TensorType({ DT_INT32, DT_INT64 }))
    .INPUT(prob, TensorType({ DT_FLOAT16, DT_FLOAT, DT_BF16 }))
    .INPUT(seed, TensorType({ DT_INT32, DT_INT64 }))
    .INPUT(seed1, TensorType({ DT_INT32, DT_INT64 }))
    .OPTIONAL_INPUT(offset, TensorType({ DT_INT64 }))
    .OUTPUT(y, TensorType({ DT_UINT8 }))
    .OP_END_FACTORY_REG(StatelessDropOutGenMask)

REG_OP(StatelessRandomChoiceWithMask)
    .INPUT(x, TensorType({DT_BOOL}))
    .INPUT(count, TensorType({DT_INT32}))
    .INPUT(seed, TensorType({DT_INT64}))
    .INPUT(offset, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .OUTPUT(mask, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(StatelessRandomChoiceWithMask)

REG_OP(StatelessRandomNormalV2)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(key, TensorType({DT_UINT64}))
    .INPUT(counter, TensorType({DT_UINT64}))
    .INPUT(alg, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(dtype, Type, DT_FLOAT)
    .OP_END_FACTORY_REG(StatelessRandomNormalV2)

REG_OP(StatelessRandomUniformV2)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(key, TensorType({DT_UINT64}))
    .INPUT(counter, TensorType({DT_UINT64}))
    .INPUT(alg, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16, DT_DOUBLE}))
    .ATTR(dtype, Type, DT_FLOAT)
    .OP_END_FACTORY_REG(StatelessRandomUniformV2)

REG_OP(TruncatedNormalV2)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(offset, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({ DT_BF16, DT_FLOAT16, DT_FLOAT }))
    .OUTPUT(offset, TensorType({ DT_INT64 }))
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .ATTR(dtype, Int, 0)
    .OP_END_FACTORY_REG(TruncatedNormalV2)

REG_OP(ExpandDims)
    .INPUT(x, TensorType::ALL())
    .INPUT(axis, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType::ALL())
    .OP_END_FACTORY_REG(ExpandDims)

REG_OP(Identity)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .OP_END_FACTORY_REG(Identity)

REG_OP(StatelessBernoulliV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(seed, TensorType({DT_INT64}))
    .INPUT(offset, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32,
        DT_INT64, DT_UINT64, DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BF16}))
    .ATTR(dtype, Type, DT_UNDEFINED)
    .OP_END_FACTORY_REG(StatelessBernoulliV2)

REG_OP(LeftShift)
    .INPUT(x, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, \
           DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .INPUT(y, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, \
           DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .OUTPUT(z, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, \
            DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .OP_END_FACTORY_REG(LeftShift)

REG_OP(Correlation)
    .INPUT(filter, TensorType({DT_FLOAT16, DT_INT8}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_INT32}))
    .ATTR(groups, Int, 1)
    .OP_END_FACTORY_REG(Correlation)

REG_OP(OneHotD)
    .INPUT(x, TensorType({DT_UINT8, DT_INT32}))
    .INPUT(on_value, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT8,
                                 DT_INT8}))
    .INPUT(off_value, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT8,
                                  DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT8, DT_INT8}))
    .REQUIRED_ATTR(depth, Int)
    .ATTR(axis, Int, -1)
    .OP_END_FACTORY_REG(OneHotD)

REG_OP(Shape)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(dtype, Int, DT_INT32)
    .OP_END_FACTORY_REG(Shape)

REG_OP(MatrixDiagV2)
    .INPUT(diagonal, TensorType::BasicType())
    .INPUT(k, TensorType({DT_INT32}))
    .INPUT(num_rows, TensorType({DT_INT32}))
    .INPUT(num_cols, TensorType({DT_INT32}))
    .INPUT(padding_value, TensorType::BasicType())
    .OUTPUT(output, TensorType::BasicType())
    .OP_END_FACTORY_REG(MatrixDiagV2)

REG_OP(Adds)
     .INPUT(x, TensorType({DT_FLOAT, DT_INT16, DT_INT32, DT_FLOAT16, DT_BF16}))
     .OUTPUT(y, TensorType({DT_FLOAT, DT_INT16, DT_INT32, DT_FLOAT16, DT_BF16}))
     .REQUIRED_ATTR(value,Float)
     .OP_END_FACTORY_REG(Adds)

REG_OP(Multinomial)
    .INPUT(logits, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(num_samples, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(dtype, Type, DT_INT64)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(Multinomial)

REG_OP(PhonyConcat)
    .DYNAMIC_INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .REQUIRED_ATTR(concat_dim, ListInt)
    .REQUIRED_ATTR(N, ListInt)
    .ATTR(keep_input_offset, Bool, true)
    .OP_END_FACTORY_REG(PhonyConcat)

REG_OP(PhonySplit)
    .INPUT(x, TensorType::ALL())
    .DYNAMIC_OUTPUT(y, TensorType::ALL())
    .REQUIRED_ATTR(split_dim, ListInt)
    .REQUIRED_ATTR(num_split, ListInt)
    .ATTR(keep_output_offset, Bool, true)
    .OP_END_FACTORY_REG(PhonySplit)

REG_OP(RandomUniformInt)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(min, TensorType({DT_INT32, DT_INT64}))
    .INPUT(max, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(RandomUniformInt)

REG_OP(RandomUniform)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BF16}))
    .REQUIRED_ATTR(dtype, Type)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(RandomUniform)

REG_OP(Trilu)
    .INPUT(x, "T")
    .OPTIONAL_INPUT(k, "T_K")
    .ATTR(upper, Int, 0)
    .OUTPUT(y, "T")
    .DATATYPE(T, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16,
                          DT_INT8, DT_INT64, DT_QINT8, DT_QUINT8, DT_QINT32, DT_QUINT16, DT_QINT16,
                          DT_UINT16, DT_UINT32, DT_UINT64, DT_BOOL}))
    .DATATYPE(T_K, TensorType({DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(Trilu)

REG_OP(RandomStandardNormal)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BF16}))
    .REQUIRED_ATTR(dtype, Type)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(RandomStandardNormal)

}  // namespace ge

#endif // OP_MATH_PROTO_H_
