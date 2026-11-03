# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
expm1
"""

from impl.ops_legacy.util.platform_adapter import tbe
from impl.ops_legacy.util.platform_adapter import tvm
from impl.ops_legacy.util.platform_adapter import tbe_platform
from impl.ops_legacy.util.platform_adapter import classify
from impl.ops_legacy.util.platform_adapter import OpPatternMode
from impl.ops_legacy.util.platform_adapter import shape_util
from impl.ops_legacy.util.platform_adapter import para_check
from impl.ops_legacy.util.platform_adapter import register_operator
from impl.ops_legacy.util.platform_adapter import register_operator_compute
from impl.ops_legacy.util import util_soc_common


def expm1_compute_v1(input_x):
    # `define a scalar , value = -0.000400263845157291003292`
    scalar_negative_first_value = -0.000400263845157291003292
    # `define a scalar , value = 0.004163021669203088`
    scalar_second_value = 0.004163021669203088
    # `define a scalar , value = -0.04166634504074566`
    scalar_negative_third_value = -0.04166634504074566
    # `define a scalar , value = 0.5`
    scalar_fourth_value = 0.5
    # `define a scalar , value = -2`
    scalar_negative_fifth_value = -2
    # `define a scalar , value = -1`
    scalar_negative_sixth_value = -1
    # `define a scalar , value = 0.48`
    scalar_seventh_value = 0.48
    dtype = input_x.dtype
    shape = input_x.shape
    dtype_x = input_x.dtype
    if dtype.lower() == "float16":
        input_x = tbe.cast_to(input_x, "float32")
        dtype = input_x.dtype

    a = tbe.vabs(input_x)
    x = tbe.vmul(input_x, input_x)
    x1 = tbe.vmuls(x, tvm.const(scalar_negative_first_value, dtype))
    x2 = tbe.vadds(x1, tvm.const(scalar_second_value, dtype))
    x3 = tbe.vmul(x1, x2)
    x4 = tbe.vadds(x3, tvm.const(scalar_negative_third_value, dtype))
    x5 = tbe.vmul(x, x4)
    x6 = tbe.vadds(x5, tvm.const(scalar_fourth_value, dtype))
    s1 = tbe.vmul(x6, input_x)
    s2 = tbe.vmuls(s1, tvm.const(scalar_negative_fifth_value, dtype))
    s3 = tbe.vadds(s1, tvm.const(scalar_negative_sixth_value, dtype))
    value1 = tbe.vdiv(s2, s3)
    s4 = tbe.vexp(input_x)
    value2 = tbe.vadds(s4, tvm.const(scalar_negative_sixth_value, dtype))
    const_two_tensor = tbe.broadcast(tvm.const(scalar_seventh_value, dtype), shape)
    cmpzore = tbe.vcmp(a, const_two_tensor, 'lt')
    res = tbe.vsel(cmpzore, value1, value2)

    if dtype_x.lower() == "float16":
        res = tbe.cast_to(res, dtype_x)

    return res


# 'pylint: disable=locally-disabled,too-many-locals,invalid-name
def _expm1_taylor_compute(input_x):
    """
    Calculate e^x - 1, Use seventh order taylor expansion
    e^x = 1 + x + (x^2 / 2!) + (x^3 / 3!) +  (x^4 / 4!) + (x^5 / 5!) + \
          (x^6 / 6!) + (x^7 / 7!)
    e^x - 1 = x + (x^2 / 2!) + (x^3 / 3!) +  (x^4 / 4!) + (x^5 / 5!) + \
            (x^6 / 6!) + (x^7 / 7!)

    Parameters:
    ----------
    input_x : TVM tensor, the placeholder of input_x

    Returns : A Tensor. Has the same type as input_x.
    -------
    """
    # define second order parameter , value = 1 / 2.0
    taylor_second_order_param_value = 1 / 2.0
    # define third order parameter , value = 1 / 6.0
    taylor_third_order_param_value = 1 / 6.0
    # define fourth order parameter , value = 1 / 24.0
    taylor_fourth_order_param_value = 1 / 24.0
    # define fifth order parameter , value = 1 / 120.0
    taylor_fifth_order_param_value = 1 / 120.0
    # define sixth order parameter , value = 1 / 720.0
    taylor_sixth_order_param_value = 1 / 720.0
    # define seventh order parameter , value = 1 / 5040.0
    taylor_seventh_order_param_value = 1 / 5040.0
    # calculate second order tayloy section : x^2 / 2!
    taylor_second_order_param = tvm.const(taylor_second_order_param_value, "float32")
    data_power_2 = tbe.vmul(input_x, input_x)
    data_power_2_div_2 = tbe.vmuls(data_power_2,
                                   taylor_second_order_param)

    # calculate third order tayloy section : x^3 / 3!
    taylor_third_order_param = tvm.const(taylor_third_order_param_value, "float32")
    data_power_3 = tbe.vmul(data_power_2, input_x)
    data_power_3_div_6 = tbe.vmuls(data_power_3,
                                   taylor_third_order_param)

    # calculate fourth order tayloy section : x^4 / 4!
    taylor_fourth_order_param = tvm.const(taylor_fourth_order_param_value, "float32")
    data_power_4 = tbe.vmul(data_power_3, input_x)
    data_power_4_div_24 = tbe.vmuls(data_power_4,
                                    taylor_fourth_order_param)

    # calculate fifth order tayloy section : x^5 / 5!
    taylor_fifth_order_param = tvm.const(taylor_fifth_order_param_value, "float32")
    data_power_5 = tbe.vmul(data_power_4, input_x)
    data_power_5_div_120 = tbe.vmuls(data_power_5,
                                     taylor_fifth_order_param)

    # xcalculate sixth order tayloy section : ^6 / 6!
    taylor_sixth_order_param = tvm.const(taylor_sixth_order_param_value, "float32")
    data_power_6 = tbe.vmul(data_power_5, input_x)
    data_power_6_div_720 = tbe.vmuls(data_power_6,
                                     taylor_sixth_order_param)

    # calculate seventh order tayloy section : x^7 / 7!
    taylor_seventh_order_param = tvm.const(taylor_seventh_order_param_value,
                                           "float32")
    data_power_7 = tbe.vmul(data_power_6, input_x)
    data_power_7_div_5040 = tbe.vmuls(data_power_7,
                                      taylor_seventh_order_param)

    res_second_taylor = tbe.vadd(input_x, data_power_2_div_2)
    res_third_taylor = tbe.vadd(res_second_taylor, data_power_3_div_6)
    res_fourth_taylor = tbe.vadd(res_third_taylor, data_power_4_div_24)
    res_fifth_taylor = tbe.vadd(res_fourth_taylor, data_power_5_div_120)
    res_sixth_taylor = tbe.vadd(res_fifth_taylor, data_power_6_div_720)
    res = tbe.vadd(res_sixth_taylor, data_power_7_div_5040)

    return res


# 'pylint: disable=locally-disabled,too-many-locals,invalid-name
def _expm1_mini_compute(mini_res, input_x, shape):
    """
    do element-wise e^x - 1 compute in mini scene
    f(x) = e^x - 1,
    x <= TAYLOR_NEGATIVE_THRESHOLD or x >= TAYLOR_POSITIVE_THRESHOLD
    f(x) = seventh taylor computer,
    TAYLOR_NEGATIVE_THRESHOLD < x < TAYLOR_POSITIVE_THRESHOLD

    Parameters:
    ----------
    mini_res: TVM tensor, the tensor of e^x - 1

    input_x : TVM tensor, the placeholder of input data

    shape : tuple, the shape of input data

    Returns : A Tensor. Has the same type as mini_res.
    -------
    """
    # define taylor negative threshold , value = -0.7
    taylor_negative_threshold = -0.7
    # define taylor positive threshold , value = 1.7
    taylor_positive_threshold = 1.7
    taylor_res = _expm1_taylor_compute(input_x)

    input_right_border = tvm.const(taylor_positive_threshold, "float32")
    tensor_input_right_border = tbe.broadcast(input_right_border, shape)

    input_left_border = tvm.const(taylor_negative_threshold, "float32")
    tensor_input_left_border = tbe.broadcast(input_left_border, shape)

    b_gt_left_border = tbe.vcmp(input_x, tensor_input_left_border, 'gt')
    exp_taylor_neg = tbe.vsel(b_gt_left_border, taylor_res, mini_res)

    b_lt_right_border = tbe.vcmp(input_x, tensor_input_right_border, 'lt')
    mini_res = tbe.vsel(b_lt_right_border, exp_taylor_neg, mini_res)

    return mini_res


# 'pylint: disable=locally-disabled,too-many-locals,unused-argument,invalid-name
@register_operator_compute("Expm1", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def expm1_compute(input_x, output_y, kernel_name="expm1"):
    """
    algorithm: expm1
    calculating data's expm1, y = (e^x) - 1
    in cloud scene, for all inputs :
    f(x) = e^x - 1,
    in mini scene :
    f(x) = e^x - 1,
    x <= TAYLOR_NEGATIVE_THRESHOLD or x >= TAYLOR_POSITIVE_THRESHOLD
    f(x) = seventh taylor computer,
    TAYLOR_NEGATIVE_THRESHOLD < x < TAYLOR_POSITIVE_THRESHOLD

    Parameters
    ----------
    input_x : TVM tensor, the placeholders of input data
    output_y : dict, shape and dtype of output, should be same shape
               and type as input
    kernel_name : str, kernel name, default value is "expm1"

    Returns
    -------
    res : the result of compute
    """
    
    if util_soc_common.after_v200():
        flag_exp = tbe_platform.api_check_support("tbe.dsl.vexp", "float32")
        flag_add = tbe_platform.api_check_support("tbe.dsl.vadd", "float32")
        if (not flag_exp) or (not flag_add):
            return expm1_compute_v1(input_x)
        else:
            dtype = input_x.dtype
            if dtype.lower() == "float16":
                input_x = tbe.cast_to(input_x, "float32")
            scalar_negative_one_value = -1.0
            scalar_negative_one = tvm.const(scalar_negative_one_value, "float32")
            exp_res = tbe.vexp(input_x)
            res = tbe.vadds(exp_res, scalar_negative_one)
            if dtype.lower() == "float16":
                res = tbe.cast_to(res, dtype)
            return res

    # `define a scalar , value = -1`
    scalar_negative_one_value = -1.0
    dtype = input_x.dtype
    shape = input_x.shape
    flag_cloud = tbe_platform.api_check_support("tbe.dsl.vexp", "float32")
    flag_mini = tbe_platform.api_check_support("tbe.dsl.vadd", "float32")
    
    
    if dtype.lower() == "float16" and flag_cloud:
        input_x = tbe.cast_to(input_x, "float32")
    scalar_negative_one = tvm.const(scalar_negative_one_value, "float32")
    exp_res = tbe.vexp(input_x)
    res = tbe.vadds(exp_res, scalar_negative_one)

    if (not flag_cloud) and flag_mini:
        input_x = tbe.cast_to(input_x, "float32")
        res = _expm1_mini_compute(res, input_x, shape)

    if dtype.lower() == "float16" and (flag_cloud or flag_mini):
        res = tbe.cast_to(res, dtype)

    return res


@register_operator("Expm1")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def expm1(input_x, output_y, kernel_name="expm1"):
    """
    algorithm: expm1
    calculating data's expm1, y = (e^x) - 1

    Parameters
    ----------
    input_x : dict,shape and dtype of input, only support bfloat16, float16, float32
    output_y: dict,shape and dtype of output, should be same shape
              and type as input
    kernel_name : str, kernel name, default value is "expm1"

    Returns
    -------
    None
    """
    input_dtype = input_x.get("dtype").lower()
    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")

    schedules, tensors = [], []
    ins = classify([input_x], OpPatternMode.ELEWISE)

    for (classify_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([classify_x])[0]
            data_input = tvm.placeholder(x_shape, dtype=input_dtype, name="data_input")
            res = expm1_compute(data_input, output_y, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {
        "name": kernel_name,
        "tensor_list": tensors,
        "bool_storage_as_list": False
    }

    tbe.build(schedules, config)
