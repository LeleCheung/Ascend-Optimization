/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ASCENDC_MODULE_SIMT_TRANSCENDENTAL_INTERFACE_H
#define ASCENDC_MODULE_SIMT_TRANSCENDENTAL_INTERFACE_H

#include "simt_api/device_types.h"

namespace AscendC {
namespace Simt {

/**
 * The trigonometric tangent of an angle
 * @param x an angle, in radians.
 * @return the tangent of the x.
 */
template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Tan(T x);

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Tanh(T x);

/**
 * The trigonometric tangent of an angle*pi
 * @param x an angle, in radians.
 * @return the tangent of the x*PI.
 */
template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Tanpi(T x);

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Atan(T x);

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Atan2(T y, T x);

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Atanh(T x);

/**
 * The trigonometric cosine of an angle
 * @param x an angle, in radians.
 * @return the cosine of the x.
 */
template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Cos(T x);

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Cosh(T x);

/**
 * The trigonometric cosine of an angle*pi
 * @param x an angle, in radians.
 * @return the cosine of the x*pi.
 */
template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Cospi(T x);

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Asin(T x);

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Acos(T x);

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Acosh(T x);

/**
 * The trigonometric sine of an angle
 * @param x an angle, in radians.
 * @return the sine of the x.
 */
template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Sin(T x);

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Sinh(T x);

/**
 * The trigonometric sine of an angle
 * @param x an angle, in radians.
 * @return the sine of the x.
 */
template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Sinpi(T x);

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Asinh(T x);

/**
 * The trigonometric sine,cosine of an angle
 * @param x an angle, in radians.
 * @return s the sine of the x.
 *         c the cosine of the x.
 */
template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline void Sincos(T x, T &s, T &c);

/**
 * The trigonometric sine,cosine of an angle*pi
 * @param x an angle, in radians.
 * @return s the sine of the x*pi.
 *         c the cosine of the x*pi.
 */
template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline void Sincospi(T x, T &s, T &c);

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Exp(T x);

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Exp2(T x);

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Exp10(T x);

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Expm1(T x);

/**
 * The square root of x
 * @param   x  a float value.
 * @return  the square root of x
 *          if x<0 return Nan
 */
template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Sqrt(T x);

/**
 * The reciprocal of the square root of x
 * @param   x  a float value.
 * @return  the square root of x
 *          if x<0 return Nan
 */
template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Rsqrt(T x);

/**
 * The mantissa,exponent of x
 * @param x  a float value.
 * @return   the mantissa of the x
 *         exp: the exponent of the x
 */
template <typename T, typename U>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Frexp(T x, U &exp);

/**
 * The value of x multiplied by 2 exp.
 * @param x a float value.
 *        exp a int value.
 * @return  x * 2^exp
 */
template <typename T, typename U>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Ldexp(T x, U exp);

/**
 * The square root of (x^2+y^2)
 * @param   x  a float value.
 *          y  a float value.
 * @return square root of (x^2+y^2)
 */
template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Hypot(T x, T y);

 /**
  * The reciprocal of square root of (x^2+y^2)
  * @param   x  a float value.
  *          y  a float value.
  * @return The reciprocal of square root of (x^2+y^2)
  */
template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Rhypot(T x, T y);

/**
 * The square root of (a^2+b^2+c^2)
 * @param a : float value.
 * @param b : float value.
 * @param c : float value.
 * @return The square root of (a^2+b^2+c^2)
 * Special cases:
 *      If any one of a,b,c is ±INF, return INF.
 *      If any one of a,b,c is NAN and other is not ±INF, return NAN.
 *      If all of a,b,c is 0, return 0.
 *      If sqrt(a^2 + b^2 + c^2) overflows, return INF.
 */
template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Norm3d(T a, T b, T c);

/**
 * The reciprocal of square root of (a^2+b^2+c^2)
 * @param a : float value.
 * @param b : float value.
 * @param c : float value.
 * @return The reciprocal of square root of (a^2+b^2+c^2)
 * Special cases:
 *      If any one of a,b,c is ±INF, return 0.
 *      If any one of a,b,c is NAN and other is not ±INF, return NAN.
 *      If all of a,b,c is 0, return INF.
 *      If sqrt(a^2 + b^2 + c^2) overflows, return INF.
 */
template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Rnorm3d(T a, T b, T c);

/**
 * The square root of (a^2+b^2+c^2+d^2)
 * @param a : float value.
 * @param b : float value.
 * @param c : float value.
 * @param d : float value.
 * @return The square root of (a^2+b^2+c^2+d^2)
 * Special cases:
 *      If any one of a,b,c,d is ±INF, return INF.
 *      If any one of a,b,c,d is NAN and other is not ±INF, return NAN.
 *      If all of a,b,c,d is 0, return 0.
 *      If sqrt(a^2 + b^2 + c^2+ d^2) overflows, return INF.
 */
template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Norm4d(T a, T b, T c, T d);

/**
 * The reciprocal of square root of (a^2+b^2+c^2+d^2)
 * @param a : float value.
 * @param b : float value.
 * @param c : float value.
 * @param d : float value.
 * @return The reciprocal of square root of (a^2+b^2+c^2+d^2)
 * Special cases:
 *      If any one of a,b,c,d is ±INF, return 0.
 *      If any one of a,b,c,d is NAN and other is not ±INF, return NAN.
 *      If all of a,b,c,d is 0, return INF.
 *      If sqrt(a^2 + b^2 + c^2 + d^2) overflows,return 0.
 */
template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Rnorm4d(T a, T b, T c, T d);

/**
 * The square root of sum of squares of the first N elements in array a
 * @param n : int value.
 * @param x : float array
 * @return The square root of (a[0]^2+...+a[n-1]^2)
 * Special cases:
 *      If any one of a[i] is ±INF, return 0.
 *      If any one of a[i] is NAN and other is not ±INF, return NAN.
 *      If all of a,b,c,d is 0, return 0.
 *      If sqrt(a[0]^2 +... + a[n-1]^2) overflows, return INF.
 *      If n is less than 1, return |a[0]|.
 */
template <typename T, typename U>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline U Norm(T n, U* a);

/**
 * The reciprocal of square root of sum of squares of the first N elements in array a
 * @param n : int value.
 * @param x : float array
 * @return The reciprocal of square root of (a[0]^2+...+a[n-1]^2)
 * Special cases:
 *      If any one of a[i] is ±INF, return 0.
 *      If any one of a[i] is NAN and other is not ±INF, return NAN.
 *      If all of a[i] is 0, return INF.
 *      If sqrt(a[0]^2 +... + a[n-1]^2) overflows, return 0.
 *      If n is less than 1, return 1/|a[0]|.
 */
template <typename T, typename U>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline U Rnorm(T n, U* a);

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Log(T x);

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Pow(T x, T y);

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Log2(T x);

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Log10(T x);

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Log1p(T x);

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Logb(T x);

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline int Ilogb(T x);

/**
 * calculates a cube root by input x.
 * @param x a value
 * @return cbrt(x)
 * Special cases:
 *      if x is 0, return 0
 *      if x is Nan, return Nan;
 *      if x is Inf, return Inf;
 *      if x is -Inf, return -Inf;
 */
template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Cbrt(T x);

/**
 * calculates reciprocal of the cube root by input x.
 * @param x a value
 * @return rcbrt(x)
 * Special cases:
 *      if x is 0, return Inf
 *      if x is Nan, return Nan;
 *      if x is Inf, return 0;
 *      if x is -Inf, return 0;
 */
template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Rcbrt(T x);

/**
 * Calculate the error function of the input x.
 * @param x a value
 * @return erf(x)
 * Special cases:
 *      if x is 0, return 0
 *      if x is Inf, return 1;
 *      if x is -Inf, return -1;
 *      if x is Nan, return Nan;
 */
template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Erf(T x);

/**
 * Calculate the complementary error function of the input x.
 * @param x a value
 * @return erfc(x)
 * Special cases:
 *      if x is Inf, return 2;
 *      if x is -Inf, return +0;
 *      if x is Nan, return Nan;
 */
template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Erfc(T x);

/**
 * Calculate the inverse error function of the input x.
 * @param x a value
 * @return erfinv(x)
 * Special cases:
 *      if x is 0, return 0;
 *      if x is 1, return Inf;
 *      if x is -1, return -Inf;
 *      if x outside [-1, 1], return Nan;
 *      if x is Nan, return Nan;
 */
template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Erfinv(T x);

/**
 * Calculate the inverse complementary error function of the input x.
 * @param x a value
 * @return erfcinv(x)
 * Special cases:
 *      if x is 0, return Inf;
 *      if x is 2, return -Inf;
 *      if x outside [0, 2], return Nan;
 *      if x is Nan, return Nan;
 */
template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Erfcinv(T x);


/**
 * Calculate the scaled complementary error function of the input x.
 * @param x a value
 * @return erfcx(x)
 * Special cases:
 *      if x is -Inf, return Inf;
 *      if x is Inf, return +0;
 *      if x is Nan, return Nan;
 */
template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Erfcx(T x);

/**
 * Calculates gamma value by input x.
 * @param x a value
 * @return tgamma(x)
 * Special cases:
 *      if x is 0, return Inf
 *      if x is Nan, return Nan;
 *      if x is Inf, return Inf;
 *      if x is -Inf, return Nan;
 */
template<typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Tgamma(T x);

/**
 * Calculates lgamma value by input x.
 * @param x a value
 * @return lgamma(x)
 * Special cases:
 *      if x is 0, return Inf
 *      if x is Nan, return Nan;
 *      if x is Inf, return Inf;
 *      if x is -Inf, return Inf;
 */
template<typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Lgamma(T x);

/**
 * Calculates CylBesselI0 value by input x.
 * @param x a value
 * @return CylBesselI0(x)
 * Special cases:
 *      if x is 0, return 1;
 *      if x is Nan, return Nan;
 *      if x is Inf, return Inf;
 *      if x is -Inf, return Inf;
 */
template<typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T CylBesselI0(T x);

/**
 * Calculates CylBesselI1 value by input x.
 * @param x a value
 * @return CylBesselI1(x)
 * Special cases:
 *      if x is 0, return 0;
 *      if x is Nan, return Nan;
 *      if x is Inf, return Inf;
 *      if x is -Inf, return -Inf;
 */
template<typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T CylBesselI1(T x);

/**
 * Calculate the standard normal cumulative distribution function for input x.
 * @param x a value
 * @return Normcdf(x)
 * Special cases:
 *      if x is Nan, return Nan;
 *      if x is Inf, return 1;
 *      if x is -Inf, return 0;
 */
template<typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Normcdf(T x);

}  // namespace Simt
}  // namespace AscendC

#include "impl/simt_api/cpp/kernel_simt_transcendental_intf_impl.h"
#endif  // ASCENDC_MODULE_SIMT_TRANSCENDENTAL_INTERFACE_H
