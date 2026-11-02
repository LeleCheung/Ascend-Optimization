/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef IMPL_SIMT_API_CPP_KERNEL_SIMT_TRANSCENDENTAL_INTERFACE_IMPL_H
#define IMPL_SIMT_API_CPP_KERNEL_SIMT_TRANSCENDENTAL_INTERFACE_IMPL_H

#include "impl/simt_api/cpp/dav_c310/kernel_simt_transcendental_impl.h"

namespace AscendC {
namespace Simt {

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Tan(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type of input only supports float.");
    return TanImpl(x);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Tanh(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type T only supports float.");
    return TanhImpl(x);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Tanpi(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type of input only supports float.");
    return TanPiImpl(x);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Atan(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type T only supports float.");
    return AtanImpl(x);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Atan2(T y, T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type T only supports float.");
    return Atan2Impl(y, x);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Atanh(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type T only supports float.");
    return AtanhImpl(x);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Cos(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type of input only supports float.");
    return CosImpl(x);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Cosh(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type of input only supports float.");
    return CoshImpl(x);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Cospi(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type of input only supports float.");
    return CospiImpl(x);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Asin(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type of input only supports float.");
    return AsinImpl(x);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Acos(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type of input only supports float.");
    return AcosImpl(x);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Acosh(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type of input only supports float.");
    return AcoshImpl(x);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Sin(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type of input only supports float.");
    return SinImpl(x);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Sinh(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type of input only supports float.");
    return SinhImpl(x);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Sinpi(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type of input only supports float.");
    return SinpiImpl(x);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Asinh(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type of input only supports float.");
    return AsinhImpl(x);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline void Sincos(T x, T &s, T &c)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type of input only supports float.");
    SinCosImpl(x, s, c);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline void Sincospi(T x, T &s, T &c)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type of input only supports float.");
    SinCospiImpl(x, s, c);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Exp(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type of input only supports float.");
    return ExpImpl(x);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Exp2(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type T only supports float.");
    return Exp2Impl(x);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Exp10(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type T only supports float.");
    return Exp10Impl(x);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Expm1(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type T only supports float.");
    return Expm1Impl(x);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Sqrt(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type T only supports float.");
    return SqrtImpl(x);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Rsqrt(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type T only supports float.");
    return RsqrtImpl(x);
}

template <typename T, typename U>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Frexp(T x, U &exp)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type of input(x) only supports float.");
    static_assert(SupportTypeSimtInternel<U, int>, "Input type of input(exp) only supports int.");
    return FrexpImpl(x, exp);
}

template <typename T, typename U>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Ldexp(T x, U exp)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type of input(x) only supports float.");
    static_assert(SupportTypeSimtInternel<U, int>, "Input type of input(exp) only supports int.");
    return LdexpImpl(x, exp);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Hypot(T x, T y)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type of input only supports float.");
    return HypotImpl(x, y);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Rhypot(T x, T y)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type of input only supports float.");
    return RhypotImpl(x, y);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Norm3d(T a, T b, T c)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type only supports float.");
    return Norm3dImpl(a, b, c);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Rnorm3d(T a, T b, T c)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type only supports float.");
    return Rnorm3dImpl(a, b, c);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Norm4d(T a, T b, T c, T d)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type only supports float.");
    return Norm4dImpl(a, b, c, d);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Rnorm4d(T a, T b, T c, T d)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type only supports float.");
    return Rnorm4dImpl(a, b, c, d);
}

template <typename T, typename U>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline U Norm(T n, U* a)
{
    static_assert(SupportTypeSimtInternel<T, int>, "Input(n) type only supports int.");
    static_assert(SupportTypeSimtInternel<U *, float *, __ubuf__ float *, __gm__ float *>,
        "Input(a) type only supports float*, gm float*, ubuf float*.");
    return NormImpl(n, a);
}

template <typename T, typename U>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline U Rnorm(T n, U* a)
{
    static_assert(SupportTypeSimtInternel<T, int>, "Input(n) type only supports int.");
    static_assert(SupportTypeSimtInternel<U *, float *, __ubuf__ float *, __gm__ float *>,
        "Input(a) type only supports float*, gm float*, ubuf float*.");
    return RnormImpl(n, a);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Log(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type only supports float.");
    return LogImpl(x);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Pow(T x, T y)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type only supports float.");
    return PowImpl(x, y);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Log2(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type T only supports float.");
    return Log2Impl(x);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Log10(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type T only supports float.");
    return Log10Impl(x);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Log1p(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type T only supports float.");
    return Log1pImpl(x);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Logb(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type T only supports float.");
    return LogbImpl(x);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline int Ilogb(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type T only supports float.");
    return ILogbImpl(x);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Cbrt(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type only supports float.");
    return CbrtImpl(x);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Rcbrt(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type only supports float.");
    return RcbrtImpl(x);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Erf(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type only supports float.");
    return ErfImpl(x);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Erfc(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type only supports float.");
    return ErfcImpl(x);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Erfinv(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type only supports float.");
    return ErfinvImpl(x);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Erfcinv(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type only supports float.");
    return ErfcinvImpl(x);
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Erfcx(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type only supports float.");
    return ErfcxImpl(x);
}

template<typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Tgamma(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type only supports float.");
    return TgammaImpl(x);
}

template<typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Lgamma(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type only supports float.");
    return LgammaImpl(x);
}

template<typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T CylBesselI0(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type only supports float.");
    return CylBesselI0Impl(x);
}

template<typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T CylBesselI1(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type only supports float.");
    return CylBesselI1Impl(x);
}

template<typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T Normcdf(T x)
{
    static_assert(SupportTypeSimtInternel<T, float>, "Input type only supports float.");
    return NormcdfImpl(x);
}

}  // namespace Simt
}  // namespace AscendC
#endif  // IMPL_SIMT_API_CPP_KERNEL_SIMT_TRANSCENDENTAL_INTERFACE_IMPL_H
