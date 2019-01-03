/*
 * Copyright (c) 2018 Microsoft Corp, 2018-2019 Transnat Games
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef DXMATH_INT_HEADER
#define DXMATH_INT_HEADER
#pragma once

#include "dxmath_arch.h"

/* Intrinsic selection/configuration for DXMath. */
#ifndef DXMATH_NO_INTRINSICS

#ifdef DXMATH_IS_INTEL

#if !defined(DXMATH_AVX2_INTRINSICS) && defined(__AVX2__)
#define DXMATH_AVX2_INTRINSICS
#endif

#if !defined(DXMATH_FMA3_INTRINSICS) && defined(DXMATH_AVX2_INTRINSICS)
#define DXMATH_FMA3_INTRINSICS
#endif

#if !defined(DXMATH_F16C_INTRINSICS) && defined(DXMATH_AVX2_INTRINSICS)
#define DXMATH_F16C_INTRINSICS
#endif

#if defined(DXMATH_FMA3_INTRINSICS) && !defined(DXMATH_AVX_INTRINSICS)
#define DXMATH_AVX_INTRINSICS
#endif

#if defined(DXMATH_F16C_INTRINSICS) && !defined(DXMATH_AVX_INTRINSICS)
#define DXMATH_AVX_INTRINSICS
#endif

#if !defined(DXMATH_AVX_INTRINSICS) && defined(__AVX__)
#define DXMATH_AVX_INTRINSICS
#endif

#if defined(DXMATH_AVX_INTRINSICS) && !defined(DXMATH_SSE4_INTRINSICS)
#define DXMATH_SSE4_INTRINSICS
#endif

#if defined(DXMATH_SSE4_INTRINSICS) && !defined(DXMATH_SSE3_INTRINSICS)
#define DXMATH_SSE3_INTRINSICS
#endif

#if defined(DXMATH_SSE3_INTRINSICS) && !defined(DXMATH_SSE_INTRINSICS)
#define DXMATH_SSE_INTRINSICS
#endif

#endif /* DXMATH_IS_INTEL */

#ifdef DXMATH_IS_ARM

#if defined(DXMATH_USE_NEON) && !defined(DXMATH_ARM_NEON_INTRINSICS)
#define DXMATH_ARM_NEON_INTRINSICS
#endif

#endif /* DXMATH_IS_ARM */

#endif /* !DXMATH_NO_INTRINSICS */

#endif /* DXMATH_INT_HEADER */
