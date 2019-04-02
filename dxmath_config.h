/*
 * Copyright (c) 2018-2019 Transnat Games
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

#ifndef DXMATH_CONFIG_HEADER
#define DXMATH_CONFIG_HEADER
#pragma once

/* This file is completely written from scratch for use in C */

/* Definitions for DX Math. These are replacements for the C++ attributes in
 * the origin MS release.
 */

/* Alignment define */
#ifndef DXMATH_NO_INTRINSICS
#ifdef _MSC_VER
#define DXMATH_ALIGN(KEYWORD, N) __declspec(align(N)) KEYWORD
#else
#define DXMATH_ALIGN(KEYWORD, N) KEYWORD __attribute__((__aligned__ (N)))
#endif
#else
/* yes DXMATH_NO_INTRINSICS */
#define DXMATH_ALIGN(KEYWORD, _) KEYWORD
#endif

#ifdef __GNUC__
#define DXMATH_ALIGNED_VAR(TYPE, NAME, N) TYPE NAME __attribute__((__aligned__ (N)))
#elif defined _MSC_VER
#define DXMATH_ALIGNED_VAR(TYPE, NAME, N) __declspec(align(N)) TYPE NAME
#else
#define DXMATH_ALIGNED_VAR(TYPE, NAME, _) TYPE NAME
#endif

#ifndef DXMATH_ALIGN_STRUCT
#define DXMATH_ALIGN_STRUCT(N) DXMATH_ALIGN(struct, N)
#endif

#ifndef DXMATH_ALIGN_UNION
#define DXMATH_ALIGN_UNION(N) DXMATH_ALIGN(union, N)
#endif

/* Get a pure function and a pure-out-function attribute, if possible. */

#if defined(_MSC_VER) && (_MSC_VER >= 1500)
#define DXMATH_CONSTEXPR _Check_return_ __declspec(noalias)
#define DXMATH_CONSTEXPR_OUT_ARG __declspec(noalias)
#elif defined(__GNUC__) && (__GNUC__ > 2)

#define DXMATH_CONSTEXPR __attribute__((pure, warn_unused_result))

#if __GNUC__ >= 4
#define DXMATH_CONSTEXPR_OUT_ARG __attribute__((nothrow))
#else
#define DXMATH_CONSTEXPR_OUT_ARG
#endif

#else
#define DXMATH_CONSTEXPR
#define DXMATH_CONSTEXPR_OUT_ARG
#endif

#endif /* DXMATH_CONFIG_HEADER */
