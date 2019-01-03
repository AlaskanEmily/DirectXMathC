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

#ifndef DXMATH_ARCH_HEADER
#define DXMATH_ARCH_HEADER
#pragma once

/* This file is completely written from scratch for use in C. */

/* Platform macros for DXMath. */

/* Checks specifically for x86/amd64 and ARM/AARCH, which have many more
 * compilers and some redundant macros. The later checks are simpler
 */
#if defined(_MSC_VER) || defined(__WATCOMC__) || defined(__BORLANDC__)

#if defined( _M_ARM64 )
#define DXMATH_IS_AARCH64
#elif defined(_M_ARM)
#define DXMATH_IS_ARM32
#endif

#if defined(_M_X64)
#define DXMATH_IS_AMD64
#elif defined(_M_IX86) || defined(__X86__)
/* __X86__ is some very old MSVC and Pre-Sybase Watcom */
#define DXMATH_IS_X86
#endif

#elif defined(__GNUC__) || defined(__clang__) || defined(__SUNPRO_C)

#if defined(__aarch64__)
#define DXMATH_IS_AARCH64
#elif defined(__arm__)
#define DXMATH_IS_ARM32
#endif

#if defined(__amd64)
#define DXMATH_IS_AMD64
#elif defined(__i386) || defined(_X86_)
/* __i386 is Sun and GCC/Clang/Unix, _X86_ is MinGW */
#define DXMATH_IS_X86
#endif

#elif  defined(__CC_ARM) defined(__CC_NORCROFT)

/* BBC C or the compatible Norcroft C */
#ifdef __ARMCC_VERSION
#define DXMATH_IS_ARM32
#endif

#endif


#ifdef __sh__

#define DXMATH_IS_DREAMCAST

#elif defined(__sparcv9) || defined(__sparc_v9__)
/* __sparcv9 is Sun Studio, __sparc_v9__ is GCC and Clang */

#define DXMATH_IS_ULTRASPARC

#elif defined(DXMATH_IS_X86) || defined(DXMATH_IS_AMD64)

#define DXMATH_IS_INTEL

#elif defined(DXMATH_IS_ARM32) || defined(DXMATH_IS_AARCH64)

#define DXMATH_IS_ARM

#endif

#if defined(_MSC_VER) && !defined(__WATCOMC__) && !defined(__BORLANDC__)

#if !defined(DXMATH_IS_ARM) && !defined(DXMATH_NO_INTRINSICS)
#define DXMATH_VECTORCALL 1
#else
#endif

#if defined(DXMATH_VECTORCALL) && defined _MSC_VER
#define DXMATH_CALL __forceinline __vectorcall
#else
#define DXMATH_CALL __forceinline __fastcall
#endif

#elif defined __CYGWIN__

#ifdef DXMATH_IS_X86
#define DXMATH_CALL __attribute__((nothrow,fastcall,sseregparm))
#else
#define DXMATH_CALL __attribute__((nothrow,fastcall))
#endif

#elif defined __GNUC__

#if defined(DXMATH_IS_X86) && !defined __clang__
#define DXMATH_CALL __attribute__((nothrow,sseregparm))
#else
#define DXMATH_CALL __attribute__((nothrow))
#endif

#else

#define DXMATH_CALL

#endif

#endif /* DXMATH_ARCH_HEADER */
