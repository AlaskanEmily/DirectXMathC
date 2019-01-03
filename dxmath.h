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

#ifndef DXMATH_HEADER
#define DXMATH_HEADER
#pragma once

/*
 * This is a modified version of the DirectX Math library, which was originally
 * released by Microsoft under the MIT license for C++.
 *
 * This version has been modified by Transnat Games.
 *
 * This version has been translated to C, had any Visual C/C++ specific
 * features put into conditional defines to allow compiling with any C compiler
 * (include with ANSI options), and has been tested on platforms other than
 * Windows.
 *
 * The intent is to allow the use of DirectX Math vectors and matrices to be
 * used in cross-platform projects, wrapped for use in languages that can only
 * interface with C (such as Mercury and Python), and to allow use with older
 * or less capable C/C++ compilers such as GCC 2 on Haiku or Watcom C/C++.
 *
 * Work to use GCC builtins and inline assembly for Watcom hasn't started yet.
 */

#include "dxmath_config.h"
#include "dxmath_arch.h"
#include "dxmath_int.h"

#include <math.h>
#include <stdint.h>

/* Needed for memcpy */
#include <string.h>

#if defined(_MSC_VER) && defined(DXMATH_SSE_INTRINSICS)
#include <xmmintrin.h>
#include <smmintrin.h>
#endif

#if defined(_MSC_VER) && defined(DXMATH_SSE3_INTRINSICS)
#include <intrin.h>
#endif


#ifdef __cplusplus
extern "C" {
#endif

/*****************************************************************************/
/* Math constants */

#define DXMATH_PI        3.141592654f
#define DXMATH_2PI       6.283185307f
#define DXMATH_1DIVPI    0.318309886f
#define DXMATH_1DIV2PI   0.159154943f
#define DXMATH_PIDIV2    1.570796327f
#define DXMATH_PIDIV4    0.785398163f

/*****************************************************************************/
/* Algorithm constants */

#define DXMATH_SELECT_0   0x00000000
#define DXMATH_SELECT_1   0xFFFFFFFF

#define DXMATH_PERMUTE_0X 0
#define DXMATH_PERMUTE_0Y 1
#define DXMATH_PERMUTE_0Z 2
#define DXMATH_PERMUTE_0W 3
#define DXMATH_PERMUTE_1X 4
#define DXMATH_PERMUTE_1Y 5
#define DXMATH_PERMUTE_1Z 6
#define DXMATH_PERMUTE_1W 7

#define DXMATH_SWIZZLE_X  0
#define DXMATH_SWIZZLE_Y  1
#define DXMATH_SWIZZLE_Z  2
#define DXMATH_SWIZZLE_W  3

#define DXMATH_CRMASK_CR6         0x000000F0
#define DXMATH_CRMASK_CR6TRUE     0x00000080
#define DXMATH_CRMASK_CR6FALSE    0x00000020
#define DXMATH_CRMASK_CR6BOUNDS   DXMATH_CRMASK_CR6FALSE

#define DXMATH_CACHE_LINE_SIZE 64

/* sqrt definition. */
#ifdef __WATCOMC__
/* Watcom's libc lacks C99 math support. This will be kind of slow, but meh. */
#define DXMATH_SQRT(F) ((float)sqrt((double)(F)))
#elif defined __GNUC__
/* This forces GCC to consider this a builtin. */
#define DXMATH_SQRT __builtin_sqrtf
#else
#define DXMATH_SQRT sqrtf
#endif

/*****************************************************************************/
/* Unit conversion */

#define DXMATH_DEGREES_TO_RADIANS(D) ( (D) * (DXMATH_PI / 180.0f) )
#define DXMATH_RADIANS_TO_DEGREES(R) ( (R) * (180.0f / DXMATH_PI) )

/*****************************************************************************/
/* Condition register evaluation proceeding a recording (R) comparison */

#define DXMATH_COMPARE_ALL_TRUE(CR) \
    (((CR) & DXMATH_CRMASK_CR6TRUE) == DXMATH_CRMASK_CR6TRUE)
#define DXMATH_COMPARE_ANY_TRUE(CR) \
    (((CR) & DXMATH_CRMASK_CR6FALSE) != DXMATH_CRMASK_CR6FALSE)
#define DXMATH_COMPARE_ALL_FALSE(CR) \
    (((CR) & DXMATH_CRMASK_CR6FALSE) == DXMATH_CRMASK_CR6FALSE)
#define DXMATH_COMPARE_ANY_FALSE(CR) \
    return (((CR) & DXMATH_CRMASK_CR6TRUE) != DXMATH_CRMASK_CR6TRUE)
#define DXMATH_COMPARE_MIXED(CR) \
    (((CR) & DXMATH_CRMASK_CR6) == 0)
#define DXMATH_COMPARE_ALL_IN_BOUNDS(CR) \
    (((CR) & DXMATH_CRMASK_CR6BOUNDS) == DXMATH_CRMASK_CR6BOUNDS)
#define DXMATH_COMPARE_ANY_OUT_OF_BOUNDS(CR) \
    (((CR) & DXMATH_CRMASK_CR6BOUNDS) != DXMATH_CRMASK_CR6BOUNDS)

#ifdef _MM_SHUFFLE
#define DXMATH_MM_SHUFFLE _MM_SHUFFLE
#else
#define DXMATH_MM_SHUFFLE(A, B, C, D) \
    (((A) << 6) | ((B) << 4) | ((C) << 2) | (D))
#endif

#if defined(DXMATH_AVX_INTRINSICS)
#define DXMATH_PERMUTE_PS(V, C) _mm_permute_ps(V, C)
#else
#define DXMATH_PERMUTE_PS(V, C) _mm_shuffle_ps(V, V, C)
#endif

/*****************************************************************************/
/* Software implementation of a 4 component, 32-bit float vector. */
DXMATH_ALIGN_UNION(16) dxmath_vector4 {
    float vector4_f32[4];
    uint32_t vector4_u32[4];
};

static union dxmath_vector4 DXMATH_CONSTEXPR DXMATH_CALL DXMATH_SoftVector4Infinity(void){
    union dxmath_vector4 vec;
    vec.vector4_u32[0] =
        vec.vector4_u32[1] =
        vec.vector4_u32[2] =
        vec.vector4_u32[3] = 0x7F800000;
    return vec;
}

static union dxmath_vector4 DXMATH_CONSTEXPR DXMATH_CALL DXMATH_SoftVector4Zero(void){
    union dxmath_vector4 vec;
    vec.vector4_f32[0] =
        vec.vector4_f32[1] =
        vec.vector4_f32[2] =
        vec.vector4_f32[3] = 0.0f;
    return vec;
}

/*****************************************************************************/
/* Vector intrinsic: Four 32 bit floating point components aligned on a 16 byte
 * boundary and mapped to hardware vector registers.
 */
#if defined(_MSC_VER)

/* Only use the SSE or NEON types in MSVC when !DXMATH_NO_INTRINSICS */
#if defined(DXMATH_SSE_INTRINSICS)

typedef __m128 DXMATH_VECTOR4;

#define DXMATH_VectorChannel(V, C) ((V).m128_f32[(C)])

static __m128 DXMATH_CONSTEXPR DXMATH_CALL DXMATH_Vector4Infinity(void){
    const uint32_t inf_int = 0x7F800000;
    float inf;
    memcpy(&inf, &inf_int, 4);
    __m128 vec;
    vec.m128_f32[0] =
        vec.m128_f32[1] =
        vec.m128_f32[2] =
        vec.m128_f32[3] = inf;
    return vec;
}

static __m128 DXMATH_CONSTEXPR DXMATH_CALL DXMATH_Vector4Zero(void){
    __m128 vec;
    vec.m128_f32[0] =
        vec.m128_f32[1] =
        vec.m128_f32[2] =
        vec.m128_f32[3] = 0.0f;
    return vec;
}

#elif defined(DXMATH_ARM_NEON_INTRINSICS)
#error TODO! DXMATH_Vector4Infinity
#error TODO! DXMATH_Vector4Zero
typedef float32x4_t DXMATH_VECTOR4;
#else
#define DXMATH_Vector4Infinity DXMATH_SoftVector4Infinity
#define DXMATH_Vector4Zero DXMATH_SoftVector4Zero
typedef union dxmath_vector4 DXMATH_VECTOR4;
#define DXMATH_VectorChannel(V, C) ((V).vector4_f32[(C)])
#endif

#else
/* TODO: GCC intrinsics */
typedef union dxmath_vector4 DXMATH_VECTOR4;

#endif

#if defined(DXMATH_NO_INTRINSICS)

#define DXMATH_VEC_SOFTOP(OUT, V1, V2, OP) do{\
        (OUT).vector4_f32[0] = (V1).vector4_f32[0] OP (V2).vector4_f32[0];\
        (OUT).vector4_f32[1] = (V1).vector4_f32[1] OP (V2).vector4_f32[1];\
        (OUT).vector4_f32[2] = (V1).vector4_f32[2] OP (V2).vector4_f32[2];\
        (OUT).vector4_f32[3] = (V1).vector4_f32[3] OP (V2).vector4_f32[3];\
    }while(0)

static DXMATH_VECTOR4 DXMATH_CONSTEXPR DXMATH_CALL DXMATH_Vector4Set(
    const float x, const float y, const float z, const float w){
    DXMATH_VECTOR4 vec;
    vec.vector4_f32[0] = x;
    vec.vector4_f32[1] = y;
    vec.vector4_f32[2] = z;
    vec.vector4_f32[3] = w;
    return vec;
}

#elif defined(DXMATH_ARM_NEON_INTRINSICS)

static DXMATH_VECTOR4 DXMATH_CONSTEXPR DXMATH_CALL DXMATH_Vector4Set(
    const float xf, const float yf, const float zf, const float wf){
    
    const uint64_t
        x64 = *((uint32_t*)&xf),
        y64 = *((uint32_t*)&yf),
        z64 = *((uint32_t*)&zf),
        w64 = *((uint32_t*)&wf);
    const float32x2_t V0 = vcreate_f32(x64 | (y64 << 32));
    const float32x2_t V1 = vcreate_f32(z64 | (w64 << 32));
    return vcombine_f32(V0, V1);
}

#elif defined(DXMATH_SSE_INTRINSICS)
#define DXMATH_Vector4Set(X, Y, Z, W) _mm_set_ps((W), (Z), (Y), (X))
#endif

/*****************************************************************************/
#ifdef DXMATH_NO_INTRINSICS
static DXMATH_VECTOR4 DXMATH_CONSTEXPR DXMATH_CALL DXMATH_Vector4SelectLane(
    DXMATH_VECTOR4 V, unsigned lane){
    DXMATH_VECTOR4 vec;
    switch(lane){
        case 0:
            vec.vector4_f32[0] = V.vector4_f32[0];
            vec.vector4_f32[1] = 0.0f;
            vec.vector4_f32[2] = 0.0f;
            vec.vector4_f32[3] = 0.0f;
            break;
        case 1:
            vec.vector4_f32[0] = 0.0f;
            vec.vector4_f32[1] = V.vector4_f32[0];
            vec.vector4_f32[2] = 0.0f;
            vec.vector4_f32[3] = 0.0f;
            break;
        case 2:
            vec.vector4_f32[0] = 0.0f;
            vec.vector4_f32[1] = 0.0f;
            vec.vector4_f32[2] = V.vector4_f32[0];
            vec.vector4_f32[3] = 0.0f;
            break;
        default: /* FALLTHROUGH */
        case 3:
            vec.vector4_f32[0] = 0.0f;
            vec.vector4_f32[1] = 0.0f;
            vec.vector4_f32[2] = 0.0f;
            vec.vector4_f32[3] = V.vector4_f32[0];
            break;
    }
    return vec;
}

#elif defined(DXMATH_ARM_NEON_INTRINSICS)

#define DXMATH_Vector4SelectLane(V, N) vmulq_f32((V), vsetq_lane_f32(1.0f, vdupq_n_f32(0.0f), N))

#elif defined(DXMATH_SSE_INTRINSICS)

static DXMATH_VECTOR4 DXMATH_CONSTEXPR DXMATH_CALL DXMATH_Vector4SelectLane(
    DXMATH_VECTOR4 V, unsigned lane){
    __m128i mask;
    switch(lane){
        case 0:
            mask = _mm_set_epi32(0xFFFFFFFF, 0x00000000, 0x00000000, 0x00000000);
            break;
        case 1:
            mask = _mm_set_epi32(0x00000000, 0xFFFFFFFF, 0x00000000, 0x00000000);
            break;
        case 2:
            mask = _mm_set_epi32(0x00000000, 0x00000000, 0xFFFFFFFF, 0x00000000);
            break;
        default: /* FALLTHROUGH */
        case 3:
            mask = _mm_set_epi32(0xFFFFFFFF, 0x00000000, 0x00000000, 0x00000000);
            break;
    }
    {
        const __m128 or_mask = _mm_castsi128_ps(mask);
        return _mm_and_ps(V, or_mask);
    }
}

#endif

/*****************************************************************************/
#ifdef DXMATH_NO_INTRINSICS
/* DXMATH_VectorMultipy(V1, V2) = V1 * V2 */
static DXMATH_VECTOR4 DXMATH_CONSTEXPR DXMATH_CALL DXMATH_Vector4Multiply(
    const DXMATH_VECTOR4 V1,
    const DXMATH_VECTOR4 V2){
    DXMATH_VECTOR4 vec;
    DXMATH_VEC_SOFTOP(vec, V1, V2, *);
    return vec;
}
#elif defined(DXMATH_ARM_NEON_INTRINSICS)
#define DXMATH_Vector4Multiply vmulq_f32
#elif defined(DXMATH_SSE_INTRINSICS)
#define DXMATH_Vector4Multiply _mm_mul_ps
#endif

/*****************************************************************************/
#ifdef DXMATH_NO_INTRINSICS
/* DXMATH_Vector4Divide(V1, V2) = V1 / V2 */
static DXMATH_VECTOR4 DXMATH_CONSTEXPR DXMATH_CALL DXMATH_Vector4Divide(
    const DXMATH_VECTOR4 V1,
    const DXMATH_VECTOR4 V2){
    DXMATH_VECTOR4 vec;
    DXMATH_VEC_SOFTOP(vec, V1, V2, / );
    return vec;
}
#elif defined(DXMATH_ARM_NEON_INTRINSICS)
#define DXMATH_Vector4Divide vdivq_f32
#elif defined(DXMATH_SSE_INTRINSICS)
#define DXMATH_Vector4Divide _mm_div_ps
#endif

/*****************************************************************************/
/* DXMATH_Vector4Add(V1, V2) = V1 + V2 */
#ifdef DXMATH_NO_INTRINSICS
static DXMATH_VECTOR4 DXMATH_CONSTEXPR DXMATH_CALL DXMATH_Vector4Add(
    const DXMATH_VECTOR4 V1,
    const DXMATH_VECTOR4 V2){
    DXMATH_VECTOR4 vec;
    DXMATH_VEC_SOFTOP(vec, V1, V2, +);
    return vec;
}
#elif defined(DXMATH_ARM_NEON_INTRINSICS)
#define DXMATH_Vector4Add vaddq_f32
#elif defined(DXMATH_SSE_INTRINSICS)
#define DXMATH_Vector4Add _mm_add_ps
#endif

/*****************************************************************************/
/* DXMATH_Vector4Subtract(V1, V2) = V1 - V2 */
#ifdef DXMATH_NO_INTRINSICS
static DXMATH_VECTOR4 DXMATH_CONSTEXPR DXMATH_CALL DXMATH_Vector4Subtract(
    const DXMATH_VECTOR4 V1,
    const DXMATH_VECTOR4 V2){
    DXMATH_VECTOR4 vec;
    DXMATH_VEC_SOFTOP(vec, V1, V2, -);
    return vec;
}
#elif defined(DXMATH_ARM_NEON_INTRINSICS)
#define DXMATH_Vector4Subtract vsubq_f32
#elif defined(DXMATH_SSE_INTRINSICS)
#define DXMATH_Vector4Subtract _mm_sub_ps
#endif

/*****************************************************************************/
/* DXMATH_VectorSum(V) = V.x + V.y + V.z + V.w */
static float DXMATH_CONSTEXPR DXMATH_CALL DXMATH_Vector4Sum(
    const DXMATH_VECTOR4 V){
#ifdef DXMATH_NO_INTRINSICS
    return
        V.vector4_f32[0] +
        V.vector4_f32[1] +
        V.vector4_f32[2] +
        V.vector4_f32[3];
#elif defined(DXMATH_ARM_NEON_INTRINSICS)
    #error TODO!
#elif defined(DXMATH_SSE3_INTRINSICS)
    const DXMATH_VECTOR4 vec1 = _mm_hadd_ps(V, V);
    const DXMATH_VECTOR4 vec2 = _mm_hadd_ps(vec1, vec1);
    return vec2.m128_f32[0];
#elif defined(DXMATH_SSE_INTRINSICS)
    /* Copy X to the Z position and Y to the W position */
    const DXMATH_VECTOR4 vec2 =
        DXMATH_PERMUTE_PS(V, DXMATH_MM_SHUFFLE(1,0,0,0));
    /* Add Z = X+Z; W = Y+W; */
    const DXMATH_VECTOR4 vec3 = _mm_add_ps(vec2,V);
    /* Copy W to the Z position */
    const DXMATH_VECTOR4 vec4 =
        _mm_shuffle_ps(V,vec3, DXMATH_MM_SHUFFLE(0,3,0,0));
    /* Add Z and W together and return. */
    const DXMATH_VECTOR4 vec5 = _mm_add_ps(vec3,vec4);
    return vec5.m128_f32[0];
#endif
}

/*****************************************************************************/
/* Computes the dot product. */
static float DXMATH_CONSTEXPR DXMATH_CALL DXMATH_Vector4Dot(
    const DXMATH_VECTOR4 V1, 
    const DXMATH_VECTOR4 V2){
#ifdef DXMATH_NO_INTRINSICS
    return
        V1.vector4_f32[0] * V2.vector4_f32[0] +
        V1.vector4_f32[1] * V2.vector4_f32[1] +
        V1.vector4_f32[2] * V2.vector4_f32[2] +
        V1.vector4_f32[3] * V2.vector4_f32[3];
#elif defined(DXMATH_SSE4_INTRINSICS)
    const DXMATH_VECTOR4 vec = _mm_dp_ps(V1, V2, 0xff);
    return vec.m128_f32[0];
#else
    return DXMATH_Vector4Sum(DXMATH_Vector4Multiply(V1, V2));
#endif
}

/*****************************************************************************/

#define DXMATH_Vector4LengthSq(V) DXMATH_Vector4Dot(V, V)

/*****************************************************************************/
static float DXMATH_CONSTEXPR DXMATH_CALL DXMATH_Vector4Length(
    const DXMATH_VECTOR4 V){
#if defined(DXMATH_SSE3_INTRINSICS)

    #if defined(DXMATH_SSE4_INTRINSICS)
        const DXMATH_VECTOR4 dot = _mm_dp_ps(V, V, 0xff);
    #else
        const DXMATH_VECTOR4 squared = _mm_mul_ps(V, V);
        const DXMATH_VECTOR4 sum1 = _mm_hadd_ps(squared, squared);
        const DXMATH_VECTOR4 dot = _mm_hadd_ps(sum1, sum1);
    #endif
    const DXMATH_VECTOR4 len = _mm_sqrt_ss(dot);
    return len.m128_f32[0];
#else
    return DXMATH_SQRT(DXMATH_Vector4LengthSq(V));
#endif
}

/*****************************************************************************/
/* Normalizes the vector. */
static DXMATH_VECTOR4 DXMATH_CONSTEXPR DXMATH_CALL DXMATH_Vector4Normalize(
    const DXMATH_VECTOR4 V){
#if defined(DXMATH_NO_INTRINSICS)
    const float unsafe_len = DXMATH_Vector4Length(V);
    /* Prevent divide by zero */
    const float len = (unsafe_len != 0.0f) ? (1.0f / unsafe_len) : 0.0f;
    DXMATH_VECTOR4 vec;
    vec.vector4_f32[0] = V.vector4_f32[0]*len;
    vec.vector4_f32[1] = V.vector4_f32[1]*len;
    vec.vector4_f32[2] = V.vector4_f32[2]*len;
    vec.vector4_f32[3] = V.vector4_f32[3]*len;
    return vec;

#elif defined(DXMATH_ARM_NEON_INTRINSICS)
    // Dot4
    float32x4_t vTemp = vmulq_f32( V, V );
    float32x2_t v1 = vget_low_f32( vTemp );
    float32x2_t v2 = vget_high_f32( vTemp );
    v1 = vadd_f32( v1, v2 );
    v1 = vpadd_f32( v1, v1 );
    uint32x2_t VEqualsZero = vceq_f32( v1, vdup_n_f32(0) );
    uint32x2_t VEqualsInf = vceq_f32( v1, vget_low_f32(g_XMInfinity) );
    // Reciprocal sqrt (2 iterations of Newton-Raphson)
    float32x2_t S0 = vrsqrte_f32( v1 );
    float32x2_t P0 = vmul_f32( v1, S0 );
    float32x2_t R0 = vrsqrts_f32( P0, S0 );
    float32x2_t S1 = vmul_f32( S0, R0 );
    float32x2_t P1 = vmul_f32( v1, S1 );
    float32x2_t R1 = vrsqrts_f32( P1, S1 );
    v2 = vmul_f32( S1, R1 );
    // Normalize
    XMVECTOR vResult = vmulq_f32( V, vcombine_f32(v2,v2) );
    vResult = vbslq_f32(vcombine_f32(VEqualsZero,VEqualsZero), vdupq_n_f32(0), vResult);
    return vbslq_f32(vcombine_f32(VEqualsInf,VEqualsInf), g_XMQNaN, vResult );
#elif defined(DXMATH_SSE_INTRINSICS)
    /* Includes SSE4 and SSE3 specific paths, as well as an SSE fallback */
    /* First calculate length squared in an efficient manner... */
    #if defined(DXMATH_SSE4_INTRINSICS)
        const DXMATH_VECTOR4 len_sq = _mm_dp_ps(V, V, 0xff);
    #else
        /* Perform the dot product on x,y,z and w */
        const DXMATH_VECTOR4 vec_sq = _mm_mul_ps(V, V);
        /* Distribute the lengths, put in a vec called len_sq */
        #if defined(DXMATH_SSE3_INTRINSICS)    
            const DXMATH_VECTOR4 len_sq_a = _mm_hadd_ps(vec_sq, vec_sq);
            const DXMATH_VECTOR4 len_sq   = _mm_hadd_ps(len_sq_a, len_sq_a);
        #else
            /* len_sq_a has z and w */
            /* Just use shuffle instead of DXMATH_PERMUTE_PS since we are
             * definitely in SSE1/2 only territory. */
            const DXMATH_VECTOR4 len_sq_a = _mm_shuffle_ps(vec_sq, vec_sq, DXMATH_MM_SHUFFLE(3,2,3,2));
            /*x+z, y+w */
            const DXMATH_VECTOR4 len_sq_b = _mm_add_ps(vec_sq, len_sq_a);
            /* x+z,x+z,x+z,y+w */
            const DXMATH_VECTOR4 len_sq_c = _mm_shuffle_ps(len_sq_b, len_sq_b, DXMATH_MM_SHUFFLE(1,0,0,0));
            /* ?,?,y+w,y+w */
            const DXMATH_VECTOR4 len_sq_d = _mm_shuffle_ps(len_sq_a, len_sq_c, DXMATH_MM_SHUFFLE(3,3,0,0));
            /* ?,?,x+z+y+w,? */
            const DXMATH_VECTOR4 len_sq_e = _mm_add_ps(len_sq_c, len_sq_d);
            /* Splat the length */
            const DXMATH_VECTOR4 len_sq = _mm_shuffle_ps(len_sq_e, len_sq_e, DXMATH_MM_SHUFFLE(2,2,2,2));
        #endif
    #endif
    
    /* Unsquare the length, return the normalization. */
    const DXMATH_VECTOR4 len = _mm_sqrt_ps(len_sq);
    /* Divide to perform the normalization */
    return _mm_div_ps(V,len);
#endif
}

/*****************************************************************************/
/* Software implementation of a 4x4, 32-bit float matrix. */
DXMATH_ALIGN_UNION(16) dxmath_matrix4x4 {
    union dxmath_vector4 r[4];
    struct {
        float
            a, b, c, d,
            e, f, g, h,
            i, j, k, l,
            m, n, o, p;
    } f;
    struct {
        uint32_t
            a, b, c, d,
            e, f, g, h,
            i, j, k, l,
            m, n, o, p;
    } i;
    float fm[4][4];
    uint32_t im[4][4];
};

#ifdef DXMATH_NO_INTRINSICS
typedef union dxmath_matrix4x4 DXMATH_MATRIX44;
#else
DXMATH_ALIGN_STRUCT(16) DXMATH_MATRIX44_struct {
    DXMATH_VECTOR4 r[4];
};
typedef struct DXMATH_MATRIX44_struct DXMATH_MATRIX44;
#endif

/*****************************************************************************/

static DXMATH_VECTOR4 DXMATH_CONSTEXPR DXMATH_CALL DXMATH_Vector4Select(
    const DXMATH_VECTOR4 V1, const DXMATH_VECTOR4 V2, const DXMATH_VECTOR4 Control)
{
#if defined(DXMATH_NO_INTRINSICS)
    DXMATH_VECTOR4 vec;
    vec.vector4_u32[0] = (V1.vector4_u32[0] & ~Control.vector4_u32[0]) | (V2.vector4_u32[0] & Control.vector4_u32[0]);
    vec.vector4_u32[1] = (V1.vector4_u32[1] & ~Control.vector4_u32[1]) | (V2.vector4_u32[1] & Control.vector4_u32[1]);
    vec.vector4_u32[2] = (V1.vector4_u32[2] & ~Control.vector4_u32[2]) | (V2.vector4_u32[2] & Control.vector4_u32[2]);
    vec.vector4_u32[3] = (V1.vector4_u32[3] & ~Control.vector4_u32[3]) | (V2.vector4_u32[3] & Control.vector4_u32[3]);
    return vec;
#elif defined(DXMATH_ARM_NEON_INTRINSICS)
    return vbslq_f32(Control, V2, V1);
#elif defined(DXMATH_SSE_INTRINSICS)
    const DXMATH_VECTOR4 vec1 = _mm_andnot_ps(Control,V1);
    const DXMATH_VECTOR4 vec2 = _mm_and_ps(V2,Control);
    const DXMATH_VECTOR4 result = _mm_or_ps(vec1,vec2);
    return result;
#endif
}

/*****************************************************************************/
/* Matrix type: Sixteen 32 bit floating point components aligned on a
 * 16 byte boundary and mapped to four hardware vector registers.
 */

static void DXMATH_CONSTEXPR DXMATH_CALL XMMatrix44IdentityOut(
    DXMATH_MATRIX44 *Out){
#ifdef DXMATH_NO_INTRINSICS
    Out->f.a = 1.0f;
    Out->f.b = Out->f.c = Out->f.d = Out->f.e = 0.0f;
    Out->f.f = 1.0f;
    Out->f.g = Out->f.h = Out->f.i = Out->f.j = 0.0f;
    Out->f.k = 1.0f;
    Out->f.l = Out->f.m = Out->f.n = Out->f.o = 0.0f;
    Out->f.p = 1.0f;
#elif defined(DXMATH_ARM_NEON_INTRINSICS)
    const DXMATH_VECTOR4 Zero = vdupq_n_f32(0);
    Out->r[0] = vsetq_lane_f32(1.0f, Zero, 0);
    Out->r[1] = vsetq_lane_f32(1.0f, Zero, 1);
    Out->r[2] = vsetq_lane_f32(1.0f, Zero, 2);
    Out->r[3] = vsetq_lane_f32(1.0f, Zero, 3);
#elif defined(DXMATH_SSE_INTRINSICS)
    const __m128 vec0 = {1.0f, 0.0f, 0.0f, 0.0f};
    const __m128 vec1 = {0.0f, 1.0f, 0.0f, 0.0f};
    const __m128 vec2 = {0.0f, 0.0f, 1.0f, 0.0f};
    const __m128 vec3 = {0.0f, 0.0f, 0.0f, 1.0f};
    Out->r[0] = vec0;
    Out->r[1] = vec1;
    Out->r[2] = vec2;
    Out->r[3] = vec3;
#endif
}

/*****************************************************************************/

static DXMATH_MATRIX44 DXMATH_CONSTEXPR DXMATH_CALL XMMatrix44Identity(){
    DXMATH_MATRIX44 mat;
    XMMatrix44IdentityOut(&mat);
    return mat;
}

/*****************************************************************************/

static void DXMATH_CONSTEXPR DXMATH_CALL XMMatrix44SetOut(
    const float a, const float b, const float c, const float d,
    const float e, const float f, const float g, const float h,
    const float i, const float j, const float k, const float l,
    const float m, const float n, const float o, const float p,
    DXMATH_MATRIX44 *Out){
#ifdef DXMATH_NO_INTRINSICS
    Out->f.a = a; Out->f.b = b; Out->f.c = c; Out->f.d = d;
    Out->f.e = e; Out->f.f = f; Out->f.g = g; Out->f.h = h;
    Out->f.i = i; Out->f.j = j; Out->f.k = k; Out->f.l = l;
    Out->f.m = m; Out->f.n = n; Out->f.o = o; Out->f.p = p;
#else
    Out->r[0] = DXMATH_Vector4Set(a, b, c, d);
    Out->r[1] = DXMATH_Vector4Set(e, f, g, h);
    Out->r[2] = DXMATH_Vector4Set(i, j, k, l);
    Out->r[3] = DXMATH_Vector4Set(m, n, o, p);
#endif
}

/*****************************************************************************/

static DXMATH_MATRIX44 DXMATH_CONSTEXPR DXMATH_CALL XMMatrix44Set(
    const float a, const float b, const float c, const float d,
    const float e, const float f, const float g, const float h,
    const float i, const float j, const float k, const float l,
    const float m, const float n, const float o, const float p){
#ifdef DXMATH_NO_INTRINSICS
    union dxmath_matrix4x4 mat;
    mat.f.a = a; mat.f.b = b; mat.f.c = c; mat.f.d = d;
    mat.f.e = e; mat.f.f = f; mat.f.g = g; mat.f.h = h;
    mat.f.i = i; mat.f.j = j; mat.f.k = k; mat.f.l = l;
    mat.f.m = m; mat.f.n = n; mat.f.o = o; mat.f.p = p;
    return mat;
#else
    DXMATH_MATRIX44 mat;
    mat.r[0] = DXMATH_Vector4Set(a, b, c, d);
    mat.r[1] = DXMATH_Vector4Set(e, f, g, h);
    mat.r[2] = DXMATH_Vector4Set(i, j, k, l);
    mat.r[3] = DXMATH_Vector4Set(m, n, o, p);
    return mat;
#endif
}

/*****************************************************************************/

static DXMATH_MATRIX44 DXMATH_CONSTEXPR DXMATH_CALL DXMATH_Matrix44TranslationOut(
    const float x, const float y, const float z, DXMATH_MATRIX44 *Out){
#ifdef DXMATH_NO_INTRINSICS
    Out->f.a = 1.0f;
    Out->f.b = 0.0f;
    Out->f.c = 0.0f;
    Out->f.d = 0.0f;
    
    Out->f.e = 0.0f;
    Out->f.f = 1.0f;
    Out->f.g = 0.0f;
    Out->f.h = 0.0f;
    
    Out->f.i = 0.0f;
    Out->f.j = 1.0f;
    Out->f.k = 1.0f;
    Out->f.l = 0.0f;
    
    Out->f.m = x;
    Out->f.n = y;
    Out->f.o = z;
    Out->f.p = 1.0f;
#else
    #if defined(DXMATH_ARM_NEON_INTRINSICS)
        const DXMATH_VECTOR4 Zero = vdupq_n_f32(0);
        Out->r[0] = vsetq_lane_f32(1.0f, Zero, 0);
        Out->r[1] = vsetq_lane_f32(1.0f, Zero, 1);
        Out->r[2] = vsetq_lane_f32(1.0f, Zero, 2);
    #else
        static const __m128 vec0 = {1.0f, 0.0f, 0.0f, 0.0f};
        static const __m128 vec1 = {0.0f, 1.0f, 0.0f, 0.0f};
        static const __m128 vec2 = {0.0f, 0.0f, 1.0f, 0.0f};
        Out->r[0] = vec0;
        Out->r[1] = vec1;
        Out->r[2] = vec2;
    #endif
    Out->r[3] = DXMATH_Vector4Set(x, y, z, 1.0f);
#endif
}

/*****************************************************************************/

static DXMATH_MATRIX44 DXMATH_CONSTEXPR DXMATH_CALL DXMATH_Matrix44Translation(
    const float x, const float y, const float z){
#ifdef DXMATH_NO_INTRINSICS
    union dxmath_matrix4x4 mat;
    DXMATH_Matrix44TranslationOut(x, y, z, &mat);
    return mat;
#else
    DXMATH_MATRIX44 mat;
    #if defined(DXMATH_ARM_NEON_INTRINSICS)
        const DXMATH_VECTOR4 Zero = vdupq_n_f32(0);
        mat.r[0] = vsetq_lane_f32(1.0f, Zero, 0);
        mat.r[1] = vsetq_lane_f32(1.0f, Zero, 1);
        mat.r[2] = vsetq_lane_f32(1.0f, Zero, 2);
    #else
        static const __m128 vec0 = {1.0f, 0.0f, 0.0f, 0.0f};
        static const __m128 vec1 = {0.0f, 1.0f, 0.0f, 0.0f};
        static const __m128 vec2 = {0.0f, 0.0f, 1.0f, 0.0f};
        mat.r[0] = vec0;
        mat.r[1] = vec1;
        mat.r[2] = vec2;
    #endif
    mat.r[3] = DXMATH_Vector4Set(x, y, z, 1.0f);
    return mat;
#endif
}

/*****************************************************************************/

static DXMATH_MATRIX44 DXMATH_CONSTEXPR DXMATH_CALL DXMATH_Matrix44TranslationFromVector4(
    const DXMATH_VECTOR4 V){
#ifdef DXMATH_NO_INTRINSICS
    union dxmath_matrix4x4 mat;
    mat.f.a = 1.0f;
    mat.f.b = 0.0f;
    mat.f.c = 0.0f;
    mat.f.d = 0.0f;
    
    mat.f.e = 0.0f;
    mat.f.f = 1.0f;
    mat.f.g = 0.0f;
    mat.f.h = 0.0f;
    
    mat.f.i = 0.0f;
    mat.f.j = 0.0f;
    mat.f.k = 0.0f;
    mat.f.l = 1.0f;
#elif defined(DXMATH_ARM_NEON_INTRINSICS)
    DXMATH_MATRIX44 mat;
    const DXMATH_VECTOR4 Zero = vdupq_n_f32(0);
    mat.r[0] = vsetq_lane_f32(1.0f, Zero, 0);
    mat.r[1] = vsetq_lane_f32(1.0f, Zero, 1);
    mat.r[2] = vsetq_lane_f32(1.0f, Zero, 2);
#else
    DXMATH_MATRIX44 mat;
        static const __m128 vec0 = {1.0f, 0.0f, 0.0f, 0.0f};
        static const __m128 vec1 = {0.0f, 1.0f, 0.0f, 0.0f};
        static const __m128 vec2 = {0.0f, 0.0f, 1.0f, 0.0f};
        mat.r[0] = vec0;
        mat.r[1] = vec1;
        mat.r[2] = vec2;
#endif
    
    mat.r[3] = V;
    
    return mat;
}


/*****************************************************************************/

static DXMATH_MATRIX44 DXMATH_CONSTEXPR DXMATH_CALL DXMATH_Matrix44Scaling(
    const float x, const float y, const float z){
#ifdef DXMATH_NO_INTRINSICS
    union dxmath_matrix4x4 mat;
    mat.f.a = x;
    mat.f.b = 0.0f;
    mat.f.c = 0.0f;
    mat.f.d = 0.0f;
    
    mat.f.e = 0.0f;
    mat.f.f = y;
    mat.f.g = 0.0f;
    mat.f.h = 0.0f;
    
    mat.f.i = 0.0f;
    mat.f.j = z;
    mat.f.k = 1.0f;
    mat.f.l = 0.0f;
    
    mat.f.m = 0.0f;
    mat.f.n = 0.0f;
    mat.f.o = 0.0f;
    mat.f.p = 1.0f;
    
    return mat;
#elif defined(DXMATh_ARM_NEON_INTRINSICS)
    const DXMATH_VECTOR4 Zero = vdupq_n_f32(0);
    DXMATH_MATRIX44 mat;
    mat.r[0] = vsetq_lane_f32(x, Zero, 0);
    mat.r[1] = vsetq_lane_f32(y, Zero, 1);
    mat.r[2] = vsetq_lane_f32(z, Zero, 2);
    mat.r[3] = vsetq_lane_f32(1.0f, Zero, 3);
    return mat;
#else
    DXMATH_MATRIX44 mat;
    mat.r[0] = DXMATH_Vector4Set(x, 0.0f, 0.0f, 0.0f);
    mat.r[1] = DXMATH_Vector4Set(0.0f, y, 0.0f, 0.0f);
    mat.r[2] = DXMATH_Vector4Set(0.0f, 0.0f, z, 0.0f);
    {
        static const __m128 vec = {0.0f, 0.0f, 0.0f, 1.0f};
        mat.r[3] = vec;
    }
    return mat;
#endif
}

/*****************************************************************************/

static DXMATH_MATRIX44 DXMATH_CONSTEXPR DXMATH_CALL DXMATH_Matrix44ScalingFromVector4(
    const DXMATH_VECTOR4 V){
#ifdef DXMATH_NO_INTRINSICS
    union dxmath_matrix4x4 mat;
    mat.f.a = V.vector4_f32[0];
    mat.f.b = 0.0f;
    mat.f.c = 0.0f;
    mat.f.d = 0.0f;
    
    mat.f.e = 0.0f;
    mat.f.f = V.vector4_f32[1];
    mat.f.g = 0.0f;
    mat.f.h = 0.0f;
    
    mat.f.i = 0.0f;
    mat.f.j = 0.0f;
    mat.f.k = V.vector4_f32[2];
    mat.f.l = 0.0f;
    
    mat.f.m = 0.0f;
    mat.f.n = 0.0f;
    mat.f.o = 0.0f;
    mat.f.p = V.vector4_f32[3];
    return mat;

#elif defined(DXMATH_ARM_NEON_INTRINSICS)
    const DXMATH_VECTOR4 Zero = vdupq_n_f32(0);
    const DXMATH_VECTOR4 row_mask0 = vsetq_lane_f32(1.0f, Zero, 0);
    const DXMATH_VECTOR4 row_mask1 = vsetq_lane_f32(1.0f, Zero, 1);
    const DXMATH_VECTOR4 row_mask2 = vsetq_lane_f32(1.0f, Zero, 2);
    const DXMATH_VECTOR4 row_mask3 = vsetq_lane_f32(1.0f, Zero, 3);
    DXMATH_MATRIX44 mat;
    mat.r[0] = vmulq_f32(V, row_mask0);
    mat.r[1] = vmulq_f32(V, row_mask1);
    mat.r[2] = vmulq_f32(V, row_mask2);
    mat.r[3] = vmulq_f32(V, row_mask3);
    return mat;

#elif defined(DXMATH_SSE_INTRINSICS)
    DXMATH_MATRIX44 mat;
    mat.r[0] = _mm_and_ps(V, _mm_castsi128_ps(_mm_set_epi32(0xFFFFFFFF, 0x00000000, 0x00000000, 0x00000000)));
    mat.r[1] = _mm_and_ps(V, _mm_castsi128_ps(_mm_set_epi32(0x00000000, 0xFFFFFFFF, 0x00000000, 0x00000000)));
    mat.r[2] = _mm_and_ps(V, _mm_castsi128_ps(_mm_set_epi32(0x00000000, 0x00000000, 0xFFFFFFFF, 0x00000000)));
    mat.r[3] = _mm_and_ps(V, _mm_castsi128_ps(_mm_set_epi32(0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF)));
    return mat;
#endif
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* DXMATH_HEADER */
