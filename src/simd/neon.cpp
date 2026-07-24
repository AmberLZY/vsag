// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "simd/int8_simd.h"
#if defined(ENABLE_NEON)
#include <arm_neon.h>

#include "simd/kernels/kernels.h"
#include "simd/traits/simd_traits_neon.h"
#endif

#include <cmath>
#include <cstdint>

#include "simd.h"

namespace vsag::neon {

float
L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (float*)pVect1v;
    auto* pVect2 = (float*)pVect2v;
    auto qty = *((uint64_t*)qty_ptr);
    return neon::FP32ComputeL2Sqr(pVect1, pVect2, qty);
}

float
InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (float*)pVect1v;
    auto* pVect2 = (float*)pVect2v;
    auto qty = *((uint64_t*)qty_ptr);
    return neon::FP32ComputeIP(pVect1, pVect2, qty);
}

float
InnerProductDistance(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return 1.0f - neon::InnerProduct(pVect1, pVect2, qty_ptr);
}

float
INT8L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (int8_t*)pVect1v;
    auto* pVect2 = (int8_t*)pVect2v;
    auto qty = *((uint64_t*)qty_ptr);

    return neon::INT8ComputeL2Sqr(pVect1, pVect2, qty);
}

float
INT8InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (int8_t*)pVect1v;
    auto* pVect2 = (int8_t*)pVect2v;
    auto qty = *((uint64_t*)qty_ptr);
    return neon::INT8ComputeIP(pVect1, pVect2, qty);
}

float
INT8InnerProductDistance(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    return -neon::INT8InnerProduct(pVect1v, pVect2v, qty_ptr);
}

#if defined(ENABLE_NEON)
__inline float32x4x3_t __attribute__((__always_inline__)) vcvt3_f32_f16(const float16x4x3_t a) {
    float32x4x3_t c;
    c.val[0] = vcvt_f32_f16(a.val[0]);
    c.val[1] = vcvt_f32_f16(a.val[1]);
    c.val[2] = vcvt_f32_f16(a.val[2]);
    return c;
}

__inline float32x4x2_t __attribute__((__always_inline__)) vcvt2_f32_f16(const float16x4x2_t a) {
    float32x4x2_t c;
    c.val[0] = vcvt_f32_f16(a.val[0]);
    c.val[1] = vcvt_f32_f16(a.val[1]);
    return c;
}

__inline float32x4x3_t __attribute__((__always_inline__)) vcvt3_f32_half(const uint16x4x3_t x) {
    float32x4x3_t c;
    c.val[0] = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(x.val[0]), 16));
    c.val[1] = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(x.val[1]), 16));
    c.val[2] = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(x.val[2]), 16));
    return c;
}

__inline float32x4x2_t __attribute__((__always_inline__)) vcvt2_f32_half(const uint16x4x2_t x) {
    float32x4x2_t c;
    c.val[0] = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(x.val[0]), 16));
    c.val[1] = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(x.val[1]), 16));
    return c;
}
__inline float32x4_t __attribute__((__always_inline__)) vcvt_f32_half(const uint16x4_t x) {
    return vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(x), 16));
}

#endif

// calculate the dist between each pq kmeans centers and corresponding pq query dim value.
void
PQDistanceFloat256(const void* single_dim_centers, float single_dim_val, void* result) {
#if defined(ENABLE_NEON)
    simd::PQDistanceFloat256Impl<simd::SimdTraits<simd::NEON_Tag>>(
        single_dim_centers, single_dim_val, result, &generic::PQDistanceFloat256);
#else
    return generic::PQDistanceFloat256(single_dim_centers, single_dim_val, result);
#endif
}

float
FP32ComputeIP(const float* query, const float* codes, uint64_t dim) {
#if defined(ENABLE_NEON)
    return simd::ComputeIPImpl<simd::SimdTraits<simd::NEON_Tag>>(
        query, codes, dim, &generic::FP32ComputeIP);
#else
    return generic::FP32ComputeIP(query, codes, dim);
#endif
}

float
FP32ComputeL2Sqr(const float* query, const float* codes, uint64_t dim) {
#if defined(ENABLE_NEON)
    return simd::ComputeL2SqrImpl<simd::SimdTraits<simd::NEON_Tag>>(
        query, codes, dim, &generic::FP32ComputeL2Sqr);
#else
    return vsag::generic::FP32ComputeL2Sqr(query, codes, dim);
#endif
}

void
FP32SparseAccumulate(float* RESTRICT dists,
                     const uint16_t* RESTRICT ids,
                     const float* RESTRICT vals,
                     float query_val,
                     uint32_t num) {
    return generic::FP32SparseAccumulate(dists, ids, vals, query_val, num);
}

void
FP32ComputeIPBatch4(const float* RESTRICT query,
                    uint64_t dim,
                    const float* RESTRICT codes1,
                    const float* RESTRICT codes2,
                    const float* RESTRICT codes3,
                    const float* RESTRICT codes4,
                    float& result1,
                    float& result2,
                    float& result3,
                    float& result4) {
#if defined(ENABLE_NEON)
    simd::ComputeBatch4Impl<simd::SimdTraits<simd::NEON_Tag>, simd::Batch4Kind::IP>(
        query,
        dim,
        codes1,
        codes2,
        codes3,
        codes4,
        result1,
        result2,
        result3,
        result4,
        &generic::FP32ComputeIPBatch4);
#else
    return generic::FP32ComputeIPBatch4(
        query, dim, codes1, codes2, codes3, codes4, result1, result2, result3, result4);
#endif
}

void
FP32ComputeL2SqrBatch4(const float* RESTRICT query,
                       uint64_t dim,
                       const float* RESTRICT codes1,
                       const float* RESTRICT codes2,
                       const float* RESTRICT codes3,
                       const float* RESTRICT codes4,
                       float& result1,
                       float& result2,
                       float& result3,
                       float& result4) {
#if defined(ENABLE_NEON)
    simd::ComputeBatch4Impl<simd::SimdTraits<simd::NEON_Tag>, simd::Batch4Kind::L2>(
        query,
        dim,
        codes1,
        codes2,
        codes3,
        codes4,
        result1,
        result2,
        result3,
        result4,
        &generic::FP32ComputeL2SqrBatch4);
#else
    return generic::FP32ComputeL2SqrBatch4(
        query, dim, codes1, codes2, codes3, codes4, result1, result2, result3, result4);
#endif
}

void
FP32ComputeIPBatch8(const float* RESTRICT query,
                    uint64_t dim,
                    const float* RESTRICT c1,
                    const float* RESTRICT c2,
                    const float* RESTRICT c3,
                    const float* RESTRICT c4,
                    const float* RESTRICT c5,
                    const float* RESTRICT c6,
                    const float* RESTRICT c7,
                    const float* RESTRICT c8,
                    float& r1,
                    float& r2,
                    float& r3,
                    float& r4,
                    float& r5,
                    float& r6,
                    float& r7,
                    float& r8) {
#if defined(ENABLE_NEON)
    constexpr uint64_t single_round = 4;
    constexpr uint64_t multi_round = 16;
    uint64_t i;

    if (dim >= multi_round) {
        float32x4_t res1 = vdupq_n_f32(0.0f);
        float32x4_t res2 = vdupq_n_f32(0.0f);
        float32x4_t res3 = vdupq_n_f32(0.0f);
        float32x4_t res4 = vdupq_n_f32(0.0f);
        float32x4_t res5 = vdupq_n_f32(0.0f);
        float32x4_t res6 = vdupq_n_f32(0.0f);
        float32x4_t res7 = vdupq_n_f32(0.0f);
        float32x4_t res8 = vdupq_n_f32(0.0f);

        for (i = 0; i <= dim - multi_round; i += multi_round) {
            __builtin_prefetch(query + i + multi_round, 0, 1);
            __builtin_prefetch(c1 + i + multi_round, 0, 3);
            __builtin_prefetch(c2 + i + multi_round, 0, 3);
            __builtin_prefetch(c3 + i + multi_round, 0, 3);
            __builtin_prefetch(c4 + i + multi_round, 0, 3);
            __builtin_prefetch(c5 + i + multi_round, 0, 3);
            __builtin_prefetch(c6 + i + multi_round, 0, 3);
            __builtin_prefetch(c7 + i + multi_round, 0, 3);
            __builtin_prefetch(c8 + i + multi_round, 0, 3);

            for (uint64_t j = 0; j < multi_round; j += single_round) {
                const float32x4_t q = vld1q_f32(query + i + j);
                float32x4_t b1 = vld1q_f32(c1 + i + j);
                float32x4_t b2 = vld1q_f32(c2 + i + j);
                float32x4_t b3 = vld1q_f32(c3 + i + j);
                float32x4_t b4 = vld1q_f32(c4 + i + j);
                float32x4_t b5 = vld1q_f32(c5 + i + j);
                float32x4_t b6 = vld1q_f32(c6 + i + j);
                float32x4_t b7 = vld1q_f32(c7 + i + j);
                float32x4_t b8 = vld1q_f32(c8 + i + j);
                res1 = vmlaq_f32(res1, b1, q);
                res2 = vmlaq_f32(res2, b2, q);
                res3 = vmlaq_f32(res3, b3, q);
                res4 = vmlaq_f32(res4, b4, q);
                res5 = vmlaq_f32(res5, b5, q);
                res6 = vmlaq_f32(res6, b6, q);
                res7 = vmlaq_f32(res7, b7, q);
                res8 = vmlaq_f32(res8, b8, q);
            }
        }

        for (; i <= dim - single_round; i += single_round) {
            const float32x4_t q = vld1q_f32(query + i);
            float32x4_t b1 = vld1q_f32(c1 + i);
            float32x4_t b2 = vld1q_f32(c2 + i);
            float32x4_t b3 = vld1q_f32(c3 + i);
            float32x4_t b4 = vld1q_f32(c4 + i);
            float32x4_t b5 = vld1q_f32(c5 + i);
            float32x4_t b6 = vld1q_f32(c6 + i);
            float32x4_t b7 = vld1q_f32(c7 + i);
            float32x4_t b8 = vld1q_f32(c8 + i);
            res1 = vmlaq_f32(res1, b1, q);
            res2 = vmlaq_f32(res2, b2, q);
            res3 = vmlaq_f32(res3, b3, q);
            res4 = vmlaq_f32(res4, b4, q);
            res5 = vmlaq_f32(res5, b5, q);
            res6 = vmlaq_f32(res6, b6, q);
            res7 = vmlaq_f32(res7, b7, q);
            res8 = vmlaq_f32(res8, b8, q);
        }

        r1 = vaddvq_f32(res1); r2 = vaddvq_f32(res2);
        r3 = vaddvq_f32(res3); r4 = vaddvq_f32(res4);
        r5 = vaddvq_f32(res5); r6 = vaddvq_f32(res6);
        r7 = vaddvq_f32(res7); r8 = vaddvq_f32(res8);
    } else if (dim >= single_round) {
        float32x4_t q = vld1q_f32(query);
        float32x4_t b1 = vld1q_f32(c1);
        float32x4_t b2 = vld1q_f32(c2);
        float32x4_t b3 = vld1q_f32(c3);
        float32x4_t b4 = vld1q_f32(c4);
        float32x4_t b5 = vld1q_f32(c5);
        float32x4_t b6 = vld1q_f32(c6);
        float32x4_t b7 = vld1q_f32(c7);
        float32x4_t b8 = vld1q_f32(c8);
        float32x4_t res1 = vmulq_f32(b1, q);
        float32x4_t res2 = vmulq_f32(b2, q);
        float32x4_t res3 = vmulq_f32(b3, q);
        float32x4_t res4 = vmulq_f32(b4, q);
        float32x4_t res5 = vmulq_f32(b5, q);
        float32x4_t res6 = vmulq_f32(b6, q);
        float32x4_t res7 = vmulq_f32(b7, q);
        float32x4_t res8 = vmulq_f32(b8, q);

        for (i = single_round; i <= dim - single_round; i += single_round) {
            q = vld1q_f32(query + i);
            b1 = vld1q_f32(c1 + i); res1 = vmlaq_f32(res1, b1, q);
            b2 = vld1q_f32(c2 + i); res2 = vmlaq_f32(res2, b2, q);
            b3 = vld1q_f32(c3 + i); res3 = vmlaq_f32(res3, b3, q);
            b4 = vld1q_f32(c4 + i); res4 = vmlaq_f32(res4, b4, q);
            b5 = vld1q_f32(c5 + i); res5 = vmlaq_f32(res5, b5, q);
            b6 = vld1q_f32(c6 + i); res6 = vmlaq_f32(res6, b6, q);
            b7 = vld1q_f32(c7 + i); res7 = vmlaq_f32(res7, b7, q);
            b8 = vld1q_f32(c8 + i); res8 = vmlaq_f32(res8, b8, q);
        }

        r1 = vaddvq_f32(res1); r2 = vaddvq_f32(res2);
        r3 = vaddvq_f32(res3); r4 = vaddvq_f32(res4);
        r5 = vaddvq_f32(res5); r6 = vaddvq_f32(res6);
        r7 = vaddvq_f32(res7); r8 = vaddvq_f32(res8);
    } else {
        r1 = 0.0f; r2 = 0.0f; r3 = 0.0f; r4 = 0.0f;
        r5 = 0.0f; r6 = 0.0f; r7 = 0.0f; r8 = 0.0f;
        i = 0;
    }

    if (i < dim) {
        float q0 = query[i] * c1[i], q1 = query[i] * c2[i];
        float q2 = query[i] * c3[i], q3 = query[i] * c4[i];
        float q4 = query[i] * c5[i], q5 = query[i] * c6[i];
        float q6 = query[i] * c7[i], q7 = query[i] * c8[i];
        for (i++; i < dim; ++i) {
            q0 += query[i] * c1[i]; q1 += query[i] * c2[i];
            q2 += query[i] * c3[i]; q3 += query[i] * c4[i];
            q4 += query[i] * c5[i]; q5 += query[i] * c6[i];
            q6 += query[i] * c7[i]; q7 += query[i] * c8[i];
        }
        r1 += q0; r2 += q1; r3 += q2; r4 += q3;
        r5 += q4; r6 += q5; r7 += q6; r8 += q7;
    }
#else
    return generic::FP32ComputeIPBatch8(
        query, dim, c1, c2, c3, c4, c5, c6, c7, c8, r1, r2, r3, r4, r5, r6, r7, r8);
#endif
}

void
FP32ComputeIPBatch16(const float* RESTRICT query,
                     uint64_t dim,
                     const float* RESTRICT c1,
                     const float* RESTRICT c2,
                     const float* RESTRICT c3,
                     const float* RESTRICT c4,
                     const float* RESTRICT c5,
                     const float* RESTRICT c6,
                     const float* RESTRICT c7,
                     const float* RESTRICT c8,
                     const float* RESTRICT c9,
                     const float* RESTRICT c10,
                     const float* RESTRICT c11,
                     const float* RESTRICT c12,
                     const float* RESTRICT c13,
                     const float* RESTRICT c14,
                     const float* RESTRICT c15,
                     const float* RESTRICT c16,
                     float& r1,
                     float& r2,
                     float& r3,
                     float& r4,
                     float& r5,
                     float& r6,
                     float& r7,
                     float& r8,
                     float& r9,
                     float& r10,
                     float& r11,
                     float& r12,
                     float& r13,
                     float& r14,
                     float& r15,
                     float& r16) {
#if defined(ENABLE_NEON)
    constexpr uint64_t single_round = 4;
    constexpr uint64_t multi_round = 16;
    uint64_t i;

    if (dim >= multi_round) {
        float32x4_t res1 = vdupq_n_f32(0.0f);
        float32x4_t res2 = vdupq_n_f32(0.0f);
        float32x4_t res3 = vdupq_n_f32(0.0f);
        float32x4_t res4 = vdupq_n_f32(0.0f);
        float32x4_t res5 = vdupq_n_f32(0.0f);
        float32x4_t res6 = vdupq_n_f32(0.0f);
        float32x4_t res7 = vdupq_n_f32(0.0f);
        float32x4_t res8 = vdupq_n_f32(0.0f);
        float32x4_t res9 = vdupq_n_f32(0.0f);
        float32x4_t res10 = vdupq_n_f32(0.0f);
        float32x4_t res11 = vdupq_n_f32(0.0f);
        float32x4_t res12 = vdupq_n_f32(0.0f);
        float32x4_t res13 = vdupq_n_f32(0.0f);
        float32x4_t res14 = vdupq_n_f32(0.0f);
        float32x4_t res15 = vdupq_n_f32(0.0f);
        float32x4_t res16 = vdupq_n_f32(0.0f);

        for (i = 0; i <= dim - multi_round; i += multi_round) {
            __builtin_prefetch(query + i + multi_round, 0, 1);
            __builtin_prefetch(c1 + i + multi_round, 0, 3);
            __builtin_prefetch(c2 + i + multi_round, 0, 3);
            __builtin_prefetch(c3 + i + multi_round, 0, 3);
            __builtin_prefetch(c4 + i + multi_round, 0, 3);
            __builtin_prefetch(c5 + i + multi_round, 0, 3);
            __builtin_prefetch(c6 + i + multi_round, 0, 3);
            __builtin_prefetch(c7 + i + multi_round, 0, 3);
            __builtin_prefetch(c8 + i + multi_round, 0, 3);
            __builtin_prefetch(c9 + i + multi_round, 0, 3);
            __builtin_prefetch(c10 + i + multi_round, 0, 3);
            __builtin_prefetch(c11 + i + multi_round, 0, 3);
            __builtin_prefetch(c12 + i + multi_round, 0, 3);
            __builtin_prefetch(c13 + i + multi_round, 0, 3);
            __builtin_prefetch(c14 + i + multi_round, 0, 3);
            __builtin_prefetch(c15 + i + multi_round, 0, 3);
            __builtin_prefetch(c16 + i + multi_round, 0, 3);

            for (uint64_t j = 0; j < multi_round; j += single_round) {
                const float32x4_t q = vld1q_f32(query + i + j);

                float32x4_t b1 = vld1q_f32(c1 + i + j);
                float32x4_t b2 = vld1q_f32(c2 + i + j);
                float32x4_t b3 = vld1q_f32(c3 + i + j);
                float32x4_t b4 = vld1q_f32(c4 + i + j);
                float32x4_t b5 = vld1q_f32(c5 + i + j);
                float32x4_t b6 = vld1q_f32(c6 + i + j);
                float32x4_t b7 = vld1q_f32(c7 + i + j);
                float32x4_t b8 = vld1q_f32(c8 + i + j);

                res1 = vmlaq_f32(res1, b1, q);
                res2 = vmlaq_f32(res2, b2, q);
                res3 = vmlaq_f32(res3, b3, q);
                res4 = vmlaq_f32(res4, b4, q);
                res5 = vmlaq_f32(res5, b5, q);
                res6 = vmlaq_f32(res6, b6, q);
                res7 = vmlaq_f32(res7, b7, q);
                res8 = vmlaq_f32(res8, b8, q);

                b1 = vld1q_f32(c9 + i + j);
                b2 = vld1q_f32(c10 + i + j);
                b3 = vld1q_f32(c11 + i + j);
                b4 = vld1q_f32(c12 + i + j);
                b5 = vld1q_f32(c13 + i + j);
                b6 = vld1q_f32(c14 + i + j);
                b7 = vld1q_f32(c15 + i + j);
                b8 = vld1q_f32(c16 + i + j);

                res9 = vmlaq_f32(res9, b1, q);
                res10 = vmlaq_f32(res10, b2, q);
                res11 = vmlaq_f32(res11, b3, q);
                res12 = vmlaq_f32(res12, b4, q);
                res13 = vmlaq_f32(res13, b5, q);
                res14 = vmlaq_f32(res14, b6, q);
                res15 = vmlaq_f32(res15, b7, q);
                res16 = vmlaq_f32(res16, b8, q);
            }
        }

        for (; i <= dim - single_round; i += single_round) {
            const float32x4_t q = vld1q_f32(query + i);

            float32x4_t b1 = vld1q_f32(c1 + i);
            float32x4_t b2 = vld1q_f32(c2 + i);
            float32x4_t b3 = vld1q_f32(c3 + i);
            float32x4_t b4 = vld1q_f32(c4 + i);
            float32x4_t b5 = vld1q_f32(c5 + i);
            float32x4_t b6 = vld1q_f32(c6 + i);
            float32x4_t b7 = vld1q_f32(c7 + i);
            float32x4_t b8 = vld1q_f32(c8 + i);

            res1 = vmlaq_f32(res1, b1, q);
            res2 = vmlaq_f32(res2, b2, q);
            res3 = vmlaq_f32(res3, b3, q);
            res4 = vmlaq_f32(res4, b4, q);
            res5 = vmlaq_f32(res5, b5, q);
            res6 = vmlaq_f32(res6, b6, q);
            res7 = vmlaq_f32(res7, b7, q);
            res8 = vmlaq_f32(res8, b8, q);

            b1 = vld1q_f32(c9 + i);
            b2 = vld1q_f32(c10 + i);
            b3 = vld1q_f32(c11 + i);
            b4 = vld1q_f32(c12 + i);
            b5 = vld1q_f32(c13 + i);
            b6 = vld1q_f32(c14 + i);
            b7 = vld1q_f32(c15 + i);
            b8 = vld1q_f32(c16 + i);

            res9 = vmlaq_f32(res9, b1, q);
            res10 = vmlaq_f32(res10, b2, q);
            res11 = vmlaq_f32(res11, b3, q);
            res12 = vmlaq_f32(res12, b4, q);
            res13 = vmlaq_f32(res13, b5, q);
            res14 = vmlaq_f32(res14, b6, q);
            res15 = vmlaq_f32(res15, b7, q);
            res16 = vmlaq_f32(res16, b8, q);
        }

        r1 = vaddvq_f32(res1);
        r2 = vaddvq_f32(res2);
        r3 = vaddvq_f32(res3);
        r4 = vaddvq_f32(res4);
        r5 = vaddvq_f32(res5);
        r6 = vaddvq_f32(res6);
        r7 = vaddvq_f32(res7);
        r8 = vaddvq_f32(res8);
        r9 = vaddvq_f32(res9);
        r10 = vaddvq_f32(res10);
        r11 = vaddvq_f32(res11);
        r12 = vaddvq_f32(res12);
        r13 = vaddvq_f32(res13);
        r14 = vaddvq_f32(res14);
        r15 = vaddvq_f32(res15);
        r16 = vaddvq_f32(res16);
    } else if (dim >= single_round) {
        float32x4_t q = vld1q_f32(query);

        float32x4_t b1 = vld1q_f32(c1);
        float32x4_t b2 = vld1q_f32(c2);
        float32x4_t b3 = vld1q_f32(c3);
        float32x4_t b4 = vld1q_f32(c4);
        float32x4_t b5 = vld1q_f32(c5);
        float32x4_t b6 = vld1q_f32(c6);
        float32x4_t b7 = vld1q_f32(c7);
        float32x4_t b8 = vld1q_f32(c8);

        float32x4_t res1 = vmulq_f32(b1, q);
        float32x4_t res2 = vmulq_f32(b2, q);
        float32x4_t res3 = vmulq_f32(b3, q);
        float32x4_t res4 = vmulq_f32(b4, q);
        float32x4_t res5 = vmulq_f32(b5, q);
        float32x4_t res6 = vmulq_f32(b6, q);
        float32x4_t res7 = vmulq_f32(b7, q);
        float32x4_t res8 = vmulq_f32(b8, q);

        b1 = vld1q_f32(c9);
        b2 = vld1q_f32(c10);
        b3 = vld1q_f32(c11);
        b4 = vld1q_f32(c12);
        b5 = vld1q_f32(c13);
        b6 = vld1q_f32(c14);
        b7 = vld1q_f32(c15);
        b8 = vld1q_f32(c16);

        float32x4_t res9 = vmulq_f32(b1, q);
        float32x4_t res10 = vmulq_f32(b2, q);
        float32x4_t res11 = vmulq_f32(b3, q);
        float32x4_t res12 = vmulq_f32(b4, q);
        float32x4_t res13 = vmulq_f32(b5, q);
        float32x4_t res14 = vmulq_f32(b6, q);
        float32x4_t res15 = vmulq_f32(b7, q);
        float32x4_t res16 = vmulq_f32(b8, q);

        for (i = single_round; i <= dim - single_round; i += single_round) {
            q = vld1q_f32(query + i);

            b1 = vld1q_f32(c1 + i);
            b2 = vld1q_f32(c2 + i);
            b3 = vld1q_f32(c3 + i);
            b4 = vld1q_f32(c4 + i);
            b5 = vld1q_f32(c5 + i);
            b6 = vld1q_f32(c6 + i);
            b7 = vld1q_f32(c7 + i);
            b8 = vld1q_f32(c8 + i);

            res1 = vmlaq_f32(res1, b1, q);
            res2 = vmlaq_f32(res2, b2, q);
            res3 = vmlaq_f32(res3, b3, q);
            res4 = vmlaq_f32(res4, b4, q);
            res5 = vmlaq_f32(res5, b5, q);
            res6 = vmlaq_f32(res6, b6, q);
            res7 = vmlaq_f32(res7, b7, q);
            res8 = vmlaq_f32(res8, b8, q);

            b1 = vld1q_f32(c9 + i);
            b2 = vld1q_f32(c10 + i);
            b3 = vld1q_f32(c11 + i);
            b4 = vld1q_f32(c12 + i);
            b5 = vld1q_f32(c13 + i);
            b6 = vld1q_f32(c14 + i);
            b7 = vld1q_f32(c15 + i);
            b8 = vld1q_f32(c16 + i);

            res9 = vmlaq_f32(res9, b1, q);
            res10 = vmlaq_f32(res10, b2, q);
            res11 = vmlaq_f32(res11, b3, q);
            res12 = vmlaq_f32(res12, b4, q);
            res13 = vmlaq_f32(res13, b5, q);
            res14 = vmlaq_f32(res14, b6, q);
            res15 = vmlaq_f32(res15, b7, q);
            res16 = vmlaq_f32(res16, b8, q);
        }

        r1 = vaddvq_f32(res1);
        r2 = vaddvq_f32(res2);
        r3 = vaddvq_f32(res3);
        r4 = vaddvq_f32(res4);
        r5 = vaddvq_f32(res5);
        r6 = vaddvq_f32(res6);
        r7 = vaddvq_f32(res7);
        r8 = vaddvq_f32(res8);
        r9 = vaddvq_f32(res9);
        r10 = vaddvq_f32(res10);
        r11 = vaddvq_f32(res11);
        r12 = vaddvq_f32(res12);
        r13 = vaddvq_f32(res13);
        r14 = vaddvq_f32(res14);
        r15 = vaddvq_f32(res15);
        r16 = vaddvq_f32(res16);
    } else {
        r1 = 0.0f;
        r2 = 0.0f;
        r3 = 0.0f;
        r4 = 0.0f;
        r5 = 0.0f;
        r6 = 0.0f;
        r7 = 0.0f;
        r8 = 0.0f;
        r9 = 0.0f;
        r10 = 0.0f;
        r11 = 0.0f;
        r12 = 0.0f;
        r13 = 0.0f;
        r14 = 0.0f;
        r15 = 0.0f;
        r16 = 0.0f;
        i = 0;
    }

    if (i < dim) {
        float q0 = query[i] * c1[i];
        float q1 = query[i] * c2[i];
        float q2 = query[i] * c3[i];
        float q3 = query[i] * c4[i];
        float q4 = query[i] * c5[i];
        float q5 = query[i] * c6[i];
        float q6 = query[i] * c7[i];
        float q7 = query[i] * c8[i];
        float q8 = query[i] * c9[i];
        float q9 = query[i] * c10[i];
        float q10 = query[i] * c11[i];
        float q11 = query[i] * c12[i];
        float q12 = query[i] * c13[i];
        float q13 = query[i] * c14[i];
        float q14 = query[i] * c15[i];
        float q15 = query[i] * c16[i];
        for (i++; i < dim; ++i) {
            q0 += query[i] * c1[i];
            q1 += query[i] * c2[i];
            q2 += query[i] * c3[i];
            q3 += query[i] * c4[i];
            q4 += query[i] * c5[i];
            q5 += query[i] * c6[i];
            q6 += query[i] * c7[i];
            q7 += query[i] * c8[i];
            q8 += query[i] * c9[i];
            q9 += query[i] * c10[i];
            q10 += query[i] * c11[i];
            q11 += query[i] * c12[i];
            q12 += query[i] * c13[i];
            q13 += query[i] * c14[i];
            q14 += query[i] * c15[i];
            q15 += query[i] * c16[i];
        }
        r1 += q0;
        r2 += q1;
        r3 += q2;
        r4 += q3;
        r5 += q4;
        r6 += q5;
        r7 += q6;
        r8 += q7;
        r9 += q8;
        r10 += q9;
        r11 += q10;
        r12 += q11;
        r13 += q12;
        r14 += q13;
        r15 += q14;
        r16 += q15;
    }
#else
    return generic::FP32ComputeIPBatch16(query,
                                         dim,
                                         c1,
                                         c2,
                                         c3,
                                         c4,
                                         c5,
                                         c6,
                                         c7,
                                         c8,
                                         c9,
                                         c10,
                                         c11,
                                         c12,
                                         c13,
                                         c14,
                                         c15,
                                         c16,
                                         r1,
                                         r2,
                                         r3,
                                         r4,
                                         r5,
                                         r6,
                                         r7,
                                         r8,
                                         r9,
                                         r10,
                                         r11,
                                         r12,
                                         r13,
                                         r14,
                                         r15,
                                         r16);
#endif
}

void
FP32ComputeL2SqrBatch8(const float* RESTRICT query,
                       uint64_t dim,
                       const float* RESTRICT c1,
                       const float* RESTRICT c2,
                       const float* RESTRICT c3,
                       const float* RESTRICT c4,
                       const float* RESTRICT c5,
                       const float* RESTRICT c6,
                       const float* RESTRICT c7,
                       const float* RESTRICT c8,
                       float& r1,
                       float& r2,
                       float& r3,
                       float& r4,
                       float& r5,
                       float& r6,
                       float& r7,
                       float& r8) {
#if defined(ENABLE_NEON)
    constexpr uint64_t single_round = 4;
    constexpr uint64_t multi_round = 16;
    uint64_t i;

    if (dim >= multi_round) {
        float32x4_t res1 = vdupq_n_f32(0.0f);
        float32x4_t res2 = vdupq_n_f32(0.0f);
        float32x4_t res3 = vdupq_n_f32(0.0f);
        float32x4_t res4 = vdupq_n_f32(0.0f);
        float32x4_t res5 = vdupq_n_f32(0.0f);
        float32x4_t res6 = vdupq_n_f32(0.0f);
        float32x4_t res7 = vdupq_n_f32(0.0f);
        float32x4_t res8 = vdupq_n_f32(0.0f);

        for (i = 0; i <= dim - multi_round; i += multi_round) {
            __builtin_prefetch(query + i + multi_round, 0, 1);
            __builtin_prefetch(c1 + i + multi_round, 0, 3);
            __builtin_prefetch(c2 + i + multi_round, 0, 3);
            __builtin_prefetch(c3 + i + multi_round, 0, 3);
            __builtin_prefetch(c4 + i + multi_round, 0, 3);
            __builtin_prefetch(c5 + i + multi_round, 0, 3);
            __builtin_prefetch(c6 + i + multi_round, 0, 3);
            __builtin_prefetch(c7 + i + multi_round, 0, 3);
            __builtin_prefetch(c8 + i + multi_round, 0, 3);

            for (uint64_t j = 0; j < multi_round; j += single_round) {
                const float32x4_t q = vld1q_f32(query + i + j);
                float32x4_t b1 = vsubq_f32(vld1q_f32(c1 + i + j), q);
                float32x4_t b2 = vsubq_f32(vld1q_f32(c2 + i + j), q);
                float32x4_t b3 = vsubq_f32(vld1q_f32(c3 + i + j), q);
                float32x4_t b4 = vsubq_f32(vld1q_f32(c4 + i + j), q);
                float32x4_t b5 = vsubq_f32(vld1q_f32(c5 + i + j), q);
                float32x4_t b6 = vsubq_f32(vld1q_f32(c6 + i + j), q);
                float32x4_t b7 = vsubq_f32(vld1q_f32(c7 + i + j), q);
                float32x4_t b8 = vsubq_f32(vld1q_f32(c8 + i + j), q);
                res1 = vmlaq_f32(res1, b1, b1);
                res2 = vmlaq_f32(res2, b2, b2);
                res3 = vmlaq_f32(res3, b3, b3);
                res4 = vmlaq_f32(res4, b4, b4);
                res5 = vmlaq_f32(res5, b5, b5);
                res6 = vmlaq_f32(res6, b6, b6);
                res7 = vmlaq_f32(res7, b7, b7);
                res8 = vmlaq_f32(res8, b8, b8);
            }
        }

        for (; i <= dim - single_round; i += single_round) {
            const float32x4_t q = vld1q_f32(query + i);
            float32x4_t b1 = vsubq_f32(vld1q_f32(c1 + i), q);
            float32x4_t b2 = vsubq_f32(vld1q_f32(c2 + i), q);
            float32x4_t b3 = vsubq_f32(vld1q_f32(c3 + i), q);
            float32x4_t b4 = vsubq_f32(vld1q_f32(c4 + i), q);
            float32x4_t b5 = vsubq_f32(vld1q_f32(c5 + i), q);
            float32x4_t b6 = vsubq_f32(vld1q_f32(c6 + i), q);
            float32x4_t b7 = vsubq_f32(vld1q_f32(c7 + i), q);
            float32x4_t b8 = vsubq_f32(vld1q_f32(c8 + i), q);
            res1 = vmlaq_f32(res1, b1, b1);
            res2 = vmlaq_f32(res2, b2, b2);
            res3 = vmlaq_f32(res3, b3, b3);
            res4 = vmlaq_f32(res4, b4, b4);
            res5 = vmlaq_f32(res5, b5, b5);
            res6 = vmlaq_f32(res6, b6, b6);
            res7 = vmlaq_f32(res7, b7, b7);
            res8 = vmlaq_f32(res8, b8, b8);
        }

        r1 = vaddvq_f32(res1); r2 = vaddvq_f32(res2);
        r3 = vaddvq_f32(res3); r4 = vaddvq_f32(res4);
        r5 = vaddvq_f32(res5); r6 = vaddvq_f32(res6);
        r7 = vaddvq_f32(res7); r8 = vaddvq_f32(res8);
    } else if (dim >= single_round) {
        float32x4_t q = vld1q_f32(query);
        float32x4_t b1 = vsubq_f32(vld1q_f32(c1), q);
        float32x4_t b2 = vsubq_f32(vld1q_f32(c2), q);
        float32x4_t b3 = vsubq_f32(vld1q_f32(c3), q);
        float32x4_t b4 = vsubq_f32(vld1q_f32(c4), q);
        float32x4_t b5 = vsubq_f32(vld1q_f32(c5), q);
        float32x4_t b6 = vsubq_f32(vld1q_f32(c6), q);
        float32x4_t b7 = vsubq_f32(vld1q_f32(c7), q);
        float32x4_t b8 = vsubq_f32(vld1q_f32(c8), q);
        float32x4_t res1 = vmulq_f32(b1, b1);
        float32x4_t res2 = vmulq_f32(b2, b2);
        float32x4_t res3 = vmulq_f32(b3, b3);
        float32x4_t res4 = vmulq_f32(b4, b4);
        float32x4_t res5 = vmulq_f32(b5, b5);
        float32x4_t res6 = vmulq_f32(b6, b6);
        float32x4_t res7 = vmulq_f32(b7, b7);
        float32x4_t res8 = vmulq_f32(b8, b8);

        for (i = single_round; i <= dim - single_round; i += single_round) {
            q = vld1q_f32(query + i);
            b1 = vsubq_f32(vld1q_f32(c1 + i), q);
            b2 = vsubq_f32(vld1q_f32(c2 + i), q);
            b3 = vsubq_f32(vld1q_f32(c3 + i), q);
            b4 = vsubq_f32(vld1q_f32(c4 + i), q);
            b5 = vsubq_f32(vld1q_f32(c5 + i), q);
            b6 = vsubq_f32(vld1q_f32(c6 + i), q);
            b7 = vsubq_f32(vld1q_f32(c7 + i), q);
            b8 = vsubq_f32(vld1q_f32(c8 + i), q);
            res1 = vmlaq_f32(res1, b1, b1);
            res2 = vmlaq_f32(res2, b2, b2);
            res3 = vmlaq_f32(res3, b3, b3);
            res4 = vmlaq_f32(res4, b4, b4);
            res5 = vmlaq_f32(res5, b5, b5);
            res6 = vmlaq_f32(res6, b6, b6);
            res7 = vmlaq_f32(res7, b7, b7);
            res8 = vmlaq_f32(res8, b8, b8);
        }

        r1 = vaddvq_f32(res1); r2 = vaddvq_f32(res2);
        r3 = vaddvq_f32(res3); r4 = vaddvq_f32(res4);
        r5 = vaddvq_f32(res5); r6 = vaddvq_f32(res6);
        r7 = vaddvq_f32(res7); r8 = vaddvq_f32(res8);
    } else {
        r1 = 0.0f; r2 = 0.0f; r3 = 0.0f; r4 = 0.0f;
        r5 = 0.0f; r6 = 0.0f; r7 = 0.0f; r8 = 0.0f;
        i = 0;
    }

    if (i < dim) {
        float q0 = c1[i] - query[i], q1 = c2[i] - query[i];
        float q2 = c3[i] - query[i], q3 = c4[i] - query[i];
        float q4 = c5[i] - query[i], q5 = c6[i] - query[i];
        float q6 = c7[i] - query[i], q7 = c8[i] - query[i];
        float d0 = q0 * q0, d1 = q1 * q1, d2 = q2 * q2, d3 = q3 * q3;
        float d4 = q4 * q4, d5 = q5 * q5, d6 = q6 * q6, d7 = q7 * q7;
        for (i++; i < dim; ++i) {
            q0 = c1[i] - query[i]; d0 += q0 * q0;
            q1 = c2[i] - query[i]; d1 += q1 * q1;
            q2 = c3[i] - query[i]; d2 += q2 * q2;
            q3 = c4[i] - query[i]; d3 += q3 * q3;
            q4 = c5[i] - query[i]; d4 += q4 * q4;
            q5 = c6[i] - query[i]; d5 += q5 * q5;
            q6 = c7[i] - query[i]; d6 += q6 * q6;
            q7 = c8[i] - query[i]; d7 += q7 * q7;
        }
        r1 += d0; r2 += d1; r3 += d2; r4 += d3;
        r5 += d4; r6 += d5; r7 += d6; r8 += d7;
    }
#else
    return generic::FP32ComputeL2SqrBatch8(
        query, dim, c1, c2, c3, c4, c5, c6, c7, c8, r1, r2, r3, r4, r5, r6, r7, r8);
#endif
}

void
FP32ComputeL2SqrBatch16(const float* RESTRICT query,
                        uint64_t dim,
                        const float* RESTRICT c1,
                        const float* RESTRICT c2,
                        const float* RESTRICT c3,
                        const float* RESTRICT c4,
                        const float* RESTRICT c5,
                        const float* RESTRICT c6,
                        const float* RESTRICT c7,
                        const float* RESTRICT c8,
                        const float* RESTRICT c9,
                        const float* RESTRICT c10,
                        const float* RESTRICT c11,
                        const float* RESTRICT c12,
                        const float* RESTRICT c13,
                        const float* RESTRICT c14,
                        const float* RESTRICT c15,
                        const float* RESTRICT c16,
                        float& r1,
                        float& r2,
                        float& r3,
                        float& r4,
                        float& r5,
                        float& r6,
                        float& r7,
                        float& r8,
                        float& r9,
                        float& r10,
                        float& r11,
                        float& r12,
                        float& r13,
                        float& r14,
                        float& r15,
                        float& r16) {
#if defined(ENABLE_NEON)
    constexpr uint64_t single_round = 4;
    constexpr uint64_t multi_round = 16;
    uint64_t i;

    if (dim >= multi_round) {
        float32x4_t res1 = vdupq_n_f32(0.0f);
        float32x4_t res2 = vdupq_n_f32(0.0f);
        float32x4_t res3 = vdupq_n_f32(0.0f);
        float32x4_t res4 = vdupq_n_f32(0.0f);
        float32x4_t res5 = vdupq_n_f32(0.0f);
        float32x4_t res6 = vdupq_n_f32(0.0f);
        float32x4_t res7 = vdupq_n_f32(0.0f);
        float32x4_t res8 = vdupq_n_f32(0.0f);
        float32x4_t res9 = vdupq_n_f32(0.0f);
        float32x4_t res10 = vdupq_n_f32(0.0f);
        float32x4_t res11 = vdupq_n_f32(0.0f);
        float32x4_t res12 = vdupq_n_f32(0.0f);
        float32x4_t res13 = vdupq_n_f32(0.0f);
        float32x4_t res14 = vdupq_n_f32(0.0f);
        float32x4_t res15 = vdupq_n_f32(0.0f);
        float32x4_t res16 = vdupq_n_f32(0.0f);

        for (i = 0; i <= dim - multi_round; i += multi_round) {
            __builtin_prefetch(query + i + multi_round, 0, 1);
            __builtin_prefetch(c1 + i + multi_round, 0, 3);
            __builtin_prefetch(c2 + i + multi_round, 0, 3);
            __builtin_prefetch(c3 + i + multi_round, 0, 3);
            __builtin_prefetch(c4 + i + multi_round, 0, 3);
            __builtin_prefetch(c5 + i + multi_round, 0, 3);
            __builtin_prefetch(c6 + i + multi_round, 0, 3);
            __builtin_prefetch(c7 + i + multi_round, 0, 3);
            __builtin_prefetch(c8 + i + multi_round, 0, 3);
            __builtin_prefetch(c9 + i + multi_round, 0, 3);
            __builtin_prefetch(c10 + i + multi_round, 0, 3);
            __builtin_prefetch(c11 + i + multi_round, 0, 3);
            __builtin_prefetch(c12 + i + multi_round, 0, 3);
            __builtin_prefetch(c13 + i + multi_round, 0, 3);
            __builtin_prefetch(c14 + i + multi_round, 0, 3);
            __builtin_prefetch(c15 + i + multi_round, 0, 3);
            __builtin_prefetch(c16 + i + multi_round, 0, 3);

            for (uint64_t j = 0; j < multi_round; j += single_round) {
                const float32x4_t q = vld1q_f32(query + i + j);

                float32x4_t b1 = vld1q_f32(c1 + i + j);
                float32x4_t b2 = vld1q_f32(c2 + i + j);
                float32x4_t b3 = vld1q_f32(c3 + i + j);
                float32x4_t b4 = vld1q_f32(c4 + i + j);
                float32x4_t b5 = vld1q_f32(c5 + i + j);
                float32x4_t b6 = vld1q_f32(c6 + i + j);
                float32x4_t b7 = vld1q_f32(c7 + i + j);
                float32x4_t b8 = vld1q_f32(c8 + i + j);

                b1 = vsubq_f32(b1, q);
                b2 = vsubq_f32(b2, q);
                b3 = vsubq_f32(b3, q);
                b4 = vsubq_f32(b4, q);
                b5 = vsubq_f32(b5, q);
                b6 = vsubq_f32(b6, q);
                b7 = vsubq_f32(b7, q);
                b8 = vsubq_f32(b8, q);

                res1 = vmlaq_f32(res1, b1, b1);
                res2 = vmlaq_f32(res2, b2, b2);
                res3 = vmlaq_f32(res3, b3, b3);
                res4 = vmlaq_f32(res4, b4, b4);
                res5 = vmlaq_f32(res5, b5, b5);
                res6 = vmlaq_f32(res6, b6, b6);
                res7 = vmlaq_f32(res7, b7, b7);
                res8 = vmlaq_f32(res8, b8, b8);

                b1 = vld1q_f32(c9 + i + j);
                b2 = vld1q_f32(c10 + i + j);
                b3 = vld1q_f32(c11 + i + j);
                b4 = vld1q_f32(c12 + i + j);
                b5 = vld1q_f32(c13 + i + j);
                b6 = vld1q_f32(c14 + i + j);
                b7 = vld1q_f32(c15 + i + j);
                b8 = vld1q_f32(c16 + i + j);

                b1 = vsubq_f32(b1, q);
                b2 = vsubq_f32(b2, q);
                b3 = vsubq_f32(b3, q);
                b4 = vsubq_f32(b4, q);
                b5 = vsubq_f32(b5, q);
                b6 = vsubq_f32(b6, q);
                b7 = vsubq_f32(b7, q);
                b8 = vsubq_f32(b8, q);

                res9 = vmlaq_f32(res9, b1, b1);
                res10 = vmlaq_f32(res10, b2, b2);
                res11 = vmlaq_f32(res11, b3, b3);
                res12 = vmlaq_f32(res12, b4, b4);
                res13 = vmlaq_f32(res13, b5, b5);
                res14 = vmlaq_f32(res14, b6, b6);
                res15 = vmlaq_f32(res15, b7, b7);
                res16 = vmlaq_f32(res16, b8, b8);
            }
        }

        for (; i <= dim - single_round; i += single_round) {
            const float32x4_t q = vld1q_f32(query + i);

            float32x4_t b1 = vld1q_f32(c1 + i);
            float32x4_t b2 = vld1q_f32(c2 + i);
            float32x4_t b3 = vld1q_f32(c3 + i);
            float32x4_t b4 = vld1q_f32(c4 + i);
            float32x4_t b5 = vld1q_f32(c5 + i);
            float32x4_t b6 = vld1q_f32(c6 + i);
            float32x4_t b7 = vld1q_f32(c7 + i);
            float32x4_t b8 = vld1q_f32(c8 + i);

            b1 = vsubq_f32(b1, q);
            b2 = vsubq_f32(b2, q);
            b3 = vsubq_f32(b3, q);
            b4 = vsubq_f32(b4, q);
            b5 = vsubq_f32(b5, q);
            b6 = vsubq_f32(b6, q);
            b7 = vsubq_f32(b7, q);
            b8 = vsubq_f32(b8, q);

            res1 = vmlaq_f32(res1, b1, b1);
            res2 = vmlaq_f32(res2, b2, b2);
            res3 = vmlaq_f32(res3, b3, b3);
            res4 = vmlaq_f32(res4, b4, b4);
            res5 = vmlaq_f32(res5, b5, b5);
            res6 = vmlaq_f32(res6, b6, b6);
            res7 = vmlaq_f32(res7, b7, b7);
            res8 = vmlaq_f32(res8, b8, b8);

            b1 = vld1q_f32(c9 + i);
            b2 = vld1q_f32(c10 + i);
            b3 = vld1q_f32(c11 + i);
            b4 = vld1q_f32(c12 + i);
            b5 = vld1q_f32(c13 + i);
            b6 = vld1q_f32(c14 + i);
            b7 = vld1q_f32(c15 + i);
            b8 = vld1q_f32(c16 + i);

            b1 = vsubq_f32(b1, q);
            b2 = vsubq_f32(b2, q);
            b3 = vsubq_f32(b3, q);
            b4 = vsubq_f32(b4, q);
            b5 = vsubq_f32(b5, q);
            b6 = vsubq_f32(b6, q);
            b7 = vsubq_f32(b7, q);
            b8 = vsubq_f32(b8, q);

            res9 = vmlaq_f32(res9, b1, b1);
            res10 = vmlaq_f32(res10, b2, b2);
            res11 = vmlaq_f32(res11, b3, b3);
            res12 = vmlaq_f32(res12, b4, b4);
            res13 = vmlaq_f32(res13, b5, b5);
            res14 = vmlaq_f32(res14, b6, b6);
            res15 = vmlaq_f32(res15, b7, b7);
            res16 = vmlaq_f32(res16, b8, b8);
        }

        r1 = vaddvq_f32(res1);
        r2 = vaddvq_f32(res2);
        r3 = vaddvq_f32(res3);
        r4 = vaddvq_f32(res4);
        r5 = vaddvq_f32(res5);
        r6 = vaddvq_f32(res6);
        r7 = vaddvq_f32(res7);
        r8 = vaddvq_f32(res8);
        r9 = vaddvq_f32(res9);
        r10 = vaddvq_f32(res10);
        r11 = vaddvq_f32(res11);
        r12 = vaddvq_f32(res12);
        r13 = vaddvq_f32(res13);
        r14 = vaddvq_f32(res14);
        r15 = vaddvq_f32(res15);
        r16 = vaddvq_f32(res16);
    } else if (dim >= single_round) {
        float32x4_t q = vld1q_f32(query);

        float32x4_t b1 = vld1q_f32(c1);
        float32x4_t b2 = vld1q_f32(c2);
        float32x4_t b3 = vld1q_f32(c3);
        float32x4_t b4 = vld1q_f32(c4);
        float32x4_t b5 = vld1q_f32(c5);
        float32x4_t b6 = vld1q_f32(c6);
        float32x4_t b7 = vld1q_f32(c7);
        float32x4_t b8 = vld1q_f32(c8);

        b1 = vsubq_f32(b1, q);
        b2 = vsubq_f32(b2, q);
        b3 = vsubq_f32(b3, q);
        b4 = vsubq_f32(b4, q);
        b5 = vsubq_f32(b5, q);
        b6 = vsubq_f32(b6, q);
        b7 = vsubq_f32(b7, q);
        b8 = vsubq_f32(b8, q);

        float32x4_t res1 = vmulq_f32(b1, b1);
        float32x4_t res2 = vmulq_f32(b2, b2);
        float32x4_t res3 = vmulq_f32(b3, b3);
        float32x4_t res4 = vmulq_f32(b4, b4);
        float32x4_t res5 = vmulq_f32(b5, b5);
        float32x4_t res6 = vmulq_f32(b6, b6);
        float32x4_t res7 = vmulq_f32(b7, b7);
        float32x4_t res8 = vmulq_f32(b8, b8);

        b1 = vld1q_f32(c9);
        b2 = vld1q_f32(c10);
        b3 = vld1q_f32(c11);
        b4 = vld1q_f32(c12);
        b5 = vld1q_f32(c13);
        b6 = vld1q_f32(c14);
        b7 = vld1q_f32(c15);
        b8 = vld1q_f32(c16);

        b1 = vsubq_f32(b1, q);
        b2 = vsubq_f32(b2, q);
        b3 = vsubq_f32(b3, q);
        b4 = vsubq_f32(b4, q);
        b5 = vsubq_f32(b5, q);
        b6 = vsubq_f32(b6, q);
        b7 = vsubq_f32(b7, q);
        b8 = vsubq_f32(b8, q);

        float32x4_t res9 = vmulq_f32(b1, b1);
        float32x4_t res10 = vmulq_f32(b2, b2);
        float32x4_t res11 = vmulq_f32(b3, b3);
        float32x4_t res12 = vmulq_f32(b4, b4);
        float32x4_t res13 = vmulq_f32(b5, b5);
        float32x4_t res14 = vmulq_f32(b6, b6);
        float32x4_t res15 = vmulq_f32(b7, b7);
        float32x4_t res16 = vmulq_f32(b8, b8);

        for (i = single_round; i <= dim - single_round; i += single_round) {
            q = vld1q_f32(query + i);

            b1 = vld1q_f32(c1 + i);
            b2 = vld1q_f32(c2 + i);
            b3 = vld1q_f32(c3 + i);
            b4 = vld1q_f32(c4 + i);
            b5 = vld1q_f32(c5 + i);
            b6 = vld1q_f32(c6 + i);
            b7 = vld1q_f32(c7 + i);
            b8 = vld1q_f32(c8 + i);

            b1 = vsubq_f32(b1, q);
            b2 = vsubq_f32(b2, q);
            b3 = vsubq_f32(b3, q);
            b4 = vsubq_f32(b4, q);
            b5 = vsubq_f32(b5, q);
            b6 = vsubq_f32(b6, q);
            b7 = vsubq_f32(b7, q);
            b8 = vsubq_f32(b8, q);

            res1 = vmlaq_f32(res1, b1, b1);
            res2 = vmlaq_f32(res2, b2, b2);
            res3 = vmlaq_f32(res3, b3, b3);
            res4 = vmlaq_f32(res4, b4, b4);
            res5 = vmlaq_f32(res5, b5, b5);
            res6 = vmlaq_f32(res6, b6, b6);
            res7 = vmlaq_f32(res7, b7, b7);
            res8 = vmlaq_f32(res8, b8, b8);

            b1 = vld1q_f32(c9 + i);
            b2 = vld1q_f32(c10 + i);
            b3 = vld1q_f32(c11 + i);
            b4 = vld1q_f32(c12 + i);
            b5 = vld1q_f32(c13 + i);
            b6 = vld1q_f32(c14 + i);
            b7 = vld1q_f32(c15 + i);
            b8 = vld1q_f32(c16 + i);

            b1 = vsubq_f32(b1, q);
            b2 = vsubq_f32(b2, q);
            b3 = vsubq_f32(b3, q);
            b4 = vsubq_f32(b4, q);
            b5 = vsubq_f32(b5, q);
            b6 = vsubq_f32(b6, q);
            b7 = vsubq_f32(b7, q);
            b8 = vsubq_f32(b8, q);

            res9 = vmlaq_f32(res9, b1, b1);
            res10 = vmlaq_f32(res10, b2, b2);
            res11 = vmlaq_f32(res11, b3, b3);
            res12 = vmlaq_f32(res12, b4, b4);
            res13 = vmlaq_f32(res13, b5, b5);
            res14 = vmlaq_f32(res14, b6, b6);
            res15 = vmlaq_f32(res15, b7, b7);
            res16 = vmlaq_f32(res16, b8, b8);
        }

        r1 = vaddvq_f32(res1);
        r2 = vaddvq_f32(res2);
        r3 = vaddvq_f32(res3);
        r4 = vaddvq_f32(res4);
        r5 = vaddvq_f32(res5);
        r6 = vaddvq_f32(res6);
        r7 = vaddvq_f32(res7);
        r8 = vaddvq_f32(res8);
        r9 = vaddvq_f32(res9);
        r10 = vaddvq_f32(res10);
        r11 = vaddvq_f32(res11);
        r12 = vaddvq_f32(res12);
        r13 = vaddvq_f32(res13);
        r14 = vaddvq_f32(res14);
        r15 = vaddvq_f32(res15);
        r16 = vaddvq_f32(res16);
    } else {
        r1 = 0.0f;
        r2 = 0.0f;
        r3 = 0.0f;
        r4 = 0.0f;
        r5 = 0.0f;
        r6 = 0.0f;
        r7 = 0.0f;
        r8 = 0.0f;
        r9 = 0.0f;
        r10 = 0.0f;
        r11 = 0.0f;
        r12 = 0.0f;
        r13 = 0.0f;
        r14 = 0.0f;
        r15 = 0.0f;
        r16 = 0.0f;
        i = 0;
    }

    if (i < dim) {
        float q0 = c1[i] - query[i];
        float q1 = c2[i] - query[i];
        float q2 = c3[i] - query[i];
        float q3 = c4[i] - query[i];
        float q4 = c5[i] - query[i];
        float q5 = c6[i] - query[i];
        float q6 = c7[i] - query[i];
        float q7 = c8[i] - query[i];
        float q8 = c9[i] - query[i];
        float q9 = c10[i] - query[i];
        float q10 = c11[i] - query[i];
        float q11 = c12[i] - query[i];
        float q12 = c13[i] - query[i];
        float q13 = c14[i] - query[i];
        float q14 = c15[i] - query[i];
        float q15 = c16[i] - query[i];
        float d0 = q0 * q0;
        float d1 = q1 * q1;
        float d2 = q2 * q2;
        float d3 = q3 * q3;
        float d4 = q4 * q4;
        float d5 = q5 * q5;
        float d6 = q6 * q6;
        float d7 = q7 * q7;
        float d8 = q8 * q8;
        float d9 = q9 * q9;
        float d10 = q10 * q10;
        float d11 = q11 * q11;
        float d12 = q12 * q12;
        float d13 = q13 * q13;
        float d14 = q14 * q14;
        float d15 = q15 * q15;
        for (i++; i < dim; ++i) {
            q0 = c1[i] - query[i];
            q1 = c2[i] - query[i];
            q2 = c3[i] - query[i];
            q3 = c4[i] - query[i];
            q4 = c5[i] - query[i];
            q5 = c6[i] - query[i];
            q6 = c7[i] - query[i];
            q7 = c8[i] - query[i];
            q8 = c9[i] - query[i];
            q9 = c10[i] - query[i];
            q10 = c11[i] - query[i];
            q11 = c12[i] - query[i];
            q12 = c13[i] - query[i];
            q13 = c14[i] - query[i];
            q14 = c15[i] - query[i];
            q15 = c16[i] - query[i];
            d0 += q0 * q0;
            d1 += q1 * q1;
            d2 += q2 * q2;
            d3 += q3 * q3;
            d4 += q4 * q4;
            d5 += q5 * q5;
            d6 += q6 * q6;
            d7 += q7 * q7;
            d8 += q8 * q8;
            d9 += q9 * q9;
            d10 += q10 * q10;
            d11 += q11 * q11;
            d12 += q12 * q12;
            d13 += q13 * q13;
            d14 += q14 * q14;
            d15 += q15 * q15;
        }
        r1 += d0;
        r2 += d1;
        r3 += d2;
        r4 += d3;
        r5 += d4;
        r6 += d5;
        r7 += d6;
        r8 += d7;
        r9 += d8;
        r10 += d9;
        r11 += d10;
        r12 += d11;
        r13 += d12;
        r14 += d13;
        r15 += d14;
        r16 += d15;
    }
#else
    return generic::FP32ComputeL2SqrBatch16(query,
                                            dim,
                                            c1,
                                            c2,
                                            c3,
                                            c4,
                                            c5,
                                            c6,
                                            c7,
                                            c8,
                                            c9,
                                            c10,
                                            c11,
                                            c12,
                                            c13,
                                            c14,
                                            c15,
                                            c16,
                                            r1,
                                            r2,
                                            r3,
                                            r4,
                                            r5,
                                            r6,
                                            r7,
                                            r8,
                                            r9,
                                            r10,
                                            r11,
                                            r12,
                                            r13,
                                            r14,
                                            r15,
                                            r16);
#endif
}

void
FP32Sub(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_NEON)
    simd::BinaryOpImpl<simd::SimdTraits<simd::NEON_Tag>, simd::BinaryOp::Sub>(
        x, y, z, dim, &generic::FP32Sub);
#else
    return generic::FP32Sub(x, y, z, dim);
#endif
}

void
FP32Add(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_NEON)
    simd::BinaryOpImpl<simd::SimdTraits<simd::NEON_Tag>, simd::BinaryOp::Add>(
        x, y, z, dim, &generic::FP32Add);
#else
    return generic::FP32Add(x, y, z, dim);
#endif
}

void
FP32Mul(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_NEON)
    simd::BinaryOpImpl<simd::SimdTraits<simd::NEON_Tag>, simd::BinaryOp::Mul>(
        x, y, z, dim, &generic::FP32Mul);
#else
    return generic::FP32Mul(x, y, z, dim);
#endif
}

void
FP32Div(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_NEON)
    simd::BinaryOpImpl<simd::SimdTraits<simd::NEON_Tag>, simd::BinaryOp::Div>(
        x, y, z, dim, &generic::FP32Div);
#else
    return generic::FP32Div(x, y, z, dim);
#endif
}

float
FP32ReduceAdd(const float* x, uint64_t dim) {
#if defined(ENABLE_NEON)
    return simd::ReduceAddImpl<simd::SimdTraits<simd::NEON_Tag>>(x, dim, &generic::FP32ReduceAdd);
#else
    return generic::FP32ReduceAdd(x, dim);
#endif
}

#if defined(ENABLE_NEON)
__inline uint16x8_t __attribute__((__always_inline__)) load_4_short(const uint16_t* data) {
    uint16_t tmp[] = {data[3], 0, data[2], 0, data[1], 0, data[0], 0};
    return vld1q_u16(tmp);
}
#endif

float
BF16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_NEON)
    return simd::HalfComputeIPImpl<simd::BF16Traits<simd::NEON_BF16_Tag>>(
        query, codes, dim, &generic::BF16ComputeIP);
#else
    return generic::BF16ComputeIP(query, codes, dim);
#endif
}

float
BF16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_NEON)
    return simd::HalfComputeL2SqrImpl<simd::BF16Traits<simd::NEON_BF16_Tag>>(
        query, codes, dim, &generic::BF16ComputeL2Sqr);
#else
    return generic::BF16ComputeL2Sqr(query, codes, dim);
#endif
}

float
FP16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_NEON)
    return simd::HalfComputeIPImpl<simd::FP16Traits<simd::NEON_FP16_Tag>>(
        query, codes, dim, &generic::FP16ComputeIP);
#else
    return generic::FP16ComputeIP(query, codes, dim);
#endif
}

float
FP16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_NEON)
    return simd::HalfComputeL2SqrImpl<simd::FP16Traits<simd::NEON_FP16_Tag>>(
        query, codes, dim, &generic::FP16ComputeL2Sqr);
#else
    return generic::FP16ComputeL2Sqr(query, codes, dim);
#endif
}

void
FP16SparseAccumulate(float* RESTRICT dists,
                     const uint16_t* RESTRICT ids,
                     const uint16_t* RESTRICT vals,
                     float query_val,
                     uint32_t num) {
    return generic::FP16SparseAccumulate(dists, ids, vals, query_val, num);
}

#if defined(ENABLE_NEON)
__inline float32x4_t __attribute__((__always_inline__)) load_4_uint8_to_float(const uint8_t* data) {
    uint32x4_t code_values = {data[0], data[1], data[2], data[3]};
    return vcvtq_f32_u32(code_values);
}

__inline void __attribute__((__always_inline__))
load_8_uint8_to_float(const uint8_t* data, float32x4_t& low, float32x4_t& high) {
    uint8x8_t code_vec = vld1_u8(data);
    uint16x8_t code_16 = vmovl_u8(code_vec);
    uint32x4_t code_32_low = vmovl_u16(vget_low_u16(code_16));
    uint32x4_t code_32_high = vmovl_u16(vget_high_u16(code_16));
    low = vcvtq_f32_u32(code_32_low);
    high = vcvtq_f32_u32(code_32_high);
}

__inline void __attribute__((__always_inline__)) load_16_uint8_to_float(
    const uint8_t* data, float32x4_t& f0, float32x4_t& f1, float32x4_t& f2, float32x4_t& f3) {
    uint8x16_t code_vec = vld1q_u8(data);
    uint16x8_t code_16_low = vmovl_u8(vget_low_u8(code_vec));
    uint16x8_t code_16_high = vmovl_u8(vget_high_u8(code_vec));

    uint32x4_t code_32_0 = vmovl_u16(vget_low_u16(code_16_low));
    uint32x4_t code_32_1 = vmovl_u16(vget_high_u16(code_16_low));
    uint32x4_t code_32_2 = vmovl_u16(vget_low_u16(code_16_high));
    uint32x4_t code_32_3 = vmovl_u16(vget_high_u16(code_16_high));

    f0 = vcvtq_f32_u32(code_32_0);
    f1 = vcvtq_f32_u32(code_32_1);
    f2 = vcvtq_f32_u32(code_32_2);
    f3 = vcvtq_f32_u32(code_32_3);
}

__inline void __attribute__((__always_inline__))
load_12_uint8_to_float(const uint8_t* data, float32x4_t& f0, float32x4_t& f1, float32x4_t& f2) {
    // Load 12 bytes
    uint8x8_t code_low = vld1_u8(data);                             // Load first 8 bytes
    uint32x4_t code_last = {data[8], data[9], data[10], data[11]};  // Load last 4 bytes

    uint16x8_t code_16 = vmovl_u8(code_low);
    uint32x4_t code_32_0 = vmovl_u16(vget_low_u16(code_16));
    uint32x4_t code_32_1 = vmovl_u16(vget_high_u16(code_16));

    f0 = vcvtq_f32_u32(code_32_0);
    f1 = vcvtq_f32_u32(code_32_1);
    f2 = vcvtq_f32_u32(code_last);
}
#endif

float
INT8ComputeL2Sqr(const int8_t* RESTRICT query, const int8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_NEON)
    return simd::Int8ComputeL2SqrImpl<simd::Int8Traits<simd::NEON_Int8_Tag>>(
        query, codes, dim, &generic::INT8ComputeL2Sqr);
#else
    return generic::INT8ComputeL2Sqr(query, codes, dim);
#endif
}

float
INT8ComputeIP(const int8_t* __restrict query, const int8_t* __restrict codes, uint64_t dim) {
#if defined(ENABLE_NEON)
    return simd::Int8ComputeIPImpl<simd::Int8Traits<simd::NEON_Int8_Tag>>(
        query, codes, dim, &generic::INT8ComputeIP);
#else
    return generic::INT8ComputeIP(query, codes, dim);
#endif
}

float
SQ8ComputeIP(const float* RESTRICT query,
             const uint8_t* RESTRICT codes,
             const float* RESTRICT lower_bound,
             const float* RESTRICT diff,
             uint64_t dim) {
#if defined(ENABLE_NEON)
    return simd::SQ8ComputeIPImpl<simd::SQ8Traits<simd::NEON_SQ8_Tag>>(
        query, codes, lower_bound, diff, dim, &generic::SQ8ComputeIP);
#else
    return generic::SQ8ComputeIP(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ8ComputeL2Sqr(const float* RESTRICT query,
                const uint8_t* RESTRICT codes,
                const float* RESTRICT lower_bound,
                const float* RESTRICT diff,
                uint64_t dim) {
#if defined(ENABLE_NEON)
    return simd::SQ8ComputeL2SqrImpl<simd::SQ8Traits<simd::NEON_SQ8_Tag>>(
        query, codes, lower_bound, diff, dim, &generic::SQ8ComputeL2Sqr);
#else
    return generic::SQ8ComputeL2Sqr(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ8ComputeCodesIP(const uint8_t* RESTRICT codes1,
                  const uint8_t* RESTRICT codes2,
                  const float* RESTRICT lower_bound,
                  const float* RESTRICT diff,
                  uint64_t dim) {
#if defined(ENABLE_NEON)
    return simd::SQ8ComputeCodesIPImpl<simd::SQ8Traits<simd::NEON_SQ8_Tag>>(
        codes1, codes2, lower_bound, diff, dim, &generic::SQ8ComputeCodesIP);
#else
    return generic::SQ8ComputeCodesIP(codes1, codes2, lower_bound, diff, dim);
#endif
}

float
SQ8ComputeCodesL2Sqr(const uint8_t* RESTRICT codes1,
                     const uint8_t* RESTRICT codes2,
                     const float* RESTRICT lower_bound,
                     const float* RESTRICT diff,
                     uint64_t dim) {
#if defined(ENABLE_NEON)
    return simd::SQ8ComputeCodesL2SqrImpl<simd::SQ8Traits<simd::NEON_SQ8_Tag>>(
        codes1, codes2, lower_bound, diff, dim, &generic::SQ8ComputeCodesL2Sqr);
#else
    return generic::SQ8ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
#endif
}

void
SQ8SparseAccumulate(float* RESTRICT dists,
                    const uint16_t* RESTRICT ids,
                    const uint8_t* RESTRICT vals,
                    float query_val,
                    uint32_t num) {
    return generic::SQ8SparseAccumulate(dists, ids, vals, query_val, num);
}

float
SQ4ComputeIP(const float* RESTRICT query,
             const uint8_t* RESTRICT codes,
             const float* RESTRICT lower_bound,
             const float* RESTRICT diff,
             uint64_t dim) {
#if defined(ENABLE_NEON)
    return simd::SQ4ComputeIPImpl<simd::SQ4Traits<simd::NEON_SQ4_Tag>>(
        query, codes, lower_bound, diff, dim, &generic::SQ4ComputeIP);
#else
    return generic::SQ4ComputeIP(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ4ComputeL2Sqr(const float* RESTRICT query,
                const uint8_t* RESTRICT codes,
                const float* RESTRICT lower_bound,
                const float* RESTRICT diff,
                uint64_t dim) {
#if defined(ENABLE_NEON)
    return simd::SQ4ComputeL2SqrImpl<simd::SQ4Traits<simd::NEON_SQ4_Tag>>(
        query, codes, lower_bound, diff, dim, &generic::SQ4ComputeL2Sqr);
#else
    return generic::SQ4ComputeL2Sqr(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ4ComputeCodesIP(const uint8_t* RESTRICT codes1,
                  const uint8_t* RESTRICT codes2,
                  const float* RESTRICT lower_bound,
                  const float* RESTRICT diff,
                  uint64_t dim) {
#if defined(ENABLE_NEON)
    return simd::SQ4ComputeCodesIPImpl<simd::SQ4Traits<simd::NEON_SQ4_Tag>>(
        codes1, codes2, lower_bound, diff, dim, &generic::SQ4ComputeCodesIP);
#else
    return generic::SQ4ComputeCodesIP(codes1, codes2, lower_bound, diff, dim);
#endif
}

float
SQ4ComputeCodesL2Sqr(const uint8_t* RESTRICT codes1,
                     const uint8_t* RESTRICT codes2,
                     const float* RESTRICT lower_bound,
                     const float* RESTRICT diff,
                     uint64_t dim) {
#if defined(ENABLE_NEON)
    return simd::SQ4ComputeCodesL2SqrImpl<simd::SQ4Traits<simd::NEON_SQ4_Tag>>(
        codes1, codes2, lower_bound, diff, dim, &generic::SQ4ComputeCodesL2Sqr);
#else
    return generic::SQ4ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
#endif
}

#if defined(ENABLE_NEON)
__inline void __attribute__((__always_inline__))
compute_part(const uint8x16_t& a_vec, const uint8x16_t& b_vec, uint32x4_t& sum) {
    uint8x8_t a_lo = vget_low_u8(a_vec);
    uint8x8_t a_hi = vget_high_u8(a_vec);
    uint8x8_t b_lo = vget_low_u8(b_vec);
    uint8x8_t b_hi = vget_high_u8(b_vec);

    uint16x8_t prod_lo = vmull_u8(a_lo, b_lo);
    uint16x8_t prod_hi = vmull_u8(a_hi, b_hi);

    uint32x4_t sum_lo = vaddl_u16(vget_low_u16(prod_lo), vget_high_u16(prod_lo));
    uint32x4_t sum_hi = vaddl_u16(vget_low_u16(prod_hi), vget_high_u16(prod_hi));

    sum = vaddq_u32(sum, sum_lo);
    sum = vaddq_u32(sum, sum_hi);
}
#endif

float
SQ4UniformComputeCodesIP(const uint8_t* RESTRICT codes1,
                         const uint8_t* RESTRICT codes2,
                         uint64_t dim) {
#if defined(ENABLE_NEON)
    if (dim == 0) {
        return 0.0f;
    }

    uint32x4_t sum = vdupq_n_u32(0);
    uint64_t d = 0;

    for (; d + 31 < dim; d += 32) {
        uint8x16_t a = vld1q_u8(codes1 + (d >> 1));
        uint8x16_t b = vld1q_u8(codes2 + (d >> 1));
        uint8x16_t mask = vdupq_n_u8(0x0f);

        uint8x16_t a_low = vandq_u8(a, mask);
        uint8x16_t a_high = vandq_u8(vshrq_n_u8(a, 4), mask);
        uint8x16_t b_low = vandq_u8(b, mask);
        uint8x16_t b_high = vandq_u8(vshrq_n_u8(b, 4), mask);

        compute_part(a_low, b_low, sum);
        compute_part(a_high, b_high, sum);
    }
    int scalar_sum =
        generic::SQ4UniformComputeCodesIP(codes1 + (d >> 1), codes2 + (d >> 1), dim - d);

    return static_cast<float>(vaddvq_u32(sum) + scalar_sum);
#else
    return generic::SQ4UniformComputeCodesIP(codes1, codes2, dim);
#endif
}

float
SQ8UniformComputeCodesIP(const uint8_t* codes1, const uint8_t* codes2, uint64_t d) {
#if defined(ENABLE_NEON)
    uint32x4_t sum_ = vdupq_n_u32(0);
    while (d >= 16) {
        uint8x16_t a = vld1q_u8(codes1);
        uint8x16_t b = vld1q_u8(codes2);

        uint16x8_t a_low = vmovl_u8(vget_low_u8(a));
        uint16x8_t a_high = vmovl_u8(vget_high_u8(a));
        uint16x8_t b_low = vmovl_u8(vget_low_u8(b));
        uint16x8_t b_high = vmovl_u8(vget_high_u8(b));

        uint32x4_t a_low_low = vmovl_u16(vget_low_u16(a_low));
        uint32x4_t a_low_high = vmovl_u16(vget_high_u16(a_low));
        uint32x4_t a_high_low = vmovl_u16(vget_low_u16(a_high));
        uint32x4_t a_high_high = vmovl_u16(vget_high_u16(a_high));

        uint32x4_t b_low_low = vmovl_u16(vget_low_u16(b_low));
        uint32x4_t b_low_high = vmovl_u16(vget_high_u16(b_low));
        uint32x4_t b_high_low = vmovl_u16(vget_low_u16(b_high));
        uint32x4_t b_high_high = vmovl_u16(vget_high_u16(b_high));

        sum_ = vaddq_u32(sum_, vmulq_u32(a_low_low, b_low_low));
        sum_ = vaddq_u32(sum_, vmulq_u32(a_low_high, b_low_high));
        sum_ = vaddq_u32(sum_, vmulq_u32(a_high_low, b_high_low));
        sum_ = vaddq_u32(sum_, vmulq_u32(a_high_high, b_high_high));

        codes1 += 16;
        codes2 += 16;
        d -= 16;
    }

    if (d >= 8) {
        uint8x8_t a = vld1_u8(codes1);
        uint8x8_t b = vld1_u8(codes2);

        uint16x8_t a_ext = vmovl_u8(a);
        uint16x8_t b_ext = vmovl_u8(b);

        uint32x4_t a_low = vmovl_u16(vget_low_u16(a_ext));
        uint32x4_t a_high = vmovl_u16(vget_high_u16(a_ext));
        uint32x4_t b_low = vmovl_u16(vget_low_u16(b_ext));
        uint32x4_t b_high = vmovl_u16(vget_high_u16(b_ext));

        sum_ = vaddq_u32(sum_, vmulq_u32(a_low, b_low));
        sum_ = vaddq_u32(sum_, vmulq_u32(a_high, b_high));

        codes1 += 8;
        codes2 += 8;
        d -= 8;
    }

    int32_t rem_sum = 0;
    for (uint64_t i = 0; i < d; ++i) {
        rem_sum += static_cast<int32_t>(codes1[i]) * static_cast<int32_t>(codes2[i]);
    }

    // accumulate the total sum
    return static_cast<float>(vaddvq_u32(sum_) + rem_sum);
#else
    return generic::SQ8UniformComputeCodesIP(codes1, codes2, d);
#endif
}

void
SQ8UniformComputeCodesIPBatch(const uint8_t* RESTRICT query,
                              const uint8_t* RESTRICT codes,
                              uint64_t dim,
                              uint64_t n_codes,
                              uint64_t code_stride,
                              float* RESTRICT out) {
    for (uint64_t i = 0; i < n_codes; ++i) {
        out[i] = neon::SQ8UniformComputeCodesIP(query, codes + i * code_stride, dim);
    }
}

#if defined(ENABLE_NEON)
__inline void __attribute__((__always_inline__)) extract_12_bits_to_mask(const uint8_t* bits,
                                                                         uint64_t bit_offset,
                                                                         uint32x4_t& mask0,
                                                                         uint32x4_t& mask1,
                                                                         uint32x4_t& mask2) {
    uint64_t byte_idx = bit_offset / 8;
    uint64_t bit_start = bit_offset % 8;

    uint32_t mask_bits;
    if (bit_start <= 4) {
        // 12 bits span at most 2 bytes
        mask_bits =
            (bits[byte_idx] >> bit_start) | ((uint32_t)bits[byte_idx + 1] << (8 - bit_start));
    } else {
        // 12 bits span 3 bytes
        mask_bits = (bits[byte_idx] >> bit_start) |
                    ((uint32_t)bits[byte_idx + 1] << (8 - bit_start)) |
                    ((uint32_t)bits[byte_idx + 2] << (16 - bit_start));
    }
    mask_bits &= 0xFFF;  // Keep only 12 bits

    // Create mask vectors for 12 elements (4+4+4)
    mask0 = (uint32x4_t){(mask_bits & 0x001) ? 0xFFFFFFFF : 0,
                         (mask_bits & 0x002) ? 0xFFFFFFFF : 0,
                         (mask_bits & 0x004) ? 0xFFFFFFFF : 0,
                         (mask_bits & 0x008) ? 0xFFFFFFFF : 0};
    mask1 = (uint32x4_t){(mask_bits & 0x010) ? 0xFFFFFFFF : 0,
                         (mask_bits & 0x020) ? 0xFFFFFFFF : 0,
                         (mask_bits & 0x040) ? 0xFFFFFFFF : 0,
                         (mask_bits & 0x080) ? 0xFFFFFFFF : 0};
    mask2 = (uint32x4_t){(mask_bits & 0x100) ? 0xFFFFFFFF : 0,
                         (mask_bits & 0x200) ? 0xFFFFFFFF : 0,
                         (mask_bits & 0x400) ? 0xFFFFFFFF : 0,
                         (mask_bits & 0x800) ? 0xFFFFFFFF : 0};
}

__inline void __attribute__((__always_inline__)) extract_8_bits_to_mask(const uint8_t* bits,
                                                                        uint64_t bit_offset,
                                                                        uint32x4_t& mask0,
                                                                        uint32x4_t& mask1) {
    uint64_t byte_idx = bit_offset / 8;
    uint64_t bit_start = bit_offset % 8;

    uint16_t mask_bits;
    if (bit_start == 0) {
        mask_bits = bits[byte_idx];
    } else {
        mask_bits =
            (bits[byte_idx] >> bit_start) | ((uint16_t)bits[byte_idx + 1] << (8 - bit_start));
    }
    mask_bits &= 0xFF;  // Keep only 8 bits

    // Create mask vectors for 8 elements (4+4)
    mask0 = (uint32x4_t){(mask_bits & 0x01) ? 0xFFFFFFFF : 0,
                         (mask_bits & 0x02) ? 0xFFFFFFFF : 0,
                         (mask_bits & 0x04) ? 0xFFFFFFFF : 0,
                         (mask_bits & 0x08) ? 0xFFFFFFFF : 0};
    mask1 = (uint32x4_t){(mask_bits & 0x10) ? 0xFFFFFFFF : 0,
                         (mask_bits & 0x20) ? 0xFFFFFFFF : 0,
                         (mask_bits & 0x40) ? 0xFFFFFFFF : 0,
                         (mask_bits & 0x80) ? 0xFFFFFFFF : 0};
}

__inline uint32x4_t __attribute__((__always_inline__))
extract_4_bits_to_mask(const uint8_t* bits, uint64_t bit_offset) {
    uint64_t byte_idx = bit_offset / 8;
    uint64_t bit_start = bit_offset % 8;

    uint8_t mask_bits;
    if (bit_start <= 4) {
        mask_bits = (bits[byte_idx] >> bit_start) & 0xF;
    } else {
        mask_bits = ((bits[byte_idx] >> bit_start) | (bits[byte_idx + 1] << (8 - bit_start))) & 0xF;
    }

    return (uint32x4_t){(mask_bits & 0x1) ? 0xFFFFFFFF : 0,
                        (mask_bits & 0x2) ? 0xFFFFFFFF : 0,
                        (mask_bits & 0x4) ? 0xFFFFFFFF : 0,
                        (mask_bits & 0x8) ? 0xFFFFFFFF : 0};
}
#endif
uint32_t
RaBitQSQ4UBinaryIP(const uint8_t* codes, const uint8_t* bits, uint64_t dim) {
#if defined(ENABLE_NEON)
    if (dim == 0)
        return 0;

    uint32_t result = 0;
    uint64_t num_bytes = (dim + 7) / 8;

    for (uint64_t bit_pos = 0; bit_pos < 4; ++bit_pos) {
        const uint8_t* codes_ptr = codes + bit_pos * num_bytes;
        uint32x4_t popcnt_sum = vdupq_n_u32(0);

        uint64_t i = 0;
        for (; i + 15 < num_bytes; i += 16) {
            uint8x16_t code_vec = vld1q_u8(codes_ptr + i);
            uint8x16_t bits_vec = vld1q_u8(bits + i);
            uint8x16_t and_vec = vandq_u8(code_vec, bits_vec);
            uint8x16_t cnt_vec = vcntq_u8(and_vec);
            uint16x8_t sum_low = vpaddlq_u8(cnt_vec);
            uint32x4_t sum_32 = vpaddlq_u16(sum_low);
            popcnt_sum = vaddq_u32(popcnt_sum, sum_32);
        }

        uint32_t bit_count = vaddvq_u32(popcnt_sum);

        for (; i < num_bytes; i++) {
            uint8_t bitwise_and = codes_ptr[i] & bits[i];
            bit_count += __builtin_popcount(bitwise_and);
        }

        result += bit_count << bit_pos;
    }

    return result;
#else
    return generic::RaBitQSQ4UBinaryIP(codes, bits, dim);
#endif
}
float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d) {
#if defined(ENABLE_NEON)
    if (dim == 0) {
        return 0.0f;
    }

    if (dim < 4) {
        return generic::RaBitQFloatBinaryIP(vector, bits, dim, inv_sqrt_d);
    }

    uint64_t d = 0;
    float32x4_t sum = vdupq_n_f32(0.0f);
    const float32x4_t inv_sqrt_d_vec =
        inv_sqrt_d < 1e-3 ? vdupq_n_f32(1.0f) : vdupq_n_f32(inv_sqrt_d);
    const float32x4_t neg_inv_sqrt_d_vec =
        inv_sqrt_d < 1e-3 ? vdupq_n_f32(0.0f) : vdupq_n_f32(-inv_sqrt_d);

    for (; d + 11 < dim; d += 12) {
        __builtin_prefetch(vector + d + 24, 0, 1);

        float32x4x3_t vec_values = vld1q_f32_x3(vector + d);

        // Extract 12 bits and create mask vectors
        uint32x4_t bit_mask0, bit_mask1, bit_mask2;
        extract_12_bits_to_mask(bits, d, bit_mask0, bit_mask1, bit_mask2);

        // Create conditional selection vectors for all 12 elements
        float32x4x3_t b_vec;
        b_vec.val[0] = vbslq_f32(bit_mask0, inv_sqrt_d_vec, neg_inv_sqrt_d_vec);
        b_vec.val[1] = vbslq_f32(bit_mask1, inv_sqrt_d_vec, neg_inv_sqrt_d_vec);
        b_vec.val[2] = vbslq_f32(bit_mask2, inv_sqrt_d_vec, neg_inv_sqrt_d_vec);

        // Fused multiply-accumulate for all 12 elements
        sum = vfmaq_f32(sum, b_vec.val[0], vec_values.val[0]);
        sum = vfmaq_f32(sum, b_vec.val[1], vec_values.val[1]);
        sum = vfmaq_f32(sum, b_vec.val[2], vec_values.val[2]);
    }

    uint64_t remaining = dim - d;

    if (remaining >= 8) {
        float32x4x2_t vec_values = vld1q_f32_x2(vector + d);

        uint32x4_t bit_mask0, bit_mask1;
        extract_8_bits_to_mask(bits, d, bit_mask0, bit_mask1);

        float32x4x2_t b_vec;
        b_vec.val[0] = vbslq_f32(bit_mask0, inv_sqrt_d_vec, neg_inv_sqrt_d_vec);
        b_vec.val[1] = vbslq_f32(bit_mask1, inv_sqrt_d_vec, neg_inv_sqrt_d_vec);

        sum = vfmaq_f32(sum, b_vec.val[0], vec_values.val[0]);
        sum = vfmaq_f32(sum, b_vec.val[1], vec_values.val[1]);
        d += 8;
        remaining -= 8;
    }

    if (remaining >= 4) {
        float32x4_t vec_values = vld1q_f32(vector + d);
        uint32x4_t bit_mask = extract_4_bits_to_mask(bits, d);
        float32x4_t b_vec = vbslq_f32(bit_mask, inv_sqrt_d_vec, neg_inv_sqrt_d_vec);
        sum = vfmaq_f32(sum, b_vec, vec_values);
        d += 4;
        remaining -= 4;
    }

    float32x4_t res_vec = vdupq_n_f32(0.0f);
    float32x4_t res_b = vdupq_n_f32(0.0f);

    if (remaining >= 3) {
        res_vec = vld1q_lane_f32(vector + d, res_vec, 2);
        uint64_t byte_idx = d / 8;
        uint64_t bit_idx = d % 8;
        bool bit_set = (bits[byte_idx] & (1 << bit_idx)) != 0;
        res_b = vsetq_lane_f32(bit_set ? inv_sqrt_d : -inv_sqrt_d, res_b, 2);
        d++;
        remaining--;
    }

    if (remaining >= 2) {
        res_vec = vld1q_lane_f32(vector + d, res_vec, 1);
        uint64_t byte_idx = d / 8;
        uint64_t bit_idx = d % 8;
        bool bit_set = (bits[byte_idx] & (1 << bit_idx)) != 0;
        res_b = vsetq_lane_f32(bit_set ? inv_sqrt_d : -inv_sqrt_d, res_b, 1);
        d++;
        remaining--;
    }

    if (remaining >= 1) {
        res_vec = vld1q_lane_f32(vector + d, res_vec, 0);
        uint64_t byte_idx = d / 8;
        uint64_t bit_idx = d % 8;
        bool bit_set = (bits[byte_idx] & (1 << bit_idx)) != 0;
        res_b = vsetq_lane_f32(bit_set ? inv_sqrt_d : -inv_sqrt_d, res_b, 0);
    }

    if (dim > d) {
        sum = vfmaq_f32(sum, res_b, res_vec);
    }

    return vaddvq_f32(sum);
#else
    return generic::RaBitQFloatBinaryIP(vector, bits, dim, inv_sqrt_d);
#endif
}

void
RaBitQFloatBinaryIPBatch4(const float* vector,
                          const uint8_t* bits1,
                          const uint8_t* bits2,
                          const uint8_t* bits3,
                          const uint8_t* bits4,
                          uint64_t dim,
                          float inv_sqrt_d,
                          float* results) {
    generic::RaBitQFloatBinaryIPBatch4(
        vector, bits1, bits2, bits3, bits4, dim, inv_sqrt_d, results);
}

void
RaBitQFloatThreeBitIPBatch4(const float* vector,
                            const uint8_t* bits1,
                            const uint8_t* bits2,
                            const uint8_t* bits3,
                            const uint8_t* bits4,
                            uint64_t dim,
                            uint32_t reorder_bits,
                            float* results) {
    generic::RaBitQFloatThreeBitIPBatch4(
        vector, bits1, bits2, bits3, bits4, dim, reorder_bits, results);
}

float
RaBitQFloatSplitCodeIP(const float* vector,
                       const uint8_t* one_bit_code,
                       const uint8_t* supplement_code,
                       uint64_t dim,
                       uint32_t supplement_bits) {
    return generic::RaBitQFloatSplitCodeIP(
        vector, one_bit_code, supplement_code, dim, supplement_bits);
}

float
RaBitQFloatSupplementCodeIP(const float* vector,
                            const uint8_t* supplement_code,
                            uint64_t dim,
                            uint32_t supplement_bits) {
    return generic::RaBitQFloatSupplementCodeIP(vector, supplement_code, dim, supplement_bits);
}

void
DivScalar(const float* from, float* to, uint64_t dim, float scalar) {
#if defined(ENABLE_NEON)
    simd::DivScalarImpl<simd::SimdTraits<simd::NEON_Tag>>(
        from, to, dim, scalar, &generic::DivScalar);
#else
    generic::DivScalar(from, to, dim, scalar);
#endif
}

float
Normalize(const float* from, float* to, uint64_t dim) {
    float norm = std::sqrt(neon::FP32ComputeIP(from, from, dim));
    neon::DivScalar(from, to, dim, norm);
    return norm;
}

#if defined(ENABLE_NEON)
__inline uint16x8_t __attribute__((__always_inline__))
shuffle_16_char(const uint8x16_t* a, const uint8x16_t* b) {
    int8x16_t tbl = vreinterpretq_s8_u8(*a);
    uint8x16_t idx = *b;
    uint8x16_t idx_masked = vandq_u8(idx, vdupq_n_u8(0x8F));  // avoid using meaningless bits

    return vreinterpretq_u16_s8(vqtbl1q_s8(tbl, idx_masked));
}
#endif

void
Prefetch(const void* data) {
#if defined(ENABLE_NEON)
    __builtin_prefetch(data, 0, 3);
#endif
};

void
PQFastScanLookUp32(const uint8_t* RESTRICT lookup_table,
                   const uint8_t* RESTRICT codes,
                   uint64_t pq_dim,
                   int32_t* RESTRICT result) {
#if defined(ENABLE_NEON)
    uint32x4_t sum[4];
    for (uint64_t i = 0; i < 4; ++i) {
        sum[i] = vdupq_n_u32(0);
    }
    const auto sign4 = vdupq_n_u8(0x0F);
    const auto sign8 = vdupq_n_u16(0xFF);

    for (uint64_t i = 0; i < pq_dim; ++i) {
        auto dict = vld1q_u8(lookup_table);
        auto code = vld1q_u8(codes);
        lookup_table += 16;
        codes += 16;

        auto code1 = vandq_u8(code, sign4);
        auto code2 = vandq_u8(vshrq_n_u8(code, 4), sign4);
        auto res1 = shuffle_16_char(&dict, &code1);
        auto res2 = shuffle_16_char(&dict, &code2);
        sum[0] = vaddq_u32(sum[0], vreinterpretq_u32_u16(vandq_u16(res1, sign8)));
        sum[1] = vaddq_u32(sum[1], vreinterpretq_u32_u16(vshrq_n_u16(res1, 8)));
        sum[2] = vaddq_u32(sum[2], vreinterpretq_u32_u16(vandq_u16(res2, sign8)));
        sum[3] = vaddq_u32(sum[3], vreinterpretq_u32_u16(vshrq_n_u16(res2, 8)));
    }
    alignas(128) uint16_t temp[8];
    for (int64_t i = 0; i < 4; ++i) {
        vst1q_u16(temp, vreinterpretq_u16_u32(sum[i]));
        for (int64_t j = 0; j < 8; j++) {
            result[i * 8 + j] += temp[j];
        }
    }
#else
    generic::PQFastScanLookUp32(lookup_table, codes, pq_dim, result);
#endif
}

void
BitAnd(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_NEON)
    simd::BitAndImpl<simd::BitTraits<simd::NEON_Bit_Tag>>(x, y, num_byte, result, &generic::BitAnd);
#else
    return generic::BitAnd(x, y, num_byte, result);
#endif
}

void
BitOr(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_NEON)
    simd::BitOrImpl<simd::BitTraits<simd::NEON_Bit_Tag>>(x, y, num_byte, result, &generic::BitOr);
#else
    return generic::BitOr(x, y, num_byte, result);
#endif
}

void
BitXor(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_NEON)
    simd::BitXorImpl<simd::BitTraits<simd::NEON_Bit_Tag>>(x, y, num_byte, result, &generic::BitXor);
#else
    return generic::BitXor(x, y, num_byte, result);
#endif
}

void
BitNot(const uint8_t* x, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_NEON)
    simd::BitNotImpl<simd::BitTraits<simd::NEON_Bit_Tag>>(x, num_byte, result, &generic::BitNot);
#else
    return generic::BitNot(x, num_byte, result);
#endif
}
void
KacsWalk(float* data, uint64_t len) {
#if defined(ENABLE_NEON)
    simd::KacsWalkImpl<simd::SimdTraits<simd::NEON_Tag>>(data, len, &generic::KacsWalk);
#else
    generic::KacsWalk(data, len);
#endif
}

void
FlipSign(const uint8_t* flip, float* data, uint64_t dim) {
#if defined(ENABLE_NEON)
    uint64_t i = 0;

    for (; i + 3 < dim; i += 4) {
        uint8_t byte_val = flip[i / 8];
        uint8_t bit_offset = i % 8;

        uint8_t four_bits = byte_val >> bit_offset;
        if (bit_offset > 4 && (i / 8 + 1) < (dim + 7) / 8) {
            four_bits |= flip[i / 8 + 1] << (8 - bit_offset);
        }

        float32x4_t vec = vld1q_f32(data + i);

        uint32x4_t sign_mask = {(four_bits & 1) ? 0x80000000 : 0,
                                (four_bits & 2) ? 0x80000000 : 0,
                                (four_bits & 4) ? 0x80000000 : 0,
                                (four_bits & 8) ? 0x80000000 : 0};

        vec = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(vec), sign_mask));
        vst1q_f32(data + i, vec);
    }

    for (; i < dim; i++) {
        bool mask = (flip[i / 8] & (1 << (i % 8))) != 0;
        if (mask) {
            data[i] = -data[i];
        }
    }
#else
    generic::FlipSign(flip, data, dim);
#endif
}

void
VecRescale(float* data, uint64_t dim, float val) {
#if defined(ENABLE_NEON)
    simd::VecRescaleImpl<simd::SimdTraits<simd::NEON_Tag>>(data, dim, val, &generic::VecRescale);
#else
    generic::VecRescale(data, dim, val);
#endif
}

void
RotateOp(float* data, int idx, int dim_, int step) {
#if defined(ENABLE_NEON)
    simd::RotateOpImpl<simd::SimdTraits<simd::NEON_Tag>>(data, idx, dim_, step);
#else
    generic::RotateOp(data, idx, dim_, step);
#endif
}

void
FHTRotate(float* data, uint64_t dim_) {
#if defined(ENABLE_NEON)
    uint64_t n = dim_;
    uint64_t step = 1;
    while (step < n) {
        if (step >= 4) {
            neon::RotateOp(data, 0, dim_, step);
        } else {
            generic::RotateOp(data, 0, dim_, step);
        }
        step *= 2;
    }
#else
    generic::FHTRotate(data, dim_);
#endif
}

float
NormalizeWithCentroid(const float* from, const float* centroid, float* to, uint64_t dim) {
#if defined(ENABLE_NEON)
    return simd::NormalizeWithCentroidImpl<simd::SimdTraits<simd::NEON_Tag>>(
        from, centroid, to, dim, &generic::NormalizeWithCentroid);
#else
    return generic::NormalizeWithCentroid(from, centroid, to, dim);
#endif
}

void
InverseNormalizeWithCentroid(
    const float* from, const float* centroid, float* to, uint64_t dim, float norm) {
#if defined(ENABLE_NEON)
    simd::InverseNormalizeWithCentroidImpl<simd::SimdTraits<simd::NEON_Tag>>(
        from, centroid, to, dim, norm, &generic::InverseNormalizeWithCentroid);
#else
    generic::InverseNormalizeWithCentroid(from, centroid, to, dim, norm);
#endif
}

}  // namespace vsag::neon
