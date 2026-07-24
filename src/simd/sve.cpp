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
#if defined(ENABLE_SVE)
#include <arm_sve.h>
#endif

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>

#include "simd.h"
constexpr auto
generate_bit_lookup_table() {
    std::array<std::array<uint8_t, 8>, 256> table{};
    for (int byte_value = 0; byte_value < 256; ++byte_value) {
        for (int bit_pos = 0; bit_pos < 8; ++bit_pos) {
            table[byte_value][bit_pos] = ((byte_value >> bit_pos) & 1) ? 1 : 0;
        }
    }
    return table;
}

static constexpr auto g_bit_lookup_table = generate_bit_lookup_table();

namespace vsag::sve {

float
L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (float*)pVect1v;
    auto* pVect2 = (float*)pVect2v;
    auto qty = *((uint64_t*)qty_ptr);
    return sve::FP32ComputeL2Sqr(pVect1, pVect2, qty);
}
float
INT8L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (int8_t*)pVect1v;
    auto* pVect2 = (int8_t*)pVect2v;
    auto qty = *((uint64_t*)qty_ptr);
    return sve::INT8ComputeL2Sqr(pVect1, pVect2, qty);
}
float
InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (float*)pVect1v;
    auto* pVect2 = (float*)pVect2v;
    auto qty = *((uint64_t*)qty_ptr);
    return sve::FP32ComputeIP(pVect1, pVect2, qty);
}

float
InnerProductDistance(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return 1.0f - sve::InnerProduct(pVect1, pVect2, qty_ptr);
}

float
INT8InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (int8_t*)pVect1v;
    auto* pVect2 = (int8_t*)pVect2v;
    auto qty = *((uint64_t*)qty_ptr);
    return sve::INT8ComputeIP(pVect1, pVect2, qty);
}

float
INT8ComputeL2Sqr(const int8_t* RESTRICT query, const int8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SVE)
    svint64_t sum = svdup_s64(0);
    uint64_t i = 0;
    const uint64_t step = svcnth();

    svbool_t predicate = svwhilelt_b16(i, dim);
    do {
        svint16_t vec_query = svld1sb_s16(predicate, query + i);
        svint16_t vec_codes = svld1sb_s16(predicate, codes + i);

        svint16_t diff = svsub_s16_x(predicate, vec_query, vec_codes);

        sum = svdot_s64(sum, diff, diff);

        i += step;
        predicate = svwhilelt_b16(i, dim);
    } while (svptest_first(svptrue_b16(), predicate));

    return static_cast<float>(svaddv_s64(svptrue_b64(), sum));
#else
    return neon::INT8ComputeL2Sqr(query, codes, dim);
#endif
}

float
INT8ComputeIP(const int8_t* __restrict query, const int8_t* __restrict codes, uint64_t dim) {
#if defined(ENABLE_SVE)
    svint32_t sum = svdup_s32(0);
    uint64_t i = 0;
    const uint64_t step = svcntb();

    svbool_t predicate = svwhilelt_b8(i, dim);
    do {
        svint8_t vec1 = svld1_s8(predicate, query + i);
        svint8_t vec2 = svld1_s8(predicate, codes + i);
        sum = svdot_s32(sum, vec1, vec2);
        i += step;
        predicate = svwhilelt_b8(i, dim);
    } while (svptest_first(svptrue_b8(), predicate));

    return static_cast<float>(svaddv_s32(svptrue_b32(), sum));
#else
    return neon::INT8ComputeIP(query, codes, dim);
#endif
}

float
INT8InnerProductDistance(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return -sve::INT8InnerProduct(pVect1, pVect2, qty_ptr);
}

void
PQDistanceFloat256(const void* single_dim_centers, float single_dim_val, void* result) {
#if defined(ENABLE_SVE)
    const auto* float_centers = (const float*)single_dim_centers;
    auto* float_result = (float*)result;
    uint64_t num_floats_per_vector = svcntw();
    svfloat32_t value = svdup_f32(single_dim_val);
    int i = 0;
    do {
        svbool_t predicate = svwhilelt_b32(i, 256);
        svfloat32_t centers = svld1_f32(predicate, float_centers + i);
        svfloat32_t results = svld1_f32(predicate, float_result + i);
        svfloat32_t diff = svsub_f32_m(predicate, centers, value);
        results = svmad_f32_m(predicate, diff, diff, results);
        svst1_f32(predicate, float_result + i, results);
        i += num_floats_per_vector;
    } while (i < 256);
#else
    neon::PQDistanceFloat256(single_dim_centers, single_dim_val, result);
#endif
}

float
FP32ComputeIP(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum = svdup_f32(0.0f);
    uint64_t i = 0;

    const uint64_t step = svcntw();

    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t query_vec = svld1_f32(predicate, query + i);
        svfloat32_t codes_vec = svld1_f32(predicate, codes + i);

        sum = svmla_f32_m(predicate, sum, query_vec, codes_vec);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::FP32ComputeIP(query, codes, dim);
#endif
}

float
FP32ComputeL2Sqr(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum = svdup_f32(0.0f);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t query_vec = svld1_f32(predicate, query + i);
        svfloat32_t codes_vec = svld1_f32(predicate, codes + i);

        svfloat32_t diff = svsub_f32_z(predicate, query_vec, codes_vec);

        sum = svmla_f32_m(predicate, sum, diff, diff);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::FP32ComputeL2Sqr(query, codes, dim);
#endif
}

void
FP32SparseAccumulate(float* RESTRICT dists,
                     const uint16_t* RESTRICT ids,
                     const float* RESTRICT vals,
                     float query_val,
                     uint32_t num) {
    return neon::FP32SparseAccumulate(dists, ids, vals, query_val, num);
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
#if defined(ENABLE_SVE)

    svfloat32_t sum1 = svdup_f32(0.0f);
    svfloat32_t sum2 = svdup_f32(0.0f);
    svfloat32_t sum3 = svdup_f32(0.0f);
    svfloat32_t sum4 = svdup_f32(0.0f);

    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t query_vec = svld1_f32(predicate, query + i);

        svfloat32_t codes1_vec = svld1_f32(predicate, codes1 + i);
        sum1 = svmla_f32_m(predicate, sum1, query_vec, codes1_vec);

        svfloat32_t codes2_vec = svld1_f32(predicate, codes2 + i);
        sum2 = svmla_f32_m(predicate, sum2, query_vec, codes2_vec);

        svfloat32_t codes3_vec = svld1_f32(predicate, codes3 + i);
        sum3 = svmla_f32_m(predicate, sum3, query_vec, codes3_vec);

        svfloat32_t codes4_vec = svld1_f32(predicate, codes4 + i);
        sum4 = svmla_f32_m(predicate, sum4, query_vec, codes4_vec);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    result1 = svaddv_f32(svptrue_b32(), sum1);
    result2 = svaddv_f32(svptrue_b32(), sum2);
    result3 = svaddv_f32(svptrue_b32(), sum3);
    result4 = svaddv_f32(svptrue_b32(), sum4);
#else
    neon::FP32ComputeIPBatch4(
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
#if defined(ENABLE_SVE)
    svfloat32_t sum1 = svdup_f32(0.0f);
    svfloat32_t sum2 = svdup_f32(0.0f);
    svfloat32_t sum3 = svdup_f32(0.0f);
    svfloat32_t sum4 = svdup_f32(0.0f);

    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t query_vec = svld1_f32(predicate, query + i);

        svfloat32_t codes1_vec = svld1_f32(predicate, codes1 + i);
        svfloat32_t diff1 = svsub_f32_z(predicate, query_vec, codes1_vec);
        sum1 = svmla_f32_m(predicate, sum1, diff1, diff1);

        svfloat32_t codes2_vec = svld1_f32(predicate, codes2 + i);
        svfloat32_t diff2 = svsub_f32_z(predicate, query_vec, codes2_vec);
        sum2 = svmla_f32_m(predicate, sum2, diff2, diff2);

        svfloat32_t codes3_vec = svld1_f32(predicate, codes3 + i);
        svfloat32_t diff3 = svsub_f32_z(predicate, query_vec, codes3_vec);
        sum3 = svmla_f32_m(predicate, sum3, diff3, diff3);

        svfloat32_t codes4_vec = svld1_f32(predicate, codes4 + i);
        svfloat32_t diff4 = svsub_f32_z(predicate, query_vec, codes4_vec);
        sum4 = svmla_f32_m(predicate, sum4, diff4, diff4);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    result1 = svaddv_f32(svptrue_b32(), sum1);
    result2 = svaddv_f32(svptrue_b32(), sum2);
    result3 = svaddv_f32(svptrue_b32(), sum3);
    result4 = svaddv_f32(svptrue_b32(), sum4);
#else
    neon::FP32ComputeL2SqrBatch4(
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
                    float& r1, float& r2, float& r3, float& r4,
                    float& r5, float& r6, float& r7, float& r8) {
#if defined(ENABLE_SVE)
    const uint64_t step = svcntw();
    constexpr uint64_t multi_round = 16;
    uint64_t i;

    if (dim >= multi_round) {
        svfloat32_t res1 = svdup_f32(0.0f);
        svfloat32_t res2 = svdup_f32(0.0f);
        svfloat32_t res3 = svdup_f32(0.0f);
        svfloat32_t res4 = svdup_f32(0.0f);
        svfloat32_t res5 = svdup_f32(0.0f);
        svfloat32_t res6 = svdup_f32(0.0f);
        svfloat32_t res7 = svdup_f32(0.0f);
        svfloat32_t res8 = svdup_f32(0.0f);

        svbool_t pg = svptrue_b32();
        for (i = 0; i <= dim - multi_round; i += multi_round) {
            // __builtin_prefetch(query + i + multi_round, 0, 1);
            // __builtin_prefetch(c1 + i + multi_round, 0, 3);
            // __builtin_prefetch(c2 + i + multi_round, 0, 3);
            // __builtin_prefetch(c3 + i + multi_round, 0, 3);
            // __builtin_prefetch(c4 + i + multi_round, 0, 3);
            // __builtin_prefetch(c5 + i + multi_round, 0, 3);
            // __builtin_prefetch(c6 + i + multi_round, 0, 3);
            // __builtin_prefetch(c7 + i + multi_round, 0, 3);
            // __builtin_prefetch(c8 + i + multi_round, 0, 3);

            for (uint64_t j = 0; j < multi_round; j += step) {
                svfloat32_t q = svld1_f32(pg, query + i + j);
                svfloat32_t b1 = svld1_f32(pg, c1 + i + j);
                svfloat32_t b2 = svld1_f32(pg, c2 + i + j);
                svfloat32_t b3 = svld1_f32(pg, c3 + i + j);
                svfloat32_t b4 = svld1_f32(pg, c4 + i + j);
                svfloat32_t b5 = svld1_f32(pg, c5 + i + j);
                svfloat32_t b6 = svld1_f32(pg, c6 + i + j);
                svfloat32_t b7 = svld1_f32(pg, c7 + i + j);
                svfloat32_t b8 = svld1_f32(pg, c8 + i + j);
                res1 = svmla_f32_m(pg, res1, b1, q);
                res2 = svmla_f32_m(pg, res2, b2, q);
                res3 = svmla_f32_m(pg, res3, b3, q);
                res4 = svmla_f32_m(pg, res4, b4, q);
                res5 = svmla_f32_m(pg, res5, b5, q);
                res6 = svmla_f32_m(pg, res6, b6, q);
                res7 = svmla_f32_m(pg, res7, b7, q);
                res8 = svmla_f32_m(pg, res8, b8, q);
            }
        }

        for (; i < dim; i += step) {
            svbool_t pg_tail = svwhilelt_b32(i, dim);
            svfloat32_t q = svld1_f32(pg_tail, query + i);
            svfloat32_t b1 = svld1_f32(pg_tail, c1 + i);
            svfloat32_t b2 = svld1_f32(pg_tail, c2 + i);
            svfloat32_t b3 = svld1_f32(pg_tail, c3 + i);
            svfloat32_t b4 = svld1_f32(pg_tail, c4 + i);
            svfloat32_t b5 = svld1_f32(pg_tail, c5 + i);
            svfloat32_t b6 = svld1_f32(pg_tail, c6 + i);
            svfloat32_t b7 = svld1_f32(pg_tail, c7 + i);
            svfloat32_t b8 = svld1_f32(pg_tail, c8 + i);
            res1 = svmla_f32_m(pg_tail, res1, b1, q);
            res2 = svmla_f32_m(pg_tail, res2, b2, q);
            res3 = svmla_f32_m(pg_tail, res3, b3, q);
            res4 = svmla_f32_m(pg_tail, res4, b4, q);
            res5 = svmla_f32_m(pg_tail, res5, b5, q);
            res6 = svmla_f32_m(pg_tail, res6, b6, q);
            res7 = svmla_f32_m(pg_tail, res7, b7, q);
            res8 = svmla_f32_m(pg_tail, res8, b8, q);
        }

        r1 = svaddv_f32(svptrue_b32(), res1);
        r2 = svaddv_f32(svptrue_b32(), res2);
        r3 = svaddv_f32(svptrue_b32(), res3);
        r4 = svaddv_f32(svptrue_b32(), res4);
        r5 = svaddv_f32(svptrue_b32(), res5);
        r6 = svaddv_f32(svptrue_b32(), res6);
        r7 = svaddv_f32(svptrue_b32(), res7);
        r8 = svaddv_f32(svptrue_b32(), res8);
    } else {
        svbool_t pg = svwhilelt_b32((uint64_t)0, dim);
        svfloat32_t q = svld1_f32(pg, query);
        svfloat32_t res1 = svmul_f32_z(pg, svld1_f32(pg, c1), q);
        svfloat32_t res2 = svmul_f32_z(pg, svld1_f32(pg, c2), q);
        svfloat32_t res3 = svmul_f32_z(pg, svld1_f32(pg, c3), q);
        svfloat32_t res4 = svmul_f32_z(pg, svld1_f32(pg, c4), q);
        svfloat32_t res5 = svmul_f32_z(pg, svld1_f32(pg, c5), q);
        svfloat32_t res6 = svmul_f32_z(pg, svld1_f32(pg, c6), q);
        svfloat32_t res7 = svmul_f32_z(pg, svld1_f32(pg, c7), q);
        svfloat32_t res8 = svmul_f32_z(pg, svld1_f32(pg, c8), q);
        for (i = step; i < dim; i += step) {
            pg = svwhilelt_b32(i, dim);
            q = svld1_f32(pg, query + i);
            res1 = svmla_f32_m(pg, res1, svld1_f32(pg, c1 + i), q);
            res2 = svmla_f32_m(pg, res2, svld1_f32(pg, c2 + i), q);
            res3 = svmla_f32_m(pg, res3, svld1_f32(pg, c3 + i), q);
            res4 = svmla_f32_m(pg, res4, svld1_f32(pg, c4 + i), q);
            res5 = svmla_f32_m(pg, res5, svld1_f32(pg, c5 + i), q);
            res6 = svmla_f32_m(pg, res6, svld1_f32(pg, c6 + i), q);
            res7 = svmla_f32_m(pg, res7, svld1_f32(pg, c7 + i), q);
            res8 = svmla_f32_m(pg, res8, svld1_f32(pg, c8 + i), q);
        }
        r1 = svaddv_f32(svptrue_b32(), res1);
        r2 = svaddv_f32(svptrue_b32(), res2);
        r3 = svaddv_f32(svptrue_b32(), res3);
        r4 = svaddv_f32(svptrue_b32(), res4);
        r5 = svaddv_f32(svptrue_b32(), res5);
        r6 = svaddv_f32(svptrue_b32(), res6);
        r7 = svaddv_f32(svptrue_b32(), res7);
        r8 = svaddv_f32(svptrue_b32(), res8);
    }
#else
    return neon::FP32ComputeIPBatch8(
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
#if defined(ENABLE_SVE)
    const uint64_t step = svcntw();
    constexpr uint64_t multi_round = 16;
    uint64_t i;

    if (dim >= multi_round) {
        svfloat32_t res1 = svdup_f32(0.0f);
        svfloat32_t res2 = svdup_f32(0.0f);
        svfloat32_t res3 = svdup_f32(0.0f);
        svfloat32_t res4 = svdup_f32(0.0f);
        svfloat32_t res5 = svdup_f32(0.0f);
        svfloat32_t res6 = svdup_f32(0.0f);
        svfloat32_t res7 = svdup_f32(0.0f);
        svfloat32_t res8 = svdup_f32(0.0f);
        svfloat32_t res9 = svdup_f32(0.0f);
        svfloat32_t res10 = svdup_f32(0.0f);
        svfloat32_t res11 = svdup_f32(0.0f);
        svfloat32_t res12 = svdup_f32(0.0f);
        svfloat32_t res13 = svdup_f32(0.0f);
        svfloat32_t res14 = svdup_f32(0.0f);
        svfloat32_t res15 = svdup_f32(0.0f);
        svfloat32_t res16 = svdup_f32(0.0f);

        svbool_t pg = svptrue_b32();
        for (i = 0; i <= dim - multi_round; i += multi_round) {
            // __builtin_prefetch(query + i + multi_round, 0, 1);
            // __builtin_prefetch(c1 + i + multi_round, 0, 3);
            // __builtin_prefetch(c2 + i + multi_round, 0, 3);
            // __builtin_prefetch(c3 + i + multi_round, 0, 3);
            // __builtin_prefetch(c4 + i + multi_round, 0, 3);
            // __builtin_prefetch(c5 + i + multi_round, 0, 3);
            // __builtin_prefetch(c6 + i + multi_round, 0, 3);
            // __builtin_prefetch(c7 + i + multi_round, 0, 3);
            // __builtin_prefetch(c8 + i + multi_round, 0, 3);
            // __builtin_prefetch(c9 + i + multi_round, 0, 3);
            // __builtin_prefetch(c10 + i + multi_round, 0, 3);
            // __builtin_prefetch(c11 + i + multi_round, 0, 3);
            // __builtin_prefetch(c12 + i + multi_round, 0, 3);
            // __builtin_prefetch(c13 + i + multi_round, 0, 3);
            // __builtin_prefetch(c14 + i + multi_round, 0, 3);
            // __builtin_prefetch(c15 + i + multi_round, 0, 3);
            // __builtin_prefetch(c16 + i + multi_round, 0, 3);

            for (uint64_t j = 0; j < multi_round; j += step) {
                svfloat32_t q = svld1_f32(pg, query + i + j);

                svfloat32_t b1 = svld1_f32(pg, c1 + i + j);
                svfloat32_t b2 = svld1_f32(pg, c2 + i + j);
                svfloat32_t b3 = svld1_f32(pg, c3 + i + j);
                svfloat32_t b4 = svld1_f32(pg, c4 + i + j);
                svfloat32_t b5 = svld1_f32(pg, c5 + i + j);
                svfloat32_t b6 = svld1_f32(pg, c6 + i + j);
                svfloat32_t b7 = svld1_f32(pg, c7 + i + j);
                svfloat32_t b8 = svld1_f32(pg, c8 + i + j);

                res1 = svmla_f32_m(pg, res1, b1, q);
                res2 = svmla_f32_m(pg, res2, b2, q);
                res3 = svmla_f32_m(pg, res3, b3, q);
                res4 = svmla_f32_m(pg, res4, b4, q);
                res5 = svmla_f32_m(pg, res5, b5, q);
                res6 = svmla_f32_m(pg, res6, b6, q);
                res7 = svmla_f32_m(pg, res7, b7, q);
                res8 = svmla_f32_m(pg, res8, b8, q);

                b1 = svld1_f32(pg, c9 + i + j);
                b2 = svld1_f32(pg, c10 + i + j);
                b3 = svld1_f32(pg, c11 + i + j);
                b4 = svld1_f32(pg, c12 + i + j);
                b5 = svld1_f32(pg, c13 + i + j);
                b6 = svld1_f32(pg, c14 + i + j);
                b7 = svld1_f32(pg, c15 + i + j);
                b8 = svld1_f32(pg, c16 + i + j);

                res9 = svmla_f32_m(pg, res9, b1, q);
                res10 = svmla_f32_m(pg, res10, b2, q);
                res11 = svmla_f32_m(pg, res11, b3, q);
                res12 = svmla_f32_m(pg, res12, b4, q);
                res13 = svmla_f32_m(pg, res13, b5, q);
                res14 = svmla_f32_m(pg, res14, b6, q);
                res15 = svmla_f32_m(pg, res15, b7, q);
                res16 = svmla_f32_m(pg, res16, b8, q);
            }
        }

        for (; i <= dim - step; i += step) {
            svfloat32_t q = svld1_f32(pg, query + i);

            svfloat32_t b1 = svld1_f32(pg, c1 + i);
            svfloat32_t b2 = svld1_f32(pg, c2 + i);
            svfloat32_t b3 = svld1_f32(pg, c3 + i);
            svfloat32_t b4 = svld1_f32(pg, c4 + i);
            svfloat32_t b5 = svld1_f32(pg, c5 + i);
            svfloat32_t b6 = svld1_f32(pg, c6 + i);
            svfloat32_t b7 = svld1_f32(pg, c7 + i);
            svfloat32_t b8 = svld1_f32(pg, c8 + i);

            res1 = svmla_f32_m(pg, res1, b1, q);
            res2 = svmla_f32_m(pg, res2, b2, q);
            res3 = svmla_f32_m(pg, res3, b3, q);
            res4 = svmla_f32_m(pg, res4, b4, q);
            res5 = svmla_f32_m(pg, res5, b5, q);
            res6 = svmla_f32_m(pg, res6, b6, q);
            res7 = svmla_f32_m(pg, res7, b7, q);
            res8 = svmla_f32_m(pg, res8, b8, q);

            b1 = svld1_f32(pg, c9 + i);
            b2 = svld1_f32(pg, c10 + i);
            b3 = svld1_f32(pg, c11 + i);
            b4 = svld1_f32(pg, c12 + i);
            b5 = svld1_f32(pg, c13 + i);
            b6 = svld1_f32(pg, c14 + i);
            b7 = svld1_f32(pg, c15 + i);
            b8 = svld1_f32(pg, c16 + i);

            res9 = svmla_f32_m(pg, res9, b1, q);
            res10 = svmla_f32_m(pg, res10, b2, q);
            res11 = svmla_f32_m(pg, res11, b3, q);
            res12 = svmla_f32_m(pg, res12, b4, q);
            res13 = svmla_f32_m(pg, res13, b5, q);
            res14 = svmla_f32_m(pg, res14, b6, q);
            res15 = svmla_f32_m(pg, res15, b7, q);
            res16 = svmla_f32_m(pg, res16, b8, q);
        }

        r1 = svaddv_f32(pg, res1);
        r2 = svaddv_f32(pg, res2);
        r3 = svaddv_f32(pg, res3);
        r4 = svaddv_f32(pg, res4);
        r5 = svaddv_f32(pg, res5);
        r6 = svaddv_f32(pg, res6);
        r7 = svaddv_f32(pg, res7);
        r8 = svaddv_f32(pg, res8);
        r9 = svaddv_f32(pg, res9);
        r10 = svaddv_f32(pg, res10);
        r11 = svaddv_f32(pg, res11);
        r12 = svaddv_f32(pg, res12);
        r13 = svaddv_f32(pg, res13);
        r14 = svaddv_f32(pg, res14);
        r15 = svaddv_f32(pg, res15);
        r16 = svaddv_f32(pg, res16);
    } else if (dim >= step) {
        svbool_t pg = svptrue_b32();
        svfloat32_t q = svld1_f32(pg, query);

        svfloat32_t b1 = svld1_f32(pg, c1);
        svfloat32_t b2 = svld1_f32(pg, c2);
        svfloat32_t b3 = svld1_f32(pg, c3);
        svfloat32_t b4 = svld1_f32(pg, c4);
        svfloat32_t b5 = svld1_f32(pg, c5);
        svfloat32_t b6 = svld1_f32(pg, c6);
        svfloat32_t b7 = svld1_f32(pg, c7);
        svfloat32_t b8 = svld1_f32(pg, c8);

        svfloat32_t res1 = svmul_f32_z(pg, b1, q);
        svfloat32_t res2 = svmul_f32_z(pg, b2, q);
        svfloat32_t res3 = svmul_f32_z(pg, b3, q);
        svfloat32_t res4 = svmul_f32_z(pg, b4, q);
        svfloat32_t res5 = svmul_f32_z(pg, b5, q);
        svfloat32_t res6 = svmul_f32_z(pg, b6, q);
        svfloat32_t res7 = svmul_f32_z(pg, b7, q);
        svfloat32_t res8 = svmul_f32_z(pg, b8, q);

        b1 = svld1_f32(pg, c9);
        b2 = svld1_f32(pg, c10);
        b3 = svld1_f32(pg, c11);
        b4 = svld1_f32(pg, c12);
        b5 = svld1_f32(pg, c13);
        b6 = svld1_f32(pg, c14);
        b7 = svld1_f32(pg, c15);
        b8 = svld1_f32(pg, c16);

        svfloat32_t res9 = svmul_f32_z(pg, b1, q);
        svfloat32_t res10 = svmul_f32_z(pg, b2, q);
        svfloat32_t res11 = svmul_f32_z(pg, b3, q);
        svfloat32_t res12 = svmul_f32_z(pg, b4, q);
        svfloat32_t res13 = svmul_f32_z(pg, b5, q);
        svfloat32_t res14 = svmul_f32_z(pg, b6, q);
        svfloat32_t res15 = svmul_f32_z(pg, b7, q);
        svfloat32_t res16 = svmul_f32_z(pg, b8, q);

        for (i = step; i <= dim - step; i += step) {
            q = svld1_f32(pg, query + i);

            b1 = svld1_f32(pg, c1 + i);
            b2 = svld1_f32(pg, c2 + i);
            b3 = svld1_f32(pg, c3 + i);
            b4 = svld1_f32(pg, c4 + i);
            b5 = svld1_f32(pg, c5 + i);
            b6 = svld1_f32(pg, c6 + i);
            b7 = svld1_f32(pg, c7 + i);
            b8 = svld1_f32(pg, c8 + i);

            res1 = svmla_f32_m(pg, res1, b1, q);
            res2 = svmla_f32_m(pg, res2, b2, q);
            res3 = svmla_f32_m(pg, res3, b3, q);
            res4 = svmla_f32_m(pg, res4, b4, q);
            res5 = svmla_f32_m(pg, res5, b5, q);
            res6 = svmla_f32_m(pg, res6, b6, q);
            res7 = svmla_f32_m(pg, res7, b7, q);
            res8 = svmla_f32_m(pg, res8, b8, q);

            b1 = svld1_f32(pg, c9 + i);
            b2 = svld1_f32(pg, c10 + i);
            b3 = svld1_f32(pg, c11 + i);
            b4 = svld1_f32(pg, c12 + i);
            b5 = svld1_f32(pg, c13 + i);
            b6 = svld1_f32(pg, c14 + i);
            b7 = svld1_f32(pg, c15 + i);
            b8 = svld1_f32(pg, c16 + i);

            res9 = svmla_f32_m(pg, res9, b1, q);
            res10 = svmla_f32_m(pg, res10, b2, q);
            res11 = svmla_f32_m(pg, res11, b3, q);
            res12 = svmla_f32_m(pg, res12, b4, q);
            res13 = svmla_f32_m(pg, res13, b5, q);
            res14 = svmla_f32_m(pg, res14, b6, q);
            res15 = svmla_f32_m(pg, res15, b7, q);
            res16 = svmla_f32_m(pg, res16, b8, q);
        }

        r1 = svaddv_f32(pg, res1);
        r2 = svaddv_f32(pg, res2);
        r3 = svaddv_f32(pg, res3);
        r4 = svaddv_f32(pg, res4);
        r5 = svaddv_f32(pg, res5);
        r6 = svaddv_f32(pg, res6);
        r7 = svaddv_f32(pg, res7);
        r8 = svaddv_f32(pg, res8);
        r9 = svaddv_f32(pg, res9);
        r10 = svaddv_f32(pg, res10);
        r11 = svaddv_f32(pg, res11);
        r12 = svaddv_f32(pg, res12);
        r13 = svaddv_f32(pg, res13);
        r14 = svaddv_f32(pg, res14);
        r15 = svaddv_f32(pg, res15);
        r16 = svaddv_f32(pg, res16);
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
    return neon::FP32ComputeIPBatch16(query,
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
                       float& r1, float& r2, float& r3, float& r4,
                       float& r5, float& r6, float& r7, float& r8) {
#if defined(ENABLE_SVE)
    const uint64_t step = svcntw();
    constexpr uint64_t multi_round = 16;
    uint64_t i;

    if (dim >= multi_round) {
        svfloat32_t res1 = svdup_f32(0.0f);
        svfloat32_t res2 = svdup_f32(0.0f);
        svfloat32_t res3 = svdup_f32(0.0f);
        svfloat32_t res4 = svdup_f32(0.0f);
        svfloat32_t res5 = svdup_f32(0.0f);
        svfloat32_t res6 = svdup_f32(0.0f);
        svfloat32_t res7 = svdup_f32(0.0f);
        svfloat32_t res8 = svdup_f32(0.0f);

        svbool_t pg = svptrue_b32();
        for (i = 0; i <= dim - multi_round; i += multi_round) {
            // __builtin_prefetch(query + i + multi_round, 0, 1);
            // __builtin_prefetch(c1 + i + multi_round, 0, 3);
            // __builtin_prefetch(c2 + i + multi_round, 0, 3);
            // __builtin_prefetch(c3 + i + multi_round, 0, 3);
            // __builtin_prefetch(c4 + i + multi_round, 0, 3);
            // __builtin_prefetch(c5 + i + multi_round, 0, 3);
            // __builtin_prefetch(c6 + i + multi_round, 0, 3);
            // __builtin_prefetch(c7 + i + multi_round, 0, 3);
            // __builtin_prefetch(c8 + i + multi_round, 0, 3);

            for (uint64_t j = 0; j < multi_round; j += step) {
                svfloat32_t q = svld1_f32(pg, query + i + j);
                svfloat32_t b1 = svsub_f32_x(pg, svld1_f32(pg, c1 + i + j), q);
                svfloat32_t b2 = svsub_f32_x(pg, svld1_f32(pg, c2 + i + j), q);
                svfloat32_t b3 = svsub_f32_x(pg, svld1_f32(pg, c3 + i + j), q);
                svfloat32_t b4 = svsub_f32_x(pg, svld1_f32(pg, c4 + i + j), q);
                svfloat32_t b5 = svsub_f32_x(pg, svld1_f32(pg, c5 + i + j), q);
                svfloat32_t b6 = svsub_f32_x(pg, svld1_f32(pg, c6 + i + j), q);
                svfloat32_t b7 = svsub_f32_x(pg, svld1_f32(pg, c7 + i + j), q);
                svfloat32_t b8 = svsub_f32_x(pg, svld1_f32(pg, c8 + i + j), q);
                res1 = svmla_f32_m(pg, res1, b1, b1);
                res2 = svmla_f32_m(pg, res2, b2, b2);
                res3 = svmla_f32_m(pg, res3, b3, b3);
                res4 = svmla_f32_m(pg, res4, b4, b4);
                res5 = svmla_f32_m(pg, res5, b5, b5);
                res6 = svmla_f32_m(pg, res6, b6, b6);
                res7 = svmla_f32_m(pg, res7, b7, b7);
                res8 = svmla_f32_m(pg, res8, b8, b8);
            }
        }

        for (; i < dim; i += step) {
            svbool_t pg_tail = svwhilelt_b32(i, dim);
            svfloat32_t q = svld1_f32(pg_tail, query + i);
            svfloat32_t b1 = svsub_f32_x(pg_tail, svld1_f32(pg_tail, c1 + i), q);
            svfloat32_t b2 = svsub_f32_x(pg_tail, svld1_f32(pg_tail, c2 + i), q);
            svfloat32_t b3 = svsub_f32_x(pg_tail, svld1_f32(pg_tail, c3 + i), q);
            svfloat32_t b4 = svsub_f32_x(pg_tail, svld1_f32(pg_tail, c4 + i), q);
            svfloat32_t b5 = svsub_f32_x(pg_tail, svld1_f32(pg_tail, c5 + i), q);
            svfloat32_t b6 = svsub_f32_x(pg_tail, svld1_f32(pg_tail, c6 + i), q);
            svfloat32_t b7 = svsub_f32_x(pg_tail, svld1_f32(pg_tail, c7 + i), q);
            svfloat32_t b8 = svsub_f32_x(pg_tail, svld1_f32(pg_tail, c8 + i), q);
            res1 = svmla_f32_m(pg_tail, res1, b1, b1);
            res2 = svmla_f32_m(pg_tail, res2, b2, b2);
            res3 = svmla_f32_m(pg_tail, res3, b3, b3);
            res4 = svmla_f32_m(pg_tail, res4, b4, b4);
            res5 = svmla_f32_m(pg_tail, res5, b5, b5);
            res6 = svmla_f32_m(pg_tail, res6, b6, b6);
            res7 = svmla_f32_m(pg_tail, res7, b7, b7);
            res8 = svmla_f32_m(pg_tail, res8, b8, b8);
        }

        r1 = svaddv_f32(svptrue_b32(), res1);
        r2 = svaddv_f32(svptrue_b32(), res2);
        r3 = svaddv_f32(svptrue_b32(), res3);
        r4 = svaddv_f32(svptrue_b32(), res4);
        r5 = svaddv_f32(svptrue_b32(), res5);
        r6 = svaddv_f32(svptrue_b32(), res6);
        r7 = svaddv_f32(svptrue_b32(), res7);
        r8 = svaddv_f32(svptrue_b32(), res8);
    } else {
        svbool_t pg = svwhilelt_b32((uint64_t)0, dim);
        svfloat32_t q = svld1_f32(pg, query);
        svfloat32_t b1 = svsub_f32_x(pg, svld1_f32(pg, c1), q);
        svfloat32_t b2 = svsub_f32_x(pg, svld1_f32(pg, c2), q);
        svfloat32_t b3 = svsub_f32_x(pg, svld1_f32(pg, c3), q);
        svfloat32_t b4 = svsub_f32_x(pg, svld1_f32(pg, c4), q);
        svfloat32_t b5 = svsub_f32_x(pg, svld1_f32(pg, c5), q);
        svfloat32_t b6 = svsub_f32_x(pg, svld1_f32(pg, c6), q);
        svfloat32_t b7 = svsub_f32_x(pg, svld1_f32(pg, c7), q);
        svfloat32_t b8 = svsub_f32_x(pg, svld1_f32(pg, c8), q);
        svfloat32_t res1 = svmul_f32_z(pg, b1, b1);
        svfloat32_t res2 = svmul_f32_z(pg, b2, b2);
        svfloat32_t res3 = svmul_f32_z(pg, b3, b3);
        svfloat32_t res4 = svmul_f32_z(pg, b4, b4);
        svfloat32_t res5 = svmul_f32_z(pg, b5, b5);
        svfloat32_t res6 = svmul_f32_z(pg, b6, b6);
        svfloat32_t res7 = svmul_f32_z(pg, b7, b7);
        svfloat32_t res8 = svmul_f32_z(pg, b8, b8);
        for (i = step; i < dim; i += step) {
            pg = svwhilelt_b32(i, dim);
            q = svld1_f32(pg, query + i);
            b1 = svsub_f32_x(pg, svld1_f32(pg, c1 + i), q);
            b2 = svsub_f32_x(pg, svld1_f32(pg, c2 + i), q);
            b3 = svsub_f32_x(pg, svld1_f32(pg, c3 + i), q);
            b4 = svsub_f32_x(pg, svld1_f32(pg, c4 + i), q);
            b5 = svsub_f32_x(pg, svld1_f32(pg, c5 + i), q);
            b6 = svsub_f32_x(pg, svld1_f32(pg, c6 + i), q);
            b7 = svsub_f32_x(pg, svld1_f32(pg, c7 + i), q);
            b8 = svsub_f32_x(pg, svld1_f32(pg, c8 + i), q);
            res1 = svmla_f32_m(pg, res1, b1, b1);
            res2 = svmla_f32_m(pg, res2, b2, b2);
            res3 = svmla_f32_m(pg, res3, b3, b3);
            res4 = svmla_f32_m(pg, res4, b4, b4);
            res5 = svmla_f32_m(pg, res5, b5, b5);
            res6 = svmla_f32_m(pg, res6, b6, b6);
            res7 = svmla_f32_m(pg, res7, b7, b7);
            res8 = svmla_f32_m(pg, res8, b8, b8);
        }
        r1 = svaddv_f32(svptrue_b32(), res1);
        r2 = svaddv_f32(svptrue_b32(), res2);
        r3 = svaddv_f32(svptrue_b32(), res3);
        r4 = svaddv_f32(svptrue_b32(), res4);
        r5 = svaddv_f32(svptrue_b32(), res5);
        r6 = svaddv_f32(svptrue_b32(), res6);
        r7 = svaddv_f32(svptrue_b32(), res7);
        r8 = svaddv_f32(svptrue_b32(), res8);
    }
#else
    return neon::FP32ComputeL2SqrBatch8(
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
#if defined(ENABLE_SVE)
    const uint64_t step = svcntw();
    constexpr uint64_t multi_round = 16;
    uint64_t i;

    if (dim >= multi_round) {
        svfloat32_t res1 = svdup_f32(0.0f);
        svfloat32_t res2 = svdup_f32(0.0f);
        svfloat32_t res3 = svdup_f32(0.0f);
        svfloat32_t res4 = svdup_f32(0.0f);
        svfloat32_t res5 = svdup_f32(0.0f);
        svfloat32_t res6 = svdup_f32(0.0f);
        svfloat32_t res7 = svdup_f32(0.0f);
        svfloat32_t res8 = svdup_f32(0.0f);
        svfloat32_t res9 = svdup_f32(0.0f);
        svfloat32_t res10 = svdup_f32(0.0f);
        svfloat32_t res11 = svdup_f32(0.0f);
        svfloat32_t res12 = svdup_f32(0.0f);
        svfloat32_t res13 = svdup_f32(0.0f);
        svfloat32_t res14 = svdup_f32(0.0f);
        svfloat32_t res15 = svdup_f32(0.0f);
        svfloat32_t res16 = svdup_f32(0.0f);

        svbool_t pg = svptrue_b32();
        for (i = 0; i <= dim - multi_round; i += multi_round) {
            // __builtin_prefetch(query + i + multi_round, 0, 1);
            // __builtin_prefetch(c1 + i + multi_round, 0, 3);
            // __builtin_prefetch(c2 + i + multi_round, 0, 3);
            // __builtin_prefetch(c3 + i + multi_round, 0, 3);
            // __builtin_prefetch(c4 + i + multi_round, 0, 3);
            // __builtin_prefetch(c5 + i + multi_round, 0, 3);
            // __builtin_prefetch(c6 + i + multi_round, 0, 3);
            // __builtin_prefetch(c7 + i + multi_round, 0, 3);
            // __builtin_prefetch(c8 + i + multi_round, 0, 3);
            // __builtin_prefetch(c9 + i + multi_round, 0, 3);
            // __builtin_prefetch(c10 + i + multi_round, 0, 3);
            // __builtin_prefetch(c11 + i + multi_round, 0, 3);
            // __builtin_prefetch(c12 + i + multi_round, 0, 3);
            // __builtin_prefetch(c13 + i + multi_round, 0, 3);
            // __builtin_prefetch(c14 + i + multi_round, 0, 3);
            // __builtin_prefetch(c15 + i + multi_round, 0, 3);
            // __builtin_prefetch(c16 + i + multi_round, 0, 3);

            for (uint64_t j = 0; j < multi_round; j += step) {
                svfloat32_t q = svld1_f32(pg, query + i + j);

                svfloat32_t b1 = svsub_f32_x(pg, svld1_f32(pg, c1 + i + j), q);
                svfloat32_t b2 = svsub_f32_x(pg, svld1_f32(pg, c2 + i + j), q);
                svfloat32_t b3 = svsub_f32_x(pg, svld1_f32(pg, c3 + i + j), q);
                svfloat32_t b4 = svsub_f32_x(pg, svld1_f32(pg, c4 + i + j), q);
                svfloat32_t b5 = svsub_f32_x(pg, svld1_f32(pg, c5 + i + j), q);
                svfloat32_t b6 = svsub_f32_x(pg, svld1_f32(pg, c6 + i + j), q);
                svfloat32_t b7 = svsub_f32_x(pg, svld1_f32(pg, c7 + i + j), q);
                svfloat32_t b8 = svsub_f32_x(pg, svld1_f32(pg, c8 + i + j), q);

                res1 = svmla_f32_m(pg, res1, b1, b1);
                res2 = svmla_f32_m(pg, res2, b2, b2);
                res3 = svmla_f32_m(pg, res3, b3, b3);
                res4 = svmla_f32_m(pg, res4, b4, b4);
                res5 = svmla_f32_m(pg, res5, b5, b5);
                res6 = svmla_f32_m(pg, res6, b6, b6);
                res7 = svmla_f32_m(pg, res7, b7, b7);
                res8 = svmla_f32_m(pg, res8, b8, b8);

                b1 = svsub_f32_x(pg, svld1_f32(pg, c9 + i + j), q);
                b2 = svsub_f32_x(pg, svld1_f32(pg, c10 + i + j), q);
                b3 = svsub_f32_x(pg, svld1_f32(pg, c11 + i + j), q);
                b4 = svsub_f32_x(pg, svld1_f32(pg, c12 + i + j), q);
                b5 = svsub_f32_x(pg, svld1_f32(pg, c13 + i + j), q);
                b6 = svsub_f32_x(pg, svld1_f32(pg, c14 + i + j), q);
                b7 = svsub_f32_x(pg, svld1_f32(pg, c15 + i + j), q);
                b8 = svsub_f32_x(pg, svld1_f32(pg, c16 + i + j), q);

                res9 = svmla_f32_m(pg, res9, b1, b1);
                res10 = svmla_f32_m(pg, res10, b2, b2);
                res11 = svmla_f32_m(pg, res11, b3, b3);
                res12 = svmla_f32_m(pg, res12, b4, b4);
                res13 = svmla_f32_m(pg, res13, b5, b5);
                res14 = svmla_f32_m(pg, res14, b6, b6);
                res15 = svmla_f32_m(pg, res15, b7, b7);
                res16 = svmla_f32_m(pg, res16, b8, b8);
            }
        }

        for (; i <= dim - step; i += step) {
            svfloat32_t q = svld1_f32(pg, query + i);

            svfloat32_t b1 = svsub_f32_x(pg, svld1_f32(pg, c1 + i), q);
            svfloat32_t b2 = svsub_f32_x(pg, svld1_f32(pg, c2 + i), q);
            svfloat32_t b3 = svsub_f32_x(pg, svld1_f32(pg, c3 + i), q);
            svfloat32_t b4 = svsub_f32_x(pg, svld1_f32(pg, c4 + i), q);
            svfloat32_t b5 = svsub_f32_x(pg, svld1_f32(pg, c5 + i), q);
            svfloat32_t b6 = svsub_f32_x(pg, svld1_f32(pg, c6 + i), q);
            svfloat32_t b7 = svsub_f32_x(pg, svld1_f32(pg, c7 + i), q);
            svfloat32_t b8 = svsub_f32_x(pg, svld1_f32(pg, c8 + i), q);

            res1 = svmla_f32_m(pg, res1, b1, b1);
            res2 = svmla_f32_m(pg, res2, b2, b2);
            res3 = svmla_f32_m(pg, res3, b3, b3);
            res4 = svmla_f32_m(pg, res4, b4, b4);
            res5 = svmla_f32_m(pg, res5, b5, b5);
            res6 = svmla_f32_m(pg, res6, b6, b6);
            res7 = svmla_f32_m(pg, res7, b7, b7);
            res8 = svmla_f32_m(pg, res8, b8, b8);

            b1 = svsub_f32_x(pg, svld1_f32(pg, c9 + i), q);
            b2 = svsub_f32_x(pg, svld1_f32(pg, c10 + i), q);
            b3 = svsub_f32_x(pg, svld1_f32(pg, c11 + i), q);
            b4 = svsub_f32_x(pg, svld1_f32(pg, c12 + i), q);
            b5 = svsub_f32_x(pg, svld1_f32(pg, c13 + i), q);
            b6 = svsub_f32_x(pg, svld1_f32(pg, c14 + i), q);
            b7 = svsub_f32_x(pg, svld1_f32(pg, c15 + i), q);
            b8 = svsub_f32_x(pg, svld1_f32(pg, c16 + i), q);

            res9 = svmla_f32_m(pg, res9, b1, b1);
            res10 = svmla_f32_m(pg, res10, b2, b2);
            res11 = svmla_f32_m(pg, res11, b3, b3);
            res12 = svmla_f32_m(pg, res12, b4, b4);
            res13 = svmla_f32_m(pg, res13, b5, b5);
            res14 = svmla_f32_m(pg, res14, b6, b6);
            res15 = svmla_f32_m(pg, res15, b7, b7);
            res16 = svmla_f32_m(pg, res16, b8, b8);
        }

        r1 = svaddv_f32(pg, res1);
        r2 = svaddv_f32(pg, res2);
        r3 = svaddv_f32(pg, res3);
        r4 = svaddv_f32(pg, res4);
        r5 = svaddv_f32(pg, res5);
        r6 = svaddv_f32(pg, res6);
        r7 = svaddv_f32(pg, res7);
        r8 = svaddv_f32(pg, res8);
        r9 = svaddv_f32(pg, res9);
        r10 = svaddv_f32(pg, res10);
        r11 = svaddv_f32(pg, res11);
        r12 = svaddv_f32(pg, res12);
        r13 = svaddv_f32(pg, res13);
        r14 = svaddv_f32(pg, res14);
        r15 = svaddv_f32(pg, res15);
        r16 = svaddv_f32(pg, res16);
    } else if (dim >= step) {
        svbool_t pg = svptrue_b32();
        svfloat32_t q = svld1_f32(pg, query);

        svfloat32_t b1 = svsub_f32_x(pg, svld1_f32(pg, c1), q);
        svfloat32_t b2 = svsub_f32_x(pg, svld1_f32(pg, c2), q);
        svfloat32_t b3 = svsub_f32_x(pg, svld1_f32(pg, c3), q);
        svfloat32_t b4 = svsub_f32_x(pg, svld1_f32(pg, c4), q);
        svfloat32_t b5 = svsub_f32_x(pg, svld1_f32(pg, c5), q);
        svfloat32_t b6 = svsub_f32_x(pg, svld1_f32(pg, c6), q);
        svfloat32_t b7 = svsub_f32_x(pg, svld1_f32(pg, c7), q);
        svfloat32_t b8 = svsub_f32_x(pg, svld1_f32(pg, c8), q);

        svfloat32_t res1 = svmul_f32_z(pg, b1, b1);
        svfloat32_t res2 = svmul_f32_z(pg, b2, b2);
        svfloat32_t res3 = svmul_f32_z(pg, b3, b3);
        svfloat32_t res4 = svmul_f32_z(pg, b4, b4);
        svfloat32_t res5 = svmul_f32_z(pg, b5, b5);
        svfloat32_t res6 = svmul_f32_z(pg, b6, b6);
        svfloat32_t res7 = svmul_f32_z(pg, b7, b7);
        svfloat32_t res8 = svmul_f32_z(pg, b8, b8);

        b1 = svsub_f32_x(pg, svld1_f32(pg, c9), q);
        b2 = svsub_f32_x(pg, svld1_f32(pg, c10), q);
        b3 = svsub_f32_x(pg, svld1_f32(pg, c11), q);
        b4 = svsub_f32_x(pg, svld1_f32(pg, c12), q);
        b5 = svsub_f32_x(pg, svld1_f32(pg, c13), q);
        b6 = svsub_f32_x(pg, svld1_f32(pg, c14), q);
        b7 = svsub_f32_x(pg, svld1_f32(pg, c15), q);
        b8 = svsub_f32_x(pg, svld1_f32(pg, c16), q);

        svfloat32_t res9 = svmul_f32_z(pg, b1, b1);
        svfloat32_t res10 = svmul_f32_z(pg, b2, b2);
        svfloat32_t res11 = svmul_f32_z(pg, b3, b3);
        svfloat32_t res12 = svmul_f32_z(pg, b4, b4);
        svfloat32_t res13 = svmul_f32_z(pg, b5, b5);
        svfloat32_t res14 = svmul_f32_z(pg, b6, b6);
        svfloat32_t res15 = svmul_f32_z(pg, b7, b7);
        svfloat32_t res16 = svmul_f32_z(pg, b8, b8);

        for (i = step; i <= dim - step; i += step) {
            q = svld1_f32(pg, query + i);

            b1 = svsub_f32_x(pg, svld1_f32(pg, c1 + i), q);
            b2 = svsub_f32_x(pg, svld1_f32(pg, c2 + i), q);
            b3 = svsub_f32_x(pg, svld1_f32(pg, c3 + i), q);
            b4 = svsub_f32_x(pg, svld1_f32(pg, c4 + i), q);
            b5 = svsub_f32_x(pg, svld1_f32(pg, c5 + i), q);
            b6 = svsub_f32_x(pg, svld1_f32(pg, c6 + i), q);
            b7 = svsub_f32_x(pg, svld1_f32(pg, c7 + i), q);
            b8 = svsub_f32_x(pg, svld1_f32(pg, c8 + i), q);

            res1 = svmla_f32_m(pg, res1, b1, b1);
            res2 = svmla_f32_m(pg, res2, b2, b2);
            res3 = svmla_f32_m(pg, res3, b3, b3);
            res4 = svmla_f32_m(pg, res4, b4, b4);
            res5 = svmla_f32_m(pg, res5, b5, b5);
            res6 = svmla_f32_m(pg, res6, b6, b6);
            res7 = svmla_f32_m(pg, res7, b7, b7);
            res8 = svmla_f32_m(pg, res8, b8, b8);

            b1 = svsub_f32_x(pg, svld1_f32(pg, c9 + i), q);
            b2 = svsub_f32_x(pg, svld1_f32(pg, c10 + i), q);
            b3 = svsub_f32_x(pg, svld1_f32(pg, c11 + i), q);
            b4 = svsub_f32_x(pg, svld1_f32(pg, c12 + i), q);
            b5 = svsub_f32_x(pg, svld1_f32(pg, c13 + i), q);
            b6 = svsub_f32_x(pg, svld1_f32(pg, c14 + i), q);
            b7 = svsub_f32_x(pg, svld1_f32(pg, c15 + i), q);
            b8 = svsub_f32_x(pg, svld1_f32(pg, c16 + i), q);

            res9 = svmla_f32_m(pg, res9, b1, b1);
            res10 = svmla_f32_m(pg, res10, b2, b2);
            res11 = svmla_f32_m(pg, res11, b3, b3);
            res12 = svmla_f32_m(pg, res12, b4, b4);
            res13 = svmla_f32_m(pg, res13, b5, b5);
            res14 = svmla_f32_m(pg, res14, b6, b6);
            res15 = svmla_f32_m(pg, res15, b7, b7);
            res16 = svmla_f32_m(pg, res16, b8, b8);
        }

        r1 = svaddv_f32(pg, res1);
        r2 = svaddv_f32(pg, res2);
        r3 = svaddv_f32(pg, res3);
        r4 = svaddv_f32(pg, res4);
        r5 = svaddv_f32(pg, res5);
        r6 = svaddv_f32(pg, res6);
        r7 = svaddv_f32(pg, res7);
        r8 = svaddv_f32(pg, res8);
        r9 = svaddv_f32(pg, res9);
        r10 = svaddv_f32(pg, res10);
        r11 = svaddv_f32(pg, res11);
        r12 = svaddv_f32(pg, res12);
        r13 = svaddv_f32(pg, res13);
        r14 = svaddv_f32(pg, res14);
        r15 = svaddv_f32(pg, res15);
        r16 = svaddv_f32(pg, res16);
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
    return neon::FP32ComputeL2SqrBatch16(query,
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
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t x_vec = svld1_f32(predicate, x + i);
        svfloat32_t y_vec = svld1_f32(predicate, y + i);
        svfloat32_t result = svsub_f32_z(predicate, x_vec, y_vec);
        svst1_f32(predicate, z + i, result);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));
#else
    neon::FP32Sub(x, y, z, dim);
#endif
}

void
FP32Add(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t x_vec = svld1_f32(predicate, x + i);
        svfloat32_t y_vec = svld1_f32(predicate, y + i);
        svfloat32_t result = svadd_f32_z(predicate, x_vec, y_vec);
        svst1_f32(predicate, z + i, result);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));
#else
    neon::FP32Add(x, y, z, dim);
#endif
}

void
FP32Mul(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t x_vec = svld1_f32(predicate, x + i);
        svfloat32_t y_vec = svld1_f32(predicate, y + i);
        svfloat32_t result = svmul_f32_z(predicate, x_vec, y_vec);
        svst1_f32(predicate, z + i, result);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));
#else
    neon::FP32Mul(x, y, z, dim);
#endif
}

void
FP32Div(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t x_vec = svld1_f32(predicate, x + i);
        svfloat32_t y_vec = svld1_f32(predicate, y + i);
        svfloat32_t result = svdiv_f32_z(predicate, x_vec, y_vec);
        svst1_f32(predicate, z + i, result);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));
#else
    neon::FP32Div(x, y, z, dim);
#endif
}

float
FP32ReduceAdd(const float* x, uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum = svdup_f32(0.0f);
    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t x_vec = svld1_f32(predicate, x + i);

        sum = svadd_f32_m(predicate, sum, x_vec);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::FP32ReduceAdd(x, dim);
#endif
}

float
BF16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SVE)
    auto* query_bf16 = reinterpret_cast<const uint16_t*>(query);
    auto* codes_bf16 = reinterpret_cast<const uint16_t*>(codes);

    svfloat32_t sum = svdup_n_f32(0.0f);
    uint64_t i = 0;
    uint64_t step = svcntw();
    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svuint32_t query_u32 = svld1uh_u32(predicate, &query_bf16[i]);
        svuint32_t codes_u32 = svld1uh_u32(predicate, &codes_bf16[i]);

        query_u32 = svlsl_n_u32_x(predicate, query_u32, 16);
        codes_u32 = svlsl_n_u32_x(predicate, codes_u32, 16);

        svfloat32_t query_f32 = svreinterpret_f32_u32(query_u32);
        svfloat32_t codes_f32 = svreinterpret_f32_u32(codes_u32);

        sum = svmla_f32_x(predicate, sum, query_f32, codes_f32);
        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::BF16ComputeIP(query, codes, dim);
#endif
}

float
BF16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SVE)
    auto* query_bf16 = reinterpret_cast<const uint16_t*>(query);
    auto* codes_bf16 = reinterpret_cast<const uint16_t*>(codes);

    svfloat32_t sum = svdup_n_f32(0.0f);
    uint64_t i = 0;
    uint64_t step = svcntw();
    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svuint32_t query_u32 = svld1uh_u32(predicate, &query_bf16[i]);
        svuint32_t codes_u32 = svld1uh_u32(predicate, &codes_bf16[i]);

        query_u32 = svlsl_n_u32_x(predicate, query_u32, 16);
        codes_u32 = svlsl_n_u32_x(predicate, codes_u32, 16);

        svfloat32_t query_f32 = svreinterpret_f32_u32(query_u32);
        svfloat32_t codes_f32 = svreinterpret_f32_u32(codes_u32);

        svfloat32_t diff = svsub_f32_x(predicate, query_f32, codes_f32);
        sum = svmla_f32_x(predicate, sum, diff, diff);
        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::BF16ComputeL2Sqr(query, codes, dim);
#endif
}

float
FP16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SVE)
    auto* query_fp16 = reinterpret_cast<const __fp16*>(query);
    auto* codes_fp16 = reinterpret_cast<const __fp16*>(codes);

    svfloat32_t sum = svdup_n_f32(0.0f);
    uint64_t i = 0;
    uint64_t step = svcnth();
    svbool_t predicate = svwhilelt_b16(i, dim);
    do {
        svfloat16_t query_f16 = svld1_f16(predicate, &query_fp16[i]);
        svfloat16_t codes_f16 = svld1_f16(predicate, &codes_fp16[i]);

        svbool_t half_predicate = svptrue_pat_b16(SV_POW2);
        svfloat32_t query_f32_low = svcvt_f32_f16_x(half_predicate, query_f16);
        svfloat32_t codes_f32_low = svcvt_f32_f16_x(half_predicate, codes_f16);

        svfloat16_t query_f16_high = svreinterpret_f16_u32(
            svlsr_n_u32_x(svptrue_b32(), svreinterpret_u32_f16(query_f16), 16));
        svfloat16_t codes_f16_high = svreinterpret_f16_u32(
            svlsr_n_u32_x(svptrue_b32(), svreinterpret_u32_f16(codes_f16), 16));
        svfloat32_t query_f32_high = svcvt_f32_f16_x(half_predicate, query_f16_high);
        svfloat32_t codes_f32_high = svcvt_f32_f16_x(half_predicate, codes_f16_high);

        sum = svmla_f32_x(svptrue_b32(), sum, query_f32_low, codes_f32_low);
        sum = svmla_f32_x(svptrue_b32(), sum, query_f32_high, codes_f32_high);
        i += step;
        predicate = svwhilelt_b16(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::FP16ComputeIP(query, codes, dim);
#endif
}

float
FP16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SVE)

    auto* query_fp16 = reinterpret_cast<const __fp16*>(query);
    auto* codes_fp16 = reinterpret_cast<const __fp16*>(codes);

    svfloat32_t sum = svdup_n_f32(0.0f);
    uint64_t i = 0;
    uint64_t step = svcnth();
    svbool_t predicate = svwhilelt_b16(i, dim);
    do {
        svfloat16_t query_f16 = svld1_f16(predicate, &query_fp16[i]);
        svfloat16_t codes_f16 = svld1_f16(predicate, &codes_fp16[i]);

        svbool_t half_predicate = svptrue_pat_b16(SV_POW2);
        svfloat32_t query_f32_low = svcvt_f32_f16_x(half_predicate, query_f16);
        svfloat32_t codes_f32_low = svcvt_f32_f16_x(half_predicate, codes_f16);

        svfloat16_t query_f16_high = svreinterpret_f16_u32(
            svlsr_n_u32_x(svptrue_b32(), svreinterpret_u32_f16(query_f16), 16));
        svfloat16_t codes_f16_high = svreinterpret_f16_u32(
            svlsr_n_u32_x(svptrue_b32(), svreinterpret_u32_f16(codes_f16), 16));
        svfloat32_t query_f32_high = svcvt_f32_f16_x(half_predicate, query_f16_high);
        svfloat32_t codes_f32_high = svcvt_f32_f16_x(half_predicate, codes_f16_high);

        svfloat32_t diff_low = svsub_f32_x(svptrue_b32(), query_f32_low, codes_f32_low);
        svfloat32_t diff_high = svsub_f32_x(svptrue_b32(), query_f32_high, codes_f32_high);

        sum = svmla_f32_x(svptrue_b32(), sum, diff_low, diff_low);
        sum = svmla_f32_x(svptrue_b32(), sum, diff_high, diff_high);
        i += step;
        predicate = svwhilelt_b16(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::FP16ComputeL2Sqr(query, codes, dim);
#endif
}

void
FP16SparseAccumulate(float* RESTRICT dists,
                     const uint16_t* RESTRICT ids,
                     const uint16_t* RESTRICT vals,
                     float query_val,
                     uint32_t num) {
    return neon::FP16SparseAccumulate(dists, ids, vals, query_val, num);
}

float
SQ8ComputeIP(const float* RESTRICT query,
             const uint8_t* RESTRICT codes,
             const float* RESTRICT lower_bound,
             const float* RESTRICT diff,
             uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum = svdup_f32(0.0f);
    const svfloat32_t scale_factor = svdup_f32(1.0f / 255.0f);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svuint32_t codes_u32 = svld1ub_u32(predicate, codes + i);
        svfloat32_t codes_f32 = svcvt_f32_u32_z(predicate, codes_u32);

        svfloat32_t query_vec = svld1_f32(predicate, query + i);
        svfloat32_t lower_bound_vec = svld1_f32(predicate, lower_bound + i);
        svfloat32_t diff_vec = svld1_f32(predicate, diff + i);

        svfloat32_t dequantized = svmla_f32_m(
            predicate, lower_bound_vec, svmul_f32_m(predicate, codes_f32, scale_factor), diff_vec);

        sum = svmla_f32_m(predicate, sum, query_vec, dequantized);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::SQ8ComputeIP(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ8ComputeL2Sqr(const float* RESTRICT query,
                const uint8_t* RESTRICT codes,
                const float* RESTRICT lower_bound,
                const float* RESTRICT diff,
                uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum = svdup_f32(0.0f);
    const svfloat32_t scale_factor = svdup_f32(1.0f / 255.0f);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svuint32_t codes_u32 = svld1ub_u32(predicate, codes + i);
        svfloat32_t codes_f32 = svcvt_f32_u32_z(predicate, codes_u32);

        svfloat32_t query_vec = svld1_f32(predicate, query + i);
        svfloat32_t lower_bound_vec = svld1_f32(predicate, lower_bound + i);
        svfloat32_t diff_vec = svld1_f32(predicate, diff + i);

        svfloat32_t dequantized = svmla_f32_m(
            predicate, lower_bound_vec, svmul_f32_m(predicate, codes_f32, scale_factor), diff_vec);
        svfloat32_t delta = svsub_f32_z(predicate, query_vec, dequantized);
        sum = svmla_f32_m(predicate, sum, delta, delta);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::SQ8ComputeL2Sqr(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ8ComputeCodesIP(const uint8_t* RESTRICT codes1,
                  const uint8_t* RESTRICT codes2,
                  const float* RESTRICT lower_bound,
                  const float* RESTRICT diff,
                  uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum = svdup_f32(0.0f);
    const svfloat32_t scale_factor = svdup_f32(1.0f / 255.0f);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svuint32_t codes1_u32 = svld1ub_u32(predicate, codes1 + i);
        svfloat32_t codes1_f32 = svcvt_f32_u32_z(predicate, codes1_u32);
        svuint32_t codes2_u32 = svld1ub_u32(predicate, codes2 + i);
        svfloat32_t codes2_f32 = svcvt_f32_u32_z(predicate, codes2_u32);

        svfloat32_t lower_bound_vec = svld1_f32(predicate, lower_bound + i);
        svfloat32_t diff_vec = svld1_f32(predicate, diff + i);

        svfloat32_t dequantized1 = svmla_f32_m(
            predicate, lower_bound_vec, svmul_f32_m(predicate, codes1_f32, scale_factor), diff_vec);
        svfloat32_t dequantized2 = svmla_f32_m(
            predicate, lower_bound_vec, svmul_f32_m(predicate, codes2_f32, scale_factor), diff_vec);

        sum = svmla_f32_m(predicate, sum, dequantized1, dequantized2);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::SQ8ComputeCodesIP(codes1, codes2, lower_bound, diff, dim);
#endif
}

float
SQ8ComputeCodesL2Sqr(const uint8_t* RESTRICT codes1,
                     const uint8_t* RESTRICT codes2,
                     const float* RESTRICT lower_bound,
                     const float* RESTRICT diff,
                     uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum = svdup_f32(0.0f);
    const svfloat32_t scale_factor = svdup_f32(1.0f / 255.0f);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svuint32_t codes1_u32 = svld1ub_u32(predicate, codes1 + i);
        svfloat32_t codes1_f32 = svcvt_f32_u32_z(predicate, codes1_u32);
        svuint32_t codes2_u32 = svld1ub_u32(predicate, codes2 + i);
        svfloat32_t codes2_f32 = svcvt_f32_u32_z(predicate, codes2_u32);

        svfloat32_t lower_bound_vec = svld1_f32(predicate, lower_bound + i);
        svfloat32_t diff_vec = svld1_f32(predicate, diff + i);

        svfloat32_t dequantized1 = svmla_f32_m(
            predicate, lower_bound_vec, svmul_f32_m(predicate, codes1_f32, scale_factor), diff_vec);
        svfloat32_t dequantized2 = svmla_f32_m(
            predicate, lower_bound_vec, svmul_f32_m(predicate, codes2_f32, scale_factor), diff_vec);

        svfloat32_t delta = svsub_f32_z(predicate, dequantized1, dequantized2);
        sum = svmla_f32_m(predicate, sum, delta, delta);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::SQ8ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
#endif
}

void
SQ8SparseAccumulate(float* RESTRICT dists,
                    const uint16_t* RESTRICT ids,
                    const uint8_t* RESTRICT vals,
                    float query_val,
                    uint32_t num) {
    return neon::SQ8SparseAccumulate(dists, ids, vals, query_val, num);
}

float
SQ4ComputeIP(const float* RESTRICT query,
             const uint8_t* RESTRICT codes,
             const float* RESTRICT lower_bound,
             const float* RESTRICT diff,
             uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum = svdup_f32(0.0f);
    const svfloat32_t scale_factor = svdup_f32(1.0f / 15.0f);
    const uint64_t step = svcntw();
    uint64_t i = 0;
    const svbool_t predicate = svwhilelt_b32(i, dim);

    for (; i + 2 * step <= dim; i += 2 * step) {
        svfloat32x2_t query_pair = svld2_f32(predicate, &query[i]);
        svfloat32x2_t lower_bound_pair = svld2_f32(predicate, &lower_bound[i]);
        svfloat32x2_t diff_pair = svld2_f32(predicate, &diff[i]);

        svfloat32_t query_even = svget2_f32(query_pair, 0);
        svfloat32_t query_odd = svget2_f32(query_pair, 1);
        svfloat32_t lower_bound_even = svget2_f32(lower_bound_pair, 0);
        svfloat32_t lower_bound_odd = svget2_f32(lower_bound_pair, 1);
        svfloat32_t diff_even = svget2_f32(diff_pair, 0);
        svfloat32_t diff_odd = svget2_f32(diff_pair, 1);

        svuint32_t packed_codes = svld1ub_u32(predicate, &codes[i / 2]);
        svuint32_t codes_even_u32 = svand_n_u32_x(predicate, packed_codes, 0x0F);
        svuint32_t codes_odd_u32 = svlsr_n_u32_x(predicate, packed_codes, 4);
        svfloat32_t codes_even_f32 = svcvt_f32_u32_x(predicate, codes_even_u32);
        svfloat32_t codes_odd_f32 = svcvt_f32_u32_x(predicate, codes_odd_u32);

        svfloat32_t dequantized_even =
            svmla_f32_x(predicate,
                        lower_bound_even,
                        svmul_f32_x(predicate, codes_even_f32, scale_factor),
                        diff_even);
        svfloat32_t dequantized_odd =
            svmla_f32_x(predicate,
                        lower_bound_odd,
                        svmul_f32_x(predicate, codes_odd_f32, scale_factor),
                        diff_odd);

        sum = svmla_f32_x(predicate, sum, query_even, dequantized_even);
        sum = svmla_f32_x(predicate, sum, query_odd, dequantized_odd);
    }

    if (i < dim) {
        return svaddv_f32(predicate, sum) +
               neon::SQ4ComputeIP(&query[i], &codes[i / 2], &lower_bound[i], &diff[i], dim - i);
    }

    return svaddv_f32(predicate, sum);
#else
    return neon::SQ4ComputeIP(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ4ComputeL2Sqr(const float* RESTRICT query,
                const uint8_t* RESTRICT codes,
                const float* RESTRICT lower_bound,
                const float* RESTRICT diff,
                uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum = svdup_f32(0.0f);
    const svfloat32_t scale_factor = svdup_f32(1.0f / 15.0f);
    const uint64_t step = svcntw();
    const svbool_t predicate = svptrue_b32();

    uint64_t i = 0;
    for (; i + 2 * step <= dim; i += 2 * step) {
        svfloat32x2_t query_pair = svld2_f32(predicate, &query[i]);
        svfloat32x2_t lower_bound_pair = svld2_f32(predicate, &lower_bound[i]);
        svfloat32x2_t diff_pair = svld2_f32(predicate, &diff[i]);

        svfloat32_t query_even = svget2_f32(query_pair, 0);
        svfloat32_t query_odd = svget2_f32(query_pair, 1);
        svfloat32_t lower_bound_even = svget2_f32(lower_bound_pair, 0);
        svfloat32_t lower_bound_odd = svget2_f32(lower_bound_pair, 1);
        svfloat32_t diff_even = svget2_f32(diff_pair, 0);
        svfloat32_t diff_odd = svget2_f32(diff_pair, 1);

        svuint32_t packed_codes = svld1ub_u32(predicate, &codes[i / 2]);
        svuint32_t codes_even_u32 = svand_n_u32_x(predicate, packed_codes, 0x0F);
        svuint32_t codes_odd_u32 = svlsr_n_u32_x(predicate, packed_codes, 4);
        svfloat32_t codes_even_f32 = svcvt_f32_u32_x(predicate, codes_even_u32);
        svfloat32_t codes_odd_f32 = svcvt_f32_u32_x(predicate, codes_odd_u32);

        svfloat32_t dequantized_even =
            svmla_f32_x(predicate,
                        lower_bound_even,
                        svmul_f32_x(predicate, codes_even_f32, scale_factor),
                        diff_even);
        svfloat32_t dequantized_odd =
            svmla_f32_x(predicate,
                        lower_bound_odd,
                        svmul_f32_x(predicate, codes_odd_f32, scale_factor),
                        diff_odd);

        svfloat32_t delta_even = svsub_f32_x(predicate, query_even, dequantized_even);
        svfloat32_t delta_odd = svsub_f32_x(predicate, query_odd, dequantized_odd);

        sum = svmla_f32_x(predicate, sum, delta_even, delta_even);
        sum = svmla_f32_x(predicate, sum, delta_odd, delta_odd);
    }

    if (i < dim) {
        return svaddv_f32(predicate, sum) +
               neon::SQ4ComputeL2Sqr(&query[i], &codes[i / 2], &lower_bound[i], &diff[i], dim - i);
    }

    return svaddv_f32(predicate, sum);
#else
    return neon::SQ4ComputeL2Sqr(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ4ComputeCodesIP(const uint8_t* RESTRICT codes1,
                  const uint8_t* RESTRICT codes2,
                  const float* RESTRICT lower_bound,
                  const float* RESTRICT diff,
                  uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum = svdup_f32(0.0f);
    const svfloat32_t scale_factor = svdup_f32(1.0f / 15.0f);
    const uint64_t step = svcntw();
    const svbool_t predicate = svptrue_b32();

    uint64_t i = 0;
    for (; i + 2 * step <= dim; i += 2 * step) {
        svfloat32x2_t lower_bound_pair = svld2_f32(predicate, &lower_bound[i]);
        svfloat32x2_t diff_pair = svld2_f32(predicate, &diff[i]);

        svfloat32_t lower_bound_even = svget2_f32(lower_bound_pair, 0);
        svfloat32_t lower_bound_odd = svget2_f32(lower_bound_pair, 1);
        svfloat32_t diff_even = svget2_f32(diff_pair, 0);
        svfloat32_t diff_odd = svget2_f32(diff_pair, 1);

        svuint32_t packed_codes1 = svld1ub_u32(predicate, &codes1[i / 2]);
        svuint32_t packed_codes2 = svld1ub_u32(predicate, &codes2[i / 2]);

        svuint32_t codes1_even_u32 = svand_n_u32_x(predicate, packed_codes1, 0x0F);
        svuint32_t codes1_odd_u32 = svlsr_n_u32_x(predicate, packed_codes1, 4);
        svuint32_t codes2_even_u32 = svand_n_u32_x(predicate, packed_codes2, 0x0F);
        svuint32_t codes2_odd_u32 = svlsr_n_u32_x(predicate, packed_codes2, 4);

        svfloat32_t codes1_even_f32 = svcvt_f32_u32_x(predicate, codes1_even_u32);
        svfloat32_t codes1_odd_f32 = svcvt_f32_u32_x(predicate, codes1_odd_u32);
        svfloat32_t codes2_even_f32 = svcvt_f32_u32_x(predicate, codes2_even_u32);
        svfloat32_t codes2_odd_f32 = svcvt_f32_u32_x(predicate, codes2_odd_u32);

        svfloat32_t dequantized1_even =
            svmla_f32_x(predicate,
                        lower_bound_even,
                        svmul_f32_x(predicate, codes1_even_f32, scale_factor),
                        diff_even);
        svfloat32_t dequantized1_odd =
            svmla_f32_x(predicate,
                        lower_bound_odd,
                        svmul_f32_x(predicate, codes1_odd_f32, scale_factor),
                        diff_odd);
        svfloat32_t dequantized2_even =
            svmla_f32_x(predicate,
                        lower_bound_even,
                        svmul_f32_x(predicate, codes2_even_f32, scale_factor),
                        diff_even);
        svfloat32_t dequantized2_odd =
            svmla_f32_x(predicate,
                        lower_bound_odd,
                        svmul_f32_x(predicate, codes2_odd_f32, scale_factor),
                        diff_odd);

        sum = svmla_f32_x(predicate, sum, dequantized1_even, dequantized2_even);
        sum = svmla_f32_x(predicate, sum, dequantized1_odd, dequantized2_odd);
    }

    if (i < dim) {
        return svaddv_f32(predicate, sum) +
               neon::SQ4ComputeCodesIP(
                   &codes1[i / 2], &codes2[i / 2], &lower_bound[i], &diff[i], dim - i);
    }

    return svaddv_f32(predicate, sum);
#else
    return neon::SQ4ComputeCodesIP(codes1, codes2, lower_bound, diff, dim);
#endif
}

float
SQ4ComputeCodesL2Sqr(const uint8_t* RESTRICT codes1,
                     const uint8_t* RESTRICT codes2,
                     const float* RESTRICT lower_bound,
                     const float* RESTRICT diff,
                     uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum = svdup_f32(0.0f);
    const svfloat32_t scale_factor = svdup_f32(1.0f / 15.0f);
    const uint64_t step = svcntw();
    const svbool_t predicate = svptrue_b32();

    uint64_t i = 0;
    for (; i + 2 * step <= dim; i += 2 * step) {
        svfloat32x2_t lower_bound_pair = svld2_f32(predicate, &lower_bound[i]);
        svfloat32x2_t diff_pair = svld2_f32(predicate, &diff[i]);

        svfloat32_t lower_bound_even = svget2_f32(lower_bound_pair, 0);
        svfloat32_t lower_bound_odd = svget2_f32(lower_bound_pair, 1);
        svfloat32_t diff_even = svget2_f32(diff_pair, 0);
        svfloat32_t diff_odd = svget2_f32(diff_pair, 1);

        svuint32_t packed_codes1 = svld1ub_u32(predicate, &codes1[i / 2]);
        svuint32_t packed_codes2 = svld1ub_u32(predicate, &codes2[i / 2]);

        svuint32_t codes1_even_u32 = svand_n_u32_x(predicate, packed_codes1, 0x0F);
        svuint32_t codes1_odd_u32 = svlsr_n_u32_x(predicate, packed_codes1, 4);
        svuint32_t codes2_even_u32 = svand_n_u32_x(predicate, packed_codes2, 0x0F);
        svuint32_t codes2_odd_u32 = svlsr_n_u32_x(predicate, packed_codes2, 4);

        svfloat32_t codes1_even_f32 = svcvt_f32_u32_x(predicate, codes1_even_u32);
        svfloat32_t codes1_odd_f32 = svcvt_f32_u32_x(predicate, codes1_odd_u32);
        svfloat32_t codes2_even_f32 = svcvt_f32_u32_x(predicate, codes2_even_u32);
        svfloat32_t codes2_odd_f32 = svcvt_f32_u32_x(predicate, codes2_odd_u32);

        svfloat32_t dequantized1_even =
            svmla_f32_x(predicate,
                        lower_bound_even,
                        svmul_f32_x(predicate, codes1_even_f32, scale_factor),
                        diff_even);
        svfloat32_t dequantized1_odd =
            svmla_f32_x(predicate,
                        lower_bound_odd,
                        svmul_f32_x(predicate, codes1_odd_f32, scale_factor),
                        diff_odd);
        svfloat32_t dequantized2_even =
            svmla_f32_x(predicate,
                        lower_bound_even,
                        svmul_f32_x(predicate, codes2_even_f32, scale_factor),
                        diff_even);
        svfloat32_t dequantized2_odd =
            svmla_f32_x(predicate,
                        lower_bound_odd,
                        svmul_f32_x(predicate, codes2_odd_f32, scale_factor),
                        diff_odd);

        svfloat32_t delta_even = svsub_f32_x(predicate, dequantized1_even, dequantized2_even);
        svfloat32_t delta_odd = svsub_f32_x(predicate, dequantized1_odd, dequantized2_odd);

        sum = svmla_f32_x(predicate, sum, delta_even, delta_even);
        sum = svmla_f32_x(predicate, sum, delta_odd, delta_odd);
    }

    if (i < dim) {
        return svaddv_f32(predicate, sum) +
               neon::SQ4ComputeCodesL2Sqr(
                   &codes1[i / 2], &codes2[i / 2], &lower_bound[i], &diff[i], dim - i);
    }

    return svaddv_f32(predicate, sum);
#else
    return neon::SQ4ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
#endif
}
float
SQ4UniformComputeCodesIP(const uint8_t* RESTRICT codes1,
                         const uint8_t* RESTRICT codes2,
                         uint64_t dim) {
#if defined(ENABLE_SVE)
    svuint32_t sum = svdup_u32(0);
    uint64_t i = 0;
    const uint64_t step = svcntb() * 2;
    svbool_t predicate = svwhilelt_b8(i / 2, (dim + 1) / 2);
    do {
        svuint8_t packed_codes1 = svld1_u8(predicate, codes1 + i / 2);
        svuint8_t packed_codes2 = svld1_u8(predicate, codes2 + i / 2);

        svuint8_t codes1_low = svand_u8_z(predicate, packed_codes1, svdup_u8(0x0F));
        svuint8_t codes1_high = svlsr_n_u8_z(predicate, packed_codes1, 4);
        svuint8_t codes2_low = svand_u8_z(predicate, packed_codes2, svdup_u8(0x0F));
        svuint8_t codes2_high = svlsr_n_u8_z(predicate, packed_codes2, 4);

        sum = svdot_u32(sum, codes1_low, codes2_low);
        sum = svdot_u32(sum, codes1_high, codes2_high);

        i += step;
        predicate = svwhilelt_b8(i / 2, (dim + 1) / 2);
    } while (svptest_first(svptrue_b8(), predicate));

    return static_cast<float>(svaddv_u32(svptrue_b32(), sum));
#else
    return neon::SQ4UniformComputeCodesIP(codes1, codes2, dim);
#endif
}

float
SQ8UniformComputeCodesIP(const uint8_t* RESTRICT codes1,
                         const uint8_t* RESTRICT codes2,
                         uint64_t dim) {
#if defined(ENABLE_SVE)
    svuint32_t sum = svdup_u32(0);
    uint64_t i = 0;
    const uint64_t step = svcntb();

    svbool_t predicate = svwhilelt_b8(i, dim);
    do {
        svuint8_t codes1_vec = svld1_u8(predicate, codes1 + i);
        svuint8_t codes2_vec = svld1_u8(predicate, codes2 + i);

        sum = svdot_u32(sum, codes1_vec, codes2_vec);

        i += step;
        predicate = svwhilelt_b8(i, dim);
    } while (svptest_first(svptrue_b8(), predicate));

    return static_cast<float>(svaddv_u32(svptrue_b32(), sum));
#else
    return neon::SQ8UniformComputeCodesIP(codes1, codes2, dim);
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
        out[i] = sve::SQ8UniformComputeCodesIP(query, codes + i * code_stride, dim);
    }
}

float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d) {
#if defined(ENABLE_SVE)
    if (dim == 0) {
        return 0.0f;
    }

    auto predicate_array = std::make_unique<uint8_t[]>(dim);

    const uint64_t num_bytes = dim / 8;
    for (uint64_t i = 0; i < num_bytes; ++i) {
        memcpy(&predicate_array[i * 8], g_bit_lookup_table[bits[i]].data(), 8);
    }

    if (dim % 8 != 0) {
        const uint64_t remaining_bits = dim % 8;
        memcpy(&predicate_array[num_bytes * 8],
               g_bit_lookup_table[bits[num_bytes]].data(),
               remaining_bits);
    }

    uint64_t i = 0;
    const uint64_t step = svcntw();
    svfloat32_t sum = svdup_f32(0.0f);

    const svfloat32_t positive_val = inv_sqrt_d < 1e-3 ? svdup_f32(1.0f) : svdup_f32(inv_sqrt_d);
    const svfloat32_t negative_val = inv_sqrt_d < 1e-3 ? svdup_f32(0.0f) : svdup_f32(-inv_sqrt_d);
    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svuint32_t predicate_values = svld1ub_u32(predicate, &predicate_array[i]);

        svbool_t bit_mask = svcmpne_n_u32(predicate, predicate_values, 0);

        svfloat32_t binary_vec = svsel_f32(bit_mask, positive_val, negative_val);
        svfloat32_t vector_values = svld1_f32(predicate, &vector[i]);
        sum = svmla_f32_m(predicate, sum, vector_values, binary_vec);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::RaBitQFloatBinaryIP(vector, bits, dim, inv_sqrt_d);
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
    neon::RaBitQFloatBinaryIPBatch4(vector, bits1, bits2, bits3, bits4, dim, inv_sqrt_d, results);
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
    neon::RaBitQFloatThreeBitIPBatch4(
        vector, bits1, bits2, bits3, bits4, dim, reorder_bits, results);
}

float
RaBitQFloatSplitCodeIP(const float* vector,
                       const uint8_t* one_bit_code,
                       const uint8_t* supplement_code,
                       uint64_t dim,
                       uint32_t supplement_bits) {
    return neon::RaBitQFloatSplitCodeIP(
        vector, one_bit_code, supplement_code, dim, supplement_bits);
}

float
RaBitQFloatSupplementCodeIP(const float* vector,
                            const uint8_t* supplement_code,
                            uint64_t dim,
                            uint32_t supplement_bits) {
    return neon::RaBitQFloatSupplementCodeIP(vector, supplement_code, dim, supplement_bits);
}

uint32_t
RaBitQSQ4UBinaryIP(const uint8_t* codes, const uint8_t* bits, uint64_t dim) {
#if defined(ENABLE_SVE)
    if (dim == 0) {
        return 0;
    }

    uint32_t result = 0;
    uint64_t num_bytes = (dim + 7) / 8;

    for (uint64_t bit_pos = 0; bit_pos < 4; ++bit_pos) {
        uint64_t i = 0;
        uint64_t sum = 0;

        const uint8_t* current_codes = codes + bit_pos * num_bytes;

        svbool_t predicate = svwhilelt_b8(i, num_bytes);
        do {
            svuint8_t codes_vec = svld1_u8(predicate, current_codes + i);
            svuint8_t bits_vec = svld1_u8(predicate, bits + i);

            svuint8_t and_result = svand_u8_x(predicate, codes_vec, bits_vec);

            svuint8_t popcount = svcnt_u8_x(predicate, and_result);

            sum += svaddv_u8(predicate, popcount);

            i += svcntb();
            predicate = svwhilelt_b8(i, num_bytes);
        } while (svptest_first(svptrue_b8(), predicate));

        result += sum << bit_pos;
    }

    return result;
#else
    return neon::RaBitQSQ4UBinaryIP(codes, bits, dim);
#endif
}

void
BitAnd(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t step = svcntb();
    svbool_t predicate = svwhilelt_b8(i, num_byte);
    do {
        svuint8_t x_vec = svld1_u8(predicate, x + i);
        svuint8_t y_vec = svld1_u8(predicate, y + i);
        svuint8_t result_vec = svand_u8_z(predicate, x_vec, y_vec);
        svst1_u8(predicate, result + i, result_vec);

        i += step;
        predicate = svwhilelt_b8(i, num_byte);
    } while (svptest_first(svptrue_b8(), predicate));
#else
    neon::BitAnd(x, y, num_byte, result);
#endif
}

void
BitOr(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t step = svcntb();
    svbool_t predicate = svwhilelt_b8(i, num_byte);
    do {
        svuint8_t x_vec = svld1_u8(predicate, x + i);
        svuint8_t y_vec = svld1_u8(predicate, y + i);
        svuint8_t result_vec = svorr_u8_z(predicate, x_vec, y_vec);
        svst1_u8(predicate, result + i, result_vec);

        i += step;
        predicate = svwhilelt_b8(i, num_byte);
    } while (svptest_first(svptrue_b8(), predicate));
#else
    neon::BitOr(x, y, num_byte, result);
#endif
}

void
BitXor(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t step = svcntb();
    svbool_t predicate = svwhilelt_b8(i, num_byte);
    do {
        svuint8_t x_vec = svld1_u8(predicate, x + i);
        svuint8_t y_vec = svld1_u8(predicate, y + i);
        svuint8_t result_vec = sveor_u8_z(predicate, x_vec, y_vec);
        svst1_u8(predicate, result + i, result_vec);

        i += step;
        predicate = svwhilelt_b8(i, num_byte);
    } while (svptest_first(svptrue_b8(), predicate));
#else
    neon::BitXor(x, y, num_byte, result);
#endif
}

void
BitNot(const uint8_t* x, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t step = svcntb();
    svbool_t predicate = svwhilelt_b8(i, num_byte);
    do {
        svuint8_t x_vec = svld1_u8(predicate, x + i);
        svuint8_t result_vec = svnot_u8_z(predicate, x_vec);
        svst1_u8(predicate, result + i, result_vec);

        i += step;
        predicate = svwhilelt_b8(i, num_byte);
    } while (svptest_first(svptrue_b8(), predicate));
#else
    neon::BitNot(x, num_byte, result);
#endif
}

void
Prefetch(const void* data) {
}

void
DivScalar(const float* from, float* to, uint64_t dim, float scalar) {
#if defined(ENABLE_SVE)
    if (dim == 0) {
        return;
    }
    if (scalar == 0) {
        scalar = 1.0f;
    }
    svfloat32_t divisor = svdup_f32(scalar);
    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t values = svld1_f32(predicate, from + i);
        svfloat32_t result = svdiv_f32_z(predicate, values, divisor);
        svst1_f32(predicate, to + i, result);
        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));
#else
    neon::DivScalar(from, to, dim, scalar);
#endif
}

float
Normalize(const float* from, float* to, uint64_t dim) {
#if defined(ENABLE_SVE)
    float norm = std::sqrt(sve::FP32ComputeIP(from, from, dim));
    if (norm == 0) {
        norm = 1.0f;
    }
    sve::DivScalar(from, to, dim, norm);
    return norm;
#else
    return neon::Normalize(from, to, dim);
#endif
}

__attribute__((no_sanitize("address", "undefined"))) void
PQFastScanLookUp32(const uint8_t* RESTRICT lookup_table,
                   const uint8_t* RESTRICT codes,
                   uint64_t pq_dim,
                   int32_t* RESTRICT result) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t total_bytes = pq_dim * 16;
    auto step = svcntb();

    const svuint8_t mask_low = svdup_u8(0x0F);
    const svuint16_t mask_low16 = svdup_u16(0x00FF);

    svuint16_t accum0 = svdup_u16(0);
    svuint16_t accum1 = svdup_u16(0);
    svuint16_t accum2 = svdup_u16(0);
    svuint16_t accum3 = svdup_u16(0);

    uint8_t offsets_data[svcntb()];
    for (uint64_t c = 0; c < svcntb() / 16; ++c) std::memset(offsets_data + c * 16, c * 16, 16);

    const svuint8_t index_offsets = svld1_u8(svptrue_b8(), offsets_data);

    svbool_t predicate = svwhilelt_b8(i, total_bytes);
    do {
        svuint8_t table_data = svld1_u8(predicate, lookup_table + i);
        svuint8_t code_data = svld1_u8(predicate, codes + i);

        svuint8_t low_nibbles = svand_u8_z(predicate, code_data, mask_low);
        svuint8_t high_nibbles = svlsr_n_u8_z(predicate, code_data, 4);

        svuint8_t adjusted_low_indices = svadd_u8_z(predicate, low_nibbles, index_offsets);
        svuint8_t adjusted_high_indices = svadd_u8_z(predicate, high_nibbles, index_offsets);

        svuint8_t low_values = svtbl_u8(table_data, adjusted_low_indices);
        svuint8_t high_values = svtbl_u8(table_data, adjusted_high_indices);

        svbool_t predicate_u16 = svwhilelt_b16(i / 2, total_bytes / 2);

        accum0 =
            svadd_u16_m(predicate_u16,
                        accum0,
                        svand_u16_z(predicate_u16, svreinterpret_u16_u8(low_values), mask_low16));
        accum1 = svadd_u16_m(predicate_u16,
                             accum1,
                             svlsr_n_u16_z(predicate_u16, svreinterpret_u16_u8(low_values), 8));
        accum2 =
            svadd_u16_m(predicate_u16,
                        accum2,
                        svand_u16_z(predicate_u16, svreinterpret_u16_u8(high_values), mask_low16));
        accum3 = svadd_u16_m(predicate_u16,
                             accum3,
                             svlsr_n_u16_z(predicate_u16, svreinterpret_u16_u8(high_values), 8));

        i += step;
        predicate = svwhilelt_b8(i, total_bytes);
    } while (svptest_first(svptrue_b8(), predicate));

    uint16_t temp[svcntb() / 2];

    // Segment 0
    svst1_u16(svptrue_b16(), temp, accum0);
    for (int j = 0; j < 8; ++j)
        for (int k = 0; k < svcntb() / 16; k++) result[0 * 8 + j] += temp[j + 8 * (k)];

    // Segment 1
    svst1_u16(svptrue_b16(), temp, accum1);
    for (int j = 0; j < 8; ++j)
        for (int k = 0; k < svcntb() / 16; k++) result[1 * 8 + j] += temp[j + 8 * k];

    // Segment 2
    svst1_u16(svptrue_b16(), temp, accum2);
    for (int j = 0; j < 8; ++j)
        for (int k = 0; k < svcntb() / 16; k++) result[2 * 8 + j] += temp[j + 8 * k];

    // Segment 3
    svst1_u16(svptrue_b16(), temp, accum3);
    for (int j = 0; j < 8; ++j)
        for (int k = 0; k < svcntb() / 16; k++) result[3 * 8 + j] += temp[j + 8 * k];

#else
    neon::PQFastScanLookUp32(lookup_table, codes, pq_dim, result);
#endif
}

void
KacsWalk(float* data, uint64_t len) {
#if defined(ENABLE_SVE)
    uint64_t n = len / 2;
    uint64_t offset = (len % 2) + n;
    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t predicate = svwhilelt_b32(i, n);
    do {
        svfloat32_t vec1 = svld1_f32(predicate, data + i);
        svfloat32_t vec2 = svld1_f32(predicate, data + i + offset);
        svfloat32_t sum_vec = svadd_f32_z(predicate, vec1, vec2);
        svfloat32_t diff_vec = svsub_f32_z(predicate, vec1, vec2);
        svst1_f32(predicate, data + i, sum_vec);
        svst1_f32(predicate, data + i + offset, diff_vec);
        i += step;
        predicate = svwhilelt_b32(i, n);
    } while (svptest_first(svptrue_b32(), predicate));

    if (len % 2 != 0) {
        data[n] *= std::sqrt(2.0F);
    }
#else
    neon::KacsWalk(data, len);
#endif
}

void
FlipSign(const uint8_t* flip, float* data, uint64_t dim) {
#if defined(ENABLE_SVE)
    auto predicate_array = std::make_unique<uint8_t[]>(dim);
    const uint64_t num_bytes = dim / 8;
    for (uint64_t j = 0; j < num_bytes; ++j) {
        memcpy(&predicate_array[j * 8], g_bit_lookup_table[flip[j]].data(), 8);
    }
    if (dim % 8 != 0) {
        const uint64_t remaining_bits = dim % 8;
        memcpy(&predicate_array[num_bytes * 8],
               g_bit_lookup_table[flip[num_bytes]].data(),
               remaining_bits);
    }

    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svuint32_t predicate_values = svld1ub_u32(predicate, &predicate_array[i]);
        svbool_t bit_mask = svcmpne_n_u32(predicate, predicate_values, 0);

        svfloat32_t data_vec = svld1_f32(predicate, data + i);
        svfloat32_t result_vec = svneg_f32_m(data_vec, bit_mask, data_vec);
        svst1_f32(predicate, data + i, result_vec);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));
#else
    neon::FlipSign(flip, data, dim);
#endif
}

void
VecRescale(float* data, uint64_t dim, float val) {
#if defined(ENABLE_SVE)
    svfloat32_t scale = svdup_f32(val);
    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t data_vec = svld1_f32(predicate, data + i);
        svfloat32_t result = svmul_f32_z(predicate, data_vec, scale);
        svst1_f32(predicate, data + i, result);
        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));
#else
    neon::VecRescale(data, dim, val);
#endif
}

void
RotateOp(float* data, int idx, int dim_, int step) {
#if defined(ENABLE_SVE)
    for (int i = idx; i < dim_; i += 2 * step) {
        uint64_t j = 0;
        const uint64_t sve_step = svcntw();
        svbool_t predicate = svwhilelt_b32(j, (uint64_t)step);
        do {
            svfloat32_t x = svld1_f32(predicate, data + i + j);
            svfloat32_t y = svld1_f32(predicate, data + i + j + step);
            svst1_f32(predicate, data + i + j, svadd_f32_z(predicate, x, y));
            svst1_f32(predicate, data + i + j + step, svsub_f32_z(predicate, x, y));
            j += sve_step;
            predicate = svwhilelt_b32(j, (uint64_t)step);
        } while (svptest_first(svptrue_b32(), predicate));
    }
#else
    neon::RotateOp(data, idx, dim_, step);
#endif
}

void
FHTRotate(float* data, uint64_t dim_) {
#if defined(ENABLE_SVE)
    uint64_t n = dim_;
    uint64_t step = 1;
    while (step < n) {
        sve::RotateOp(data, 0, dim_, step);
        step *= 2;
    }
#else
    neon::FHTRotate(data, dim_);
#endif
}

float
NormalizeWithCentroid(const float* from, const float* centroid, float* to, uint64_t dim) {
#if defined(ENABLE_SVE)
    if (dim == 0) {
        return 1.0f;
    }

    svfloat32_t sum = svdup_f32(0.0f);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t f_vec = svld1_f32(predicate, from + i);
        svfloat32_t c_vec = svld1_f32(predicate, centroid + i);
        svfloat32_t diff = svsub_f32_z(predicate, f_vec, c_vec);
        sum = svmla_f32_m(predicate, sum, diff, diff);
        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    float norm_sq = svaddv_f32(svptrue_b32(), sum);
    float norm = 0;
    if (norm_sq < 1e-5f) {
        norm = 1.0f;
    } else {
        norm = std::sqrt(norm_sq);
    }

    svfloat32_t normVec = svdup_f32(norm);
    i = 0;
    predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t f_vec = svld1_f32(predicate, from + i);
        svfloat32_t c_vec = svld1_f32(predicate, centroid + i);
        svfloat32_t diff = svsub_f32_z(predicate, f_vec, c_vec);
        svfloat32_t result = svdiv_f32_z(predicate, diff, normVec);
        svst1_f32(predicate, to + i, result);
        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return norm;
#else
    return neon::NormalizeWithCentroid(from, centroid, to, dim);
#endif
}

void
InverseNormalizeWithCentroid(
    const float* from, const float* centroid, float* to, uint64_t dim, float norm) {
#if defined(ENABLE_SVE)
    if (dim == 0) {
        return;
    }

    svfloat32_t normVec = svdup_f32(norm);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t f_vec = svld1_f32(predicate, from + i);
        svfloat32_t c_vec = svld1_f32(predicate, centroid + i);
        svfloat32_t result = svmad_f32_z(predicate, f_vec, normVec, c_vec);
        svst1_f32(predicate, to + i, result);
        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));
#else
    neon::InverseNormalizeWithCentroid(from, centroid, to, dim, norm);
#endif
}

}  // namespace vsag::sve
