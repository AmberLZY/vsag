
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

#pragma once

#include <cstdint>

#include "simd_marco.h"
namespace vsag {

#define DECLARE_FP32_FUNCTIONS(ns)                                                            \
    namespace ns {                                                                            \
    float                                                                                     \
    FP32ComputeIP(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim);    \
    float                                                                                     \
    FP32ComputeL2Sqr(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim); \
    void                                                                                      \
    FP32SparseAccumulate(float* RESTRICT dists,                                               \
                         const uint16_t* RESTRICT ids,                                        \
                         const float* RESTRICT vals,                                          \
                         float query_val,                                                     \
                         uint32_t num);                                                       \
    void                                                                                      \
    FP32ComputeIPBatch4(const float* RESTRICT query,                                          \
                        uint64_t dim,                                                         \
                        const float* RESTRICT codes1,                                         \
                        const float* RESTRICT codes2,                                         \
                        const float* RESTRICT codes3,                                         \
                        const float* RESTRICT codes4,                                         \
                        float& result1,                                                       \
                        float& result2,                                                       \
                        float& result3,                                                       \
                        float& result4);                                                      \
    void                                                                                      \
    FP32ComputeL2SqrBatch4(const float* RESTRICT query,                                       \
                           uint64_t dim,                                                      \
                           const float* RESTRICT codes1,                                      \
                           const float* RESTRICT codes2,                                      \
                           const float* RESTRICT codes3,                                      \
                           const float* RESTRICT codes4,                                      \
                           float& result1,                                                    \
                           float& result2,                                                    \
                           float& result3,                                                    \
                           float& result4);                                                   \
    void                                                                                      \
    FP32Sub(const float* x, const float* y, float* z, uint64_t dim);                          \
    void                                                                                      \
    FP32Add(const float* x, const float* y, float* z, uint64_t dim);                          \
    void                                                                                      \
    FP32Mul(const float* x, const float* y, float* z, uint64_t dim);                          \
    void                                                                                      \
    FP32Div(const float* x, const float* y, float* z, uint64_t dim);                          \
    float                                                                                     \
    FP32ReduceAdd(const float* x, uint64_t dim);                                              \
    }  // namespace ns

DECLARE_FP32_FUNCTIONS(generic)
DECLARE_FP32_FUNCTIONS(sse)
DECLARE_FP32_FUNCTIONS(avx)
DECLARE_FP32_FUNCTIONS(avx2)
DECLARE_FP32_FUNCTIONS(avx512)
DECLARE_FP32_FUNCTIONS(neon)
DECLARE_FP32_FUNCTIONS(sve)
#undef DECLARE_FP32_FUNCTIONS

using FP32ComputeType = float (*)(const float* RESTRICT query,
                                  const float* RESTRICT codes,
                                  uint64_t dim);
extern FP32ComputeType FP32ComputeIP;
extern FP32ComputeType FP32ComputeL2Sqr;

using FP32SparseAccumulateType = void (*)(float* RESTRICT dists,
                                          const uint16_t* RESTRICT ids,
                                          const float* RESTRICT vals,
                                          float query_val,
                                          uint32_t num);
extern FP32SparseAccumulateType FP32SparseAccumulate;

using FP32ComputeBatch4Type = void (*)(const float* RESTRICT query,
                                       uint64_t dim,
                                       const float* RESTRICT codes1,
                                       const float* RESTRICT codes2,
                                       const float* RESTRICT codes3,
                                       const float* RESTRICT codes4,
                                       float& result1,
                                       float& result2,
                                       float& result3,
                                       float& result4);
extern FP32ComputeBatch4Type FP32ComputeIPBatch4;
extern FP32ComputeBatch4Type FP32ComputeL2SqrBatch4;

#define DECLARE_FP32_BATCH8_FUNCTIONS(ns)                                         \
    namespace ns {                                                                \
    void                                                                          \
    FP32ComputeIPBatch8(const float* RESTRICT query,                              \
                        uint64_t dim,                                             \
                        const float* RESTRICT c1, const float* RESTRICT c2,       \
                        const float* RESTRICT c3, const float* RESTRICT c4,       \
                        const float* RESTRICT c5, const float* RESTRICT c6,       \
                        const float* RESTRICT c7, const float* RESTRICT c8,       \
                        float& r1, float& r2, float& r3, float& r4,              \
                        float& r5, float& r6, float& r7, float& r8);             \
    void                                                                          \
    FP32ComputeL2SqrBatch8(const float* RESTRICT query,                           \
                           uint64_t dim,                                          \
                           const float* RESTRICT c1, const float* RESTRICT c2,    \
                           const float* RESTRICT c3, const float* RESTRICT c4,    \
                           const float* RESTRICT c5, const float* RESTRICT c6,    \
                           const float* RESTRICT c7, const float* RESTRICT c8,    \
                           float& r1, float& r2, float& r3, float& r4,           \
                           float& r5, float& r6, float& r7, float& r8);          \
    } /* namespace ns */

DECLARE_FP32_BATCH8_FUNCTIONS(generic)
DECLARE_FP32_BATCH8_FUNCTIONS(sse)
DECLARE_FP32_BATCH8_FUNCTIONS(avx)
DECLARE_FP32_BATCH8_FUNCTIONS(avx2)
DECLARE_FP32_BATCH8_FUNCTIONS(avx512)
DECLARE_FP32_BATCH8_FUNCTIONS(neon)
DECLARE_FP32_BATCH8_FUNCTIONS(sve)
#undef DECLARE_FP32_BATCH8_FUNCTIONS

using FP32ComputeBatch8Type = void (*)(const float* RESTRICT query,
                                       uint64_t dim,
                                       const float* RESTRICT c1, const float* RESTRICT c2,
                                       const float* RESTRICT c3, const float* RESTRICT c4,
                                       const float* RESTRICT c5, const float* RESTRICT c6,
                                       const float* RESTRICT c7, const float* RESTRICT c8,
                                       float& r1, float& r2, float& r3, float& r4,
                                       float& r5, float& r6, float& r7, float& r8);
extern FP32ComputeBatch8Type FP32ComputeIPBatch8;
extern FP32ComputeBatch8Type FP32ComputeL2SqrBatch8;

#define DECLARE_FP32_BATCH16_FUNCTIONS(ns)                                              \
    namespace ns {                                                                      \
    void                                                                                \
    FP32ComputeIPBatch16(const float* RESTRICT query,                                   \
                         uint64_t dim,                                                  \
                         const float* RESTRICT c1, const float* RESTRICT c2,            \
                         const float* RESTRICT c3, const float* RESTRICT c4,            \
                         const float* RESTRICT c5, const float* RESTRICT c6,            \
                         const float* RESTRICT c7, const float* RESTRICT c8,            \
                         const float* RESTRICT c9, const float* RESTRICT c10,           \
                         const float* RESTRICT c11, const float* RESTRICT c12,          \
                         const float* RESTRICT c13, const float* RESTRICT c14,          \
                         const float* RESTRICT c15, const float* RESTRICT c16,          \
                         float& r1, float& r2, float& r3, float& r4,                   \
                         float& r5, float& r6, float& r7, float& r8,                   \
                         float& r9, float& r10, float& r11, float& r12,                \
                         float& r13, float& r14, float& r15, float& r16);              \
    void                                                                                \
    FP32ComputeL2SqrBatch16(const float* RESTRICT query,                                \
                            uint64_t dim,                                               \
                            const float* RESTRICT c1, const float* RESTRICT c2,         \
                            const float* RESTRICT c3, const float* RESTRICT c4,         \
                            const float* RESTRICT c5, const float* RESTRICT c6,         \
                            const float* RESTRICT c7, const float* RESTRICT c8,         \
                            const float* RESTRICT c9, const float* RESTRICT c10,        \
                            const float* RESTRICT c11, const float* RESTRICT c12,       \
                            const float* RESTRICT c13, const float* RESTRICT c14,       \
                            const float* RESTRICT c15, const float* RESTRICT c16,       \
                            float& r1, float& r2, float& r3, float& r4,                \
                            float& r5, float& r6, float& r7, float& r8,                \
                            float& r9, float& r10, float& r11, float& r12,             \
                            float& r13, float& r14, float& r15, float& r16);           \
    } /* namespace ns */

DECLARE_FP32_BATCH16_FUNCTIONS(generic)
DECLARE_FP32_BATCH16_FUNCTIONS(sse)
DECLARE_FP32_BATCH16_FUNCTIONS(avx)
DECLARE_FP32_BATCH16_FUNCTIONS(avx2)
DECLARE_FP32_BATCH16_FUNCTIONS(avx512)
DECLARE_FP32_BATCH16_FUNCTIONS(neon)
DECLARE_FP32_BATCH16_FUNCTIONS(sve)
#undef DECLARE_FP32_BATCH16_FUNCTIONS

using FP32ComputeBatch16Type = void (*)(const float* RESTRICT query,
                                        uint64_t dim,
                                        const float* RESTRICT c1, const float* RESTRICT c2,
                                        const float* RESTRICT c3, const float* RESTRICT c4,
                                        const float* RESTRICT c5, const float* RESTRICT c6,
                                        const float* RESTRICT c7, const float* RESTRICT c8,
                                        const float* RESTRICT c9, const float* RESTRICT c10,
                                        const float* RESTRICT c11, const float* RESTRICT c12,
                                        const float* RESTRICT c13, const float* RESTRICT c14,
                                        const float* RESTRICT c15, const float* RESTRICT c16,
                                        float& r1, float& r2, float& r3, float& r4,
                                        float& r5, float& r6, float& r7, float& r8,
                                        float& r9, float& r10, float& r11, float& r12,
                                        float& r13, float& r14, float& r15, float& r16);
extern FP32ComputeBatch16Type FP32ComputeIPBatch16;
extern FP32ComputeBatch16Type FP32ComputeL2SqrBatch16;

using FP32ArithmeticType = void (*)(const float* x, const float* y, float* z, uint64_t dim);
extern FP32ArithmeticType FP32Sub;
extern FP32ArithmeticType FP32Add;
extern FP32ArithmeticType FP32Mul;
extern FP32ArithmeticType FP32Div;

using FP32ReduceType = float (*)(const float* x, uint64_t dim);
extern FP32ReduceType FP32ReduceAdd;
}  // namespace vsag
