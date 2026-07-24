
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

#include "fp32_quantizer.h"

#include "simd/fp32_simd.h"
#include "simd/normalize.h"
#include "simd/simd.h"

namespace vsag {

template <MetricType metric>
FP32Quantizer<metric>::FP32Quantizer(int dim, Allocator* allocator)
    : Quantizer<FP32Quantizer<metric>>(dim, allocator) {
    this->code_size_ = dim * sizeof(float);
    this->query_code_size_ = this->code_size_;
    this->metric_ = metric;
}

template <MetricType metric>
FP32Quantizer<metric>::FP32Quantizer(const FP32QuantizerParamPtr& param,
                                     const IndexCommonParam& common_param)
    : FP32Quantizer<metric>(common_param.dim_, common_param.allocator_.get()) {
    this->hold_molds_ = param->hold_molds;
    if (metric == MetricType::METRIC_TYPE_COSINE && this->hold_molds_) {
        this->code_size_ += sizeof(float);
    }
}

template <MetricType metric>
FP32Quantizer<metric>::FP32Quantizer(const QuantizerParamPtr& param,
                                     const IndexCommonParam& common_param)
    : FP32Quantizer<metric>(std::dynamic_pointer_cast<FP32QuantizerParameter>(param),
                            common_param) {
}

template <MetricType metric>
bool
FP32Quantizer<metric>::TrainImpl(const float* data, uint64_t count) {
    this->is_trained_ = true;
    return true;
}

template <MetricType metric>
bool
FP32Quantizer<metric>::EncodeOneImpl(const float* data, uint8_t* codes) {
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        if (this->hold_molds_) {
            // Store the mold for cosine similarity
            const auto* data_float = reinterpret_cast<const float*>(data);
            float mold = std::sqrt(FP32ComputeIP(data_float, data_float, this->dim_));
            memcpy(codes, data, this->dim_ * sizeof(float));
            memcpy(codes + this->dim_ * sizeof(float), &mold, sizeof(float));
        } else {
            Normalize(data, reinterpret_cast<float*>(codes), this->dim_);
        }
    } else {
        memcpy(codes, data, this->code_size_);
    }
    return true;
}

template <MetricType metric>
bool
FP32Quantizer<metric>::EncodeBatchImpl(const float* data, uint8_t* codes, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        EncodeOneImpl(data + i * this->dim_, codes + i * this->code_size_);
    }
    return true;
}

template <MetricType metric>
bool
FP32Quantizer<metric>::DecodeOneImpl(const uint8_t* codes, float* data) {
    memcpy(data, codes, this->dim_ * sizeof(float));
    return true;
}

template <MetricType metric>
bool
FP32Quantizer<metric>::DecodeBatchImpl(const uint8_t* codes, float* data, uint64_t count) {
    if (this->code_size_ == this->dim_ * sizeof(float)) {
        memcpy(data, codes, count * this->code_size_);
    } else {
        for (uint64_t i = 0; i < count; ++i) {
            memcpy(data + i * this->dim_, codes + i * this->code_size_, this->dim_ * sizeof(float));
        }
    }
    return true;
}

template <MetricType metric>
float
FP32Quantizer<metric>::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) {
    if (metric == MetricType::METRIC_TYPE_IP) {
        return 1.0F - FP32ComputeIP(reinterpret_cast<const float*>(codes1),
                                    reinterpret_cast<const float*>(codes2),
                                    this->dim_);
    }
    if (metric == MetricType::METRIC_TYPE_COSINE) {
        auto similarity = FP32ComputeIP(reinterpret_cast<const float*>(codes1),
                                        reinterpret_cast<const float*>(codes2),
                                        this->dim_);
        if (this->hold_molds_) {
            const auto* mold1 = reinterpret_cast<const float*>(codes1 + this->dim_ * sizeof(float));
            const auto* mold2 = reinterpret_cast<const float*>(codes2 + this->dim_ * sizeof(float));
            similarity /= mold1[0] * mold2[0];
        }
        return 1.0F - similarity;
    }
    if (metric == MetricType::METRIC_TYPE_L2SQR) {
        return FP32ComputeL2Sqr(reinterpret_cast<const float*>(codes1),
                                reinterpret_cast<const float*>(codes2),
                                this->dim_);
    }
    return 0.0F;
}

template <MetricType metric>
void
FP32Quantizer<metric>::ScanBatchDistImpl(Computer<FP32Quantizer<metric>>& computer,
                                         uint64_t count,
                                         const uint8_t* codes,
                                         float* dists) const {
    // TODO(LHT): Optimize batch for simd
    for (uint64_t i = 0; i < count; ++i) {
        this->ComputeDistImpl(computer, codes + i * this->code_size_, dists + i);
    }
}

template <MetricType metric>
void
FP32Quantizer<metric>::ProcessQueryImpl(const float* query,
                                        Computer<FP32Quantizer<metric>>& computer) const {
    try {
        if (computer.buf_ == nullptr) {
            computer.buf_ =
                reinterpret_cast<uint8_t*>(this->allocator_->Allocate(this->query_code_size_));
        }
    } catch (const std::bad_alloc& e) {
        computer.buf_ = nullptr;
        throw VsagException(ErrorType::NO_ENOUGH_MEMORY, "bad alloc when init computer buf");
    }
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        Normalize(query, reinterpret_cast<float*>(computer.buf_), this->dim_);
    } else {
        memcpy(computer.buf_, query, this->code_size_);
    }
}

template <MetricType metric>
void
FP32Quantizer<metric>::ComputeDistImpl(Computer<FP32Quantizer<metric>>& computer,
                                       const uint8_t* codes,
                                       float* dists) const {
    if (metric == MetricType::METRIC_TYPE_IP) {
        *dists = 1.0F - FP32ComputeIP(reinterpret_cast<const float*>(codes),
                                      reinterpret_cast<const float*>(computer.buf_),
                                      this->dim_);
    } else if (metric == MetricType::METRIC_TYPE_COSINE) {
        auto similarity = FP32ComputeIP(reinterpret_cast<const float*>(codes),
                                        reinterpret_cast<const float*>(computer.buf_),
                                        this->dim_);
        if (this->hold_molds_) {
            const auto* mold = reinterpret_cast<const float*>(codes + this->dim_ * sizeof(float));
            similarity /= mold[0];
        }
        *dists = 1.0F - similarity;
    } else if (metric == MetricType::METRIC_TYPE_L2SQR) {
        *dists = FP32ComputeL2Sqr(reinterpret_cast<const float*>(codes),
                                  reinterpret_cast<const float*>(computer.buf_),
                                  this->dim_);
    } else {
        *dists = 0.0F;
    }
}

template <MetricType metric>
void
FP32Quantizer<metric>::ComputeDistsBatch4Impl(Computer<FP32Quantizer<metric>>& computer,
                                              const uint8_t* codes1,
                                              const uint8_t* codes2,
                                              const uint8_t* codes3,
                                              const uint8_t* codes4,
                                              float& dists1,
                                              float& dists2,
                                              float& dists3,
                                              float& dists4) const {
    if constexpr (metric == MetricType::METRIC_TYPE_IP) {
        FP32ComputeIPBatch4(reinterpret_cast<const float*>(computer.buf_),
                            this->dim_,
                            reinterpret_cast<const float*>(codes1),
                            reinterpret_cast<const float*>(codes2),
                            reinterpret_cast<const float*>(codes3),
                            reinterpret_cast<const float*>(codes4),
                            dists1,
                            dists2,
                            dists3,
                            dists4);
        dists1 = 1.0F - dists1;
        dists2 = 1.0F - dists2;
        dists3 = 1.0F - dists3;
        dists4 = 1.0F - dists4;
    } else if (metric == MetricType::METRIC_TYPE_COSINE) {
        FP32ComputeIPBatch4(reinterpret_cast<const float*>(computer.buf_),
                            this->dim_,
                            reinterpret_cast<const float*>(codes1),
                            reinterpret_cast<const float*>(codes2),
                            reinterpret_cast<const float*>(codes3),
                            reinterpret_cast<const float*>(codes4),
                            dists1,
                            dists2,
                            dists3,
                            dists4);
        if (this->hold_molds_) {
            const auto* mold1 = reinterpret_cast<const float*>(codes1 + this->dim_ * sizeof(float));
            const auto* mold2 = reinterpret_cast<const float*>(codes2 + this->dim_ * sizeof(float));
            const auto* mold3 = reinterpret_cast<const float*>(codes3 + this->dim_ * sizeof(float));
            const auto* mold4 = reinterpret_cast<const float*>(codes4 + this->dim_ * sizeof(float));
            dists1 /= mold1[0];
            dists2 /= mold2[0];
            dists3 /= mold3[0];
            dists4 /= mold4[0];
        }
        dists1 = 1.0F - dists1;
        dists2 = 1.0F - dists2;
        dists3 = 1.0F - dists3;
        dists4 = 1.0F - dists4;
    } else if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        FP32ComputeL2SqrBatch4(reinterpret_cast<const float*>(computer.buf_),
                               this->dim_,
                               reinterpret_cast<const float*>(codes1),
                               reinterpret_cast<const float*>(codes2),
                               reinterpret_cast<const float*>(codes3),
                               reinterpret_cast<const float*>(codes4),
                               dists1,
                               dists2,
                               dists3,
                               dists4);
    } else {
        dists1 = 0.0F;
        dists2 = 0.0F;
        dists3 = 0.0F;
        dists4 = 0.0F;
    }
}

template <MetricType metric>
void
FP32Quantizer<metric>::ComputeDistsBatch8Impl(Computer<FP32Quantizer<metric>>& computer,
                                              const uint8_t* codes1,
                                              const uint8_t* codes2,
                                              const uint8_t* codes3,
                                              const uint8_t* codes4,
                                              const uint8_t* codes5,
                                              const uint8_t* codes6,
                                              const uint8_t* codes7,
                                              const uint8_t* codes8,
                                              float& dists1,
                                              float& dists2,
                                              float& dists3,
                                              float& dists4,
                                              float& dists5,
                                              float& dists6,
                                              float& dists7,
                                              float& dists8) const {
    if constexpr (metric == MetricType::METRIC_TYPE_IP) {
        FP32ComputeIPBatch8(reinterpret_cast<const float*>(computer.buf_),
                            this->dim_,
                            reinterpret_cast<const float*>(codes1),
                            reinterpret_cast<const float*>(codes2),
                            reinterpret_cast<const float*>(codes3),
                            reinterpret_cast<const float*>(codes4),
                            reinterpret_cast<const float*>(codes5),
                            reinterpret_cast<const float*>(codes6),
                            reinterpret_cast<const float*>(codes7),
                            reinterpret_cast<const float*>(codes8),
                            dists1,
                            dists2,
                            dists3,
                            dists4,
                            dists5,
                            dists6,
                            dists7,
                            dists8);
        dists1 = 1.0F - dists1;
        dists2 = 1.0F - dists2;
        dists3 = 1.0F - dists3;
        dists4 = 1.0F - dists4;
        dists5 = 1.0F - dists5;
        dists6 = 1.0F - dists6;
        dists7 = 1.0F - dists7;
        dists8 = 1.0F - dists8;
    } else if (metric == MetricType::METRIC_TYPE_COSINE) {
        FP32ComputeIPBatch8(reinterpret_cast<const float*>(computer.buf_),
                            this->dim_,
                            reinterpret_cast<const float*>(codes1),
                            reinterpret_cast<const float*>(codes2),
                            reinterpret_cast<const float*>(codes3),
                            reinterpret_cast<const float*>(codes4),
                            reinterpret_cast<const float*>(codes5),
                            reinterpret_cast<const float*>(codes6),
                            reinterpret_cast<const float*>(codes7),
                            reinterpret_cast<const float*>(codes8),
                            dists1,
                            dists2,
                            dists3,
                            dists4,
                            dists5,
                            dists6,
                            dists7,
                            dists8);
        if (this->hold_molds_) {
            const auto* mold1 = reinterpret_cast<const float*>(codes1 + this->dim_ * sizeof(float));
            const auto* mold2 = reinterpret_cast<const float*>(codes2 + this->dim_ * sizeof(float));
            const auto* mold3 = reinterpret_cast<const float*>(codes3 + this->dim_ * sizeof(float));
            const auto* mold4 = reinterpret_cast<const float*>(codes4 + this->dim_ * sizeof(float));
            const auto* mold5 = reinterpret_cast<const float*>(codes5 + this->dim_ * sizeof(float));
            const auto* mold6 = reinterpret_cast<const float*>(codes6 + this->dim_ * sizeof(float));
            const auto* mold7 = reinterpret_cast<const float*>(codes7 + this->dim_ * sizeof(float));
            const auto* mold8 = reinterpret_cast<const float*>(codes8 + this->dim_ * sizeof(float));
            dists1 /= mold1[0];
            dists2 /= mold2[0];
            dists3 /= mold3[0];
            dists4 /= mold4[0];
            dists5 /= mold5[0];
            dists6 /= mold6[0];
            dists7 /= mold7[0];
            dists8 /= mold8[0];
        }
        dists1 = 1.0F - dists1;
        dists2 = 1.0F - dists2;
        dists3 = 1.0F - dists3;
        dists4 = 1.0F - dists4;
        dists5 = 1.0F - dists5;
        dists6 = 1.0F - dists6;
        dists7 = 1.0F - dists7;
        dists8 = 1.0F - dists8;
    } else if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        FP32ComputeL2SqrBatch8(reinterpret_cast<const float*>(computer.buf_),
                               this->dim_,
                               reinterpret_cast<const float*>(codes1),
                               reinterpret_cast<const float*>(codes2),
                               reinterpret_cast<const float*>(codes3),
                               reinterpret_cast<const float*>(codes4),
                               reinterpret_cast<const float*>(codes5),
                               reinterpret_cast<const float*>(codes6),
                               reinterpret_cast<const float*>(codes7),
                               reinterpret_cast<const float*>(codes8),
                               dists1,
                               dists2,
                               dists3,
                               dists4,
                               dists5,
                               dists6,
                               dists7,
                               dists8);
    } else {
        dists1 = 0.0F;
        dists2 = 0.0F;
        dists3 = 0.0F;
        dists4 = 0.0F;
        dists5 = 0.0F;
        dists6 = 0.0F;
        dists7 = 0.0F;
        dists8 = 0.0F;
    }
}

template <MetricType metric>
void
FP32Quantizer<metric>::ComputeDistsBatch16Impl(Computer<FP32Quantizer<metric>>& computer,
                                               const uint8_t* codes1,
                                               const uint8_t* codes2,
                                               const uint8_t* codes3,
                                               const uint8_t* codes4,
                                               const uint8_t* codes5,
                                               const uint8_t* codes6,
                                               const uint8_t* codes7,
                                               const uint8_t* codes8,
                                               const uint8_t* codes9,
                                               const uint8_t* codes10,
                                               const uint8_t* codes11,
                                               const uint8_t* codes12,
                                               const uint8_t* codes13,
                                               const uint8_t* codes14,
                                               const uint8_t* codes15,
                                               const uint8_t* codes16,
                                               float& dists1,
                                               float& dists2,
                                               float& dists3,
                                               float& dists4,
                                               float& dists5,
                                               float& dists6,
                                               float& dists7,
                                               float& dists8,
                                               float& dists9,
                                               float& dists10,
                                               float& dists11,
                                               float& dists12,
                                               float& dists13,
                                               float& dists14,
                                               float& dists15,
                                               float& dists16) const {
    if constexpr (metric == MetricType::METRIC_TYPE_IP) {
        FP32ComputeIPBatch16(reinterpret_cast<const float*>(computer.buf_),
                             this->dim_,
                             reinterpret_cast<const float*>(codes1),
                             reinterpret_cast<const float*>(codes2),
                             reinterpret_cast<const float*>(codes3),
                             reinterpret_cast<const float*>(codes4),
                             reinterpret_cast<const float*>(codes5),
                             reinterpret_cast<const float*>(codes6),
                             reinterpret_cast<const float*>(codes7),
                             reinterpret_cast<const float*>(codes8),
                             reinterpret_cast<const float*>(codes9),
                             reinterpret_cast<const float*>(codes10),
                             reinterpret_cast<const float*>(codes11),
                             reinterpret_cast<const float*>(codes12),
                             reinterpret_cast<const float*>(codes13),
                             reinterpret_cast<const float*>(codes14),
                             reinterpret_cast<const float*>(codes15),
                             reinterpret_cast<const float*>(codes16),
                             dists1,
                             dists2,
                             dists3,
                             dists4,
                             dists5,
                             dists6,
                             dists7,
                             dists8,
                             dists9,
                             dists10,
                             dists11,
                             dists12,
                             dists13,
                             dists14,
                             dists15,
                             dists16);
        dists1 = 1.0F - dists1;
        dists2 = 1.0F - dists2;
        dists3 = 1.0F - dists3;
        dists4 = 1.0F - dists4;
        dists5 = 1.0F - dists5;
        dists6 = 1.0F - dists6;
        dists7 = 1.0F - dists7;
        dists8 = 1.0F - dists8;
        dists9 = 1.0F - dists9;
        dists10 = 1.0F - dists10;
        dists11 = 1.0F - dists11;
        dists12 = 1.0F - dists12;
        dists13 = 1.0F - dists13;
        dists14 = 1.0F - dists14;
        dists15 = 1.0F - dists15;
        dists16 = 1.0F - dists16;
    } else if (metric == MetricType::METRIC_TYPE_COSINE) {
        FP32ComputeIPBatch16(reinterpret_cast<const float*>(computer.buf_),
                             this->dim_,
                             reinterpret_cast<const float*>(codes1),
                             reinterpret_cast<const float*>(codes2),
                             reinterpret_cast<const float*>(codes3),
                             reinterpret_cast<const float*>(codes4),
                             reinterpret_cast<const float*>(codes5),
                             reinterpret_cast<const float*>(codes6),
                             reinterpret_cast<const float*>(codes7),
                             reinterpret_cast<const float*>(codes8),
                             reinterpret_cast<const float*>(codes9),
                             reinterpret_cast<const float*>(codes10),
                             reinterpret_cast<const float*>(codes11),
                             reinterpret_cast<const float*>(codes12),
                             reinterpret_cast<const float*>(codes13),
                             reinterpret_cast<const float*>(codes14),
                             reinterpret_cast<const float*>(codes15),
                             reinterpret_cast<const float*>(codes16),
                             dists1,
                             dists2,
                             dists3,
                             dists4,
                             dists5,
                             dists6,
                             dists7,
                             dists8,
                             dists9,
                             dists10,
                             dists11,
                             dists12,
                             dists13,
                             dists14,
                             dists15,
                             dists16);
        if (this->hold_molds_) {
            const auto* mold1 = reinterpret_cast<const float*>(codes1 + this->dim_ * sizeof(float));
            const auto* mold2 = reinterpret_cast<const float*>(codes2 + this->dim_ * sizeof(float));
            const auto* mold3 = reinterpret_cast<const float*>(codes3 + this->dim_ * sizeof(float));
            const auto* mold4 = reinterpret_cast<const float*>(codes4 + this->dim_ * sizeof(float));
            const auto* mold5 = reinterpret_cast<const float*>(codes5 + this->dim_ * sizeof(float));
            const auto* mold6 = reinterpret_cast<const float*>(codes6 + this->dim_ * sizeof(float));
            const auto* mold7 = reinterpret_cast<const float*>(codes7 + this->dim_ * sizeof(float));
            const auto* mold8 = reinterpret_cast<const float*>(codes8 + this->dim_ * sizeof(float));
            const auto* mold9 = reinterpret_cast<const float*>(codes9 + this->dim_ * sizeof(float));
            const auto* mold10 = reinterpret_cast<const float*>(codes10 + this->dim_ * sizeof(float));
            const auto* mold11 = reinterpret_cast<const float*>(codes11 + this->dim_ * sizeof(float));
            const auto* mold12 = reinterpret_cast<const float*>(codes12 + this->dim_ * sizeof(float));
            const auto* mold13 = reinterpret_cast<const float*>(codes13 + this->dim_ * sizeof(float));
            const auto* mold14 = reinterpret_cast<const float*>(codes14 + this->dim_ * sizeof(float));
            const auto* mold15 = reinterpret_cast<const float*>(codes15 + this->dim_ * sizeof(float));
            const auto* mold16 = reinterpret_cast<const float*>(codes16 + this->dim_ * sizeof(float));
            dists1 /= mold1[0];
            dists2 /= mold2[0];
            dists3 /= mold3[0];
            dists4 /= mold4[0];
            dists5 /= mold5[0];
            dists6 /= mold6[0];
            dists7 /= mold7[0];
            dists8 /= mold8[0];
            dists9 /= mold9[0];
            dists10 /= mold10[0];
            dists11 /= mold11[0];
            dists12 /= mold12[0];
            dists13 /= mold13[0];
            dists14 /= mold14[0];
            dists15 /= mold15[0];
            dists16 /= mold16[0];
        }
        dists1 = 1.0F - dists1;
        dists2 = 1.0F - dists2;
        dists3 = 1.0F - dists3;
        dists4 = 1.0F - dists4;
        dists5 = 1.0F - dists5;
        dists6 = 1.0F - dists6;
        dists7 = 1.0F - dists7;
        dists8 = 1.0F - dists8;
        dists9 = 1.0F - dists9;
        dists10 = 1.0F - dists10;
        dists11 = 1.0F - dists11;
        dists12 = 1.0F - dists12;
        dists13 = 1.0F - dists13;
        dists14 = 1.0F - dists14;
        dists15 = 1.0F - dists15;
        dists16 = 1.0F - dists16;
    } else if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        FP32ComputeL2SqrBatch16(reinterpret_cast<const float*>(computer.buf_),
                                this->dim_,
                                reinterpret_cast<const float*>(codes1),
                                reinterpret_cast<const float*>(codes2),
                                reinterpret_cast<const float*>(codes3),
                                reinterpret_cast<const float*>(codes4),
                                reinterpret_cast<const float*>(codes5),
                                reinterpret_cast<const float*>(codes6),
                                reinterpret_cast<const float*>(codes7),
                                reinterpret_cast<const float*>(codes8),
                                reinterpret_cast<const float*>(codes9),
                                reinterpret_cast<const float*>(codes10),
                                reinterpret_cast<const float*>(codes11),
                                reinterpret_cast<const float*>(codes12),
                                reinterpret_cast<const float*>(codes13),
                                reinterpret_cast<const float*>(codes14),
                                reinterpret_cast<const float*>(codes15),
                                reinterpret_cast<const float*>(codes16),
                                dists1,
                                dists2,
                                dists3,
                                dists4,
                                dists5,
                                dists6,
                                dists7,
                                dists8,
                                dists9,
                                dists10,
                                dists11,
                                dists12,
                                dists13,
                                dists14,
                                dists15,
                                dists16);
    } else {
        dists1 = 0.0F;
        dists2 = 0.0F;
        dists3 = 0.0F;
        dists4 = 0.0F;
        dists5 = 0.0F;
        dists6 = 0.0F;
        dists7 = 0.0F;
        dists8 = 0.0F;
        dists9 = 0.0F;
        dists10 = 0.0F;
        dists11 = 0.0F;
        dists12 = 0.0F;
        dists13 = 0.0F;
        dists14 = 0.0F;
        dists15 = 0.0F;
        dists16 = 0.0F;
    }
}

template <MetricType metric>
void
FP32Quantizer<metric>::ReleaseComputerImpl(Computer<FP32Quantizer<metric>>& computer) const {
    this->allocator_->Deallocate(computer.buf_);
}

TEMPLATE_QUANTIZER(FP32Quantizer);
}  // namespace vsag
