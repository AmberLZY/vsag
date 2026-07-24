
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

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>

#include "common.h"
#include "flatten_interface.h"
#include "index_common_param.h"
#include "io/common/basic_io.h"
#include "io/memory_block_io/memory_block_io.h"
#include "quantization/quantizer.h"
#include "simd/fp32_simd.h"
#include "query_context.h"
#include "utils/byte_buffer.h"
#include "utils/timer.h"

namespace vsag {
/*
* thread unsafe
*/
template <typename QuantTmpl, typename IOTmpl>
class FlattenDataCell : public FlattenInterface {
public:
    FlattenDataCell() = default;

    explicit FlattenDataCell(const QuantizerParamPtr& quantization_param,
                             const IOParamPtr& io_param,
                             const IndexCommonParam& common_param);

    void
    Query(float* result_dists,
          const ComputerInterfacePtr& computer,
          const InnerIdType* idx,
          InnerIdType id_count,
          QueryContext* ctx = nullptr) override {
        auto comp = static_cast<Computer<QuantTmpl>*>(computer.get());
        this->query(result_dists, comp, idx, id_count, ctx);
    }

    ComputerInterfacePtr
    FactoryComputer(const void* query) override {
        return this->factory_computer(static_cast<const float*>(query));
    }

    float
    ComputePairVectors(InnerIdType id1, InnerIdType id2) override;

    void
    Train(const void* data, uint64_t count) override;

    void
    InsertVector(const void* vector, InnerIdType idx) override;

    bool
    UpdateVector(const void* vector,
                 InnerIdType idx = std::numeric_limits<InnerIdType>::max()) override;

    void
    BatchInsertVector(const void* vectors, InnerIdType count, InnerIdType* idx_vec) override;

    bool
    Decode(const uint8_t* codes, float* data) override {
        return this->quantizer_->DecodeOne(codes, data);
    }

    bool
    Encode(const float* data, uint8_t* codes) override {
        return this->quantizer_->EncodeOne(data, codes);
    }

    void
    Resize(InnerIdType new_capacity) override {
        if (new_capacity <= this->max_capacity_) {
            return;
        }
        uint64_t io_size = static_cast<uint64_t>(new_capacity) * static_cast<uint64_t>(code_size_);
        this->io_->Resize(io_size);
        this->max_capacity_ = new_capacity;
    }

    void
    Prefetch(InnerIdType id) override {
        io_->Prefetch(id * code_size_, 64);
    };

    void
    ExportModel(const FlattenInterfacePtr& other) const override {
        std::stringstream ss;
        IOStreamWriter writer(ss);
        this->quantizer_->Serialize(writer);
        ss.seekg(0, std::ios::beg);
        IOStreamReader reader(ss);
        auto ptr = std::dynamic_pointer_cast<FlattenDataCell<QuantTmpl, IOTmpl>>(other);
        if (ptr == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "Export model's flatten datacell failed");
        }
        ptr->quantizer_->Deserialize(reader);
    }

    void
    MergeOther(const FlattenInterfacePtr& other, InnerIdType bias) override;

    void
    Move(InnerIdType from, InnerIdType to) override;

    void
    ShrinkToFit(InnerIdType capacity) override {
        uint64_t io_size = static_cast<uint64_t>(capacity) * static_cast<uint64_t>(code_size_);
        this->io_->Shrink(io_size);
        this->max_capacity_ = capacity;
    }

    [[nodiscard]] std::string
    GetQuantizerName() override;

    [[nodiscard]] MetricType
    GetMetricType() override;

    [[nodiscard]] const uint8_t*
    GetCodesById(InnerIdType id, bool& need_release) const override;

    void
    Release(const uint8_t* data) const override;

    [[nodiscard]] bool
    InMemory() const override;

    bool
    HoldMolds() const override;

    bool
    GetCodesById(InnerIdType id, uint8_t* codes) const override;

    void
    Serialize(StreamWriter& writer) override;

    void
    Deserialize(lvalue_or_rvalue<StreamReader> reader) override;

    inline void
    SetQuantizer(std::shared_ptr<Quantizer<QuantTmpl>> quantizer) {
        this->quantizer_ = quantizer;
        this->code_size_ = quantizer_->GetCodeSize();
    }

    inline void
    SetIO(std::shared_ptr<BasicIO<IOTmpl>> io) {
        this->io_ = io;
    }

    void
    InitIO(const IOParamPtr& io_param) override {
        this->io_->InitIO(io_param);
    }

    IndexCommonParam
    ExportCommonParam() override {
        return common_param_;
    }

    uint64_t
    GetMemoryUsage() const override;

public:
    IndexCommonParam common_param_;

    std::shared_ptr<Quantizer<QuantTmpl>> quantizer_{nullptr};
    std::shared_ptr<BasicIO<IOTmpl>> io_{nullptr};

    Allocator* const allocator_{nullptr};

private:
    inline void
    query(float* result_dists,
          Computer<QuantTmpl>* computer,
          const InnerIdType* idx,
          InnerIdType id_count,
          QueryContext* ctx);

    ComputerInterfacePtr
    factory_computer(const float* query) {
        auto computer = this->quantizer_->FactoryComputer();
        computer->SetQuery(query);
        return computer;
    }
};

template <typename QuantTmpl, typename IOTmpl>
void
FlattenDataCell<QuantTmpl, IOTmpl>::Release(const uint8_t* data) const {
    this->io_->Release(data);
}

template <typename QuantTmpl, typename IOTmpl>
bool
FlattenDataCell<QuantTmpl, IOTmpl>::HoldMolds() const {
    return this->quantizer_->HoldMolds();
}

template <typename QuantTmpl, typename IOTmpl>
FlattenDataCell<QuantTmpl, IOTmpl>::FlattenDataCell(const QuantizerParamPtr& quantization_param,
                                                    const IOParamPtr& io_param,
                                                    const IndexCommonParam& common_param)
    : allocator_(common_param.allocator_.get()) {
    this->common_param_ = common_param;
    this->quantizer_ = std::make_shared<QuantTmpl>(quantization_param, common_param);
    this->io_ = std::make_shared<IOTmpl>(io_param, common_param);
    this->code_size_ = quantizer_->GetCodeSize();
}

template <typename QuantTmpl, typename IOTmpl>
void
FlattenDataCell<QuantTmpl, IOTmpl>::Train(const void* data, uint64_t count) {
    if (this->quantizer_) {
        this->quantizer_->Train(static_cast<const float*>(data), count);
    }
}

template <typename QuantTmpl, typename IOTmpl>
void
FlattenDataCell<QuantTmpl, IOTmpl>::InsertVector(const void* vector, InnerIdType idx) {
    {
        std::lock_guard lock(mutex_);
        if (idx == std::numeric_limits<InnerIdType>::max()) {
            idx = total_count_;
            ++total_count_;
        } else {
            total_count_ = std::max(total_count_, idx + 1);
        }
    }
    ByteBuffer codes(static_cast<uint64_t>(code_size_), allocator_);
    quantizer_->EncodeOne(static_cast<const float*>(vector), codes.data);
    io_->Write(
        codes.data, code_size_, static_cast<uint64_t>(idx) * static_cast<uint64_t>(code_size_));
}

template <typename QuantTmpl, typename IOTmpl>
bool
FlattenDataCell<QuantTmpl, IOTmpl>::UpdateVector(const void* vector, InnerIdType idx) {
    if (idx >= total_count_) {
        return false;
    }
    std::lock_guard lock(mutex_);
    ByteBuffer codes(static_cast<uint64_t>(code_size_), allocator_);
    quantizer_->EncodeOne(static_cast<const float*>(vector), codes.data);
    io_->Write(
        codes.data, code_size_, static_cast<uint64_t>(idx) * static_cast<uint64_t>(code_size_));
    return true;
}

template <typename QuantTmpl, typename IOTmpl>
void
FlattenDataCell<QuantTmpl, IOTmpl>::BatchInsertVector(const void* vectors,
                                                      InnerIdType count,
                                                      InnerIdType* idx_vec) {
    if (idx_vec == nullptr) {
        ByteBuffer codes(static_cast<uint64_t>(count) * static_cast<uint64_t>(code_size_),
                         allocator_);
        quantizer_->EncodeBatch(static_cast<const float*>(vectors), codes.data, count);
        uint64_t cur_count;
        {
            std::lock_guard lock(mutex_);
            cur_count = total_count_;
            total_count_ += count;
        }
        io_->Write(codes.data,
                   static_cast<uint64_t>(count) * static_cast<uint64_t>(code_size_),
                   cur_count * static_cast<uint64_t>(code_size_));
    } else {
        auto dim = quantizer_->GetDim();
        for (int64_t i = 0; i < count; ++i) {
            this->InsertVector(static_cast<const float*>(vectors) + dim * i, idx_vec[i]);
        }
    }
}

template <typename QuantTmpl, typename IOTmpl>
std::string
FlattenDataCell<QuantTmpl, IOTmpl>::GetQuantizerName() {
    return this->quantizer_->Name();
}

template <typename QuantTmpl, typename IOTmpl>
MetricType
FlattenDataCell<QuantTmpl, IOTmpl>::GetMetricType() {
    return this->quantizer_->Metric();
}

template <typename QuantTmpl, typename IOTmpl>
bool
FlattenDataCell<QuantTmpl, IOTmpl>::InMemory() const {
    return IOTmpl::InMemory;
}

template <typename QuantTmpl, typename IOTmpl>
void
FlattenDataCell<QuantTmpl, IOTmpl>::query(float* result_dists,
                                          Computer<QuantTmpl>* computer,
                                          const InnerIdType* idx,
                                          InnerIdType id_count,
                                          QueryContext* ctx) {
    Allocator* search_alloc = select_query_allocator(ctx, allocator_);

    for (uint32_t i = 0; i < this->prefetch_stride_code_ and i < id_count; i++) {
        this->io_->Prefetch(static_cast<uint64_t>(idx[i]) * static_cast<uint64_t>(code_size_),
                            this->prefetch_depth_code_ * 64);
    }
    if constexpr (not IOTmpl::InMemory) {
        if (id_count > 1) {
            ByteBuffer codes(
                static_cast<uint64_t>(id_count) * static_cast<uint64_t>(this->code_size_),
                search_alloc);
            Vector<uint64_t> sizes(id_count, this->code_size_, search_alloc);
            Vector<uint64_t> offsets(id_count, this->code_size_, search_alloc);
            for (int64_t i = 0; i < id_count; ++i) {
                offsets[i] = static_cast<uint64_t>(idx[i]) * this->code_size_;
            }

            double io_cost_ms = 0.0F;
            {
                Timer timer(io_cost_ms);
                this->io_->MultiRead(codes.data, sizes.data(), offsets.data(), id_count);
            }

            if (ctx != nullptr and ctx->stats != nullptr) {
                ctx->stats->io_cnt.fetch_add(id_count, std::memory_order_relaxed);
                ctx->stats->io_time_ms.fetch_add(static_cast<uint32_t>(io_cost_ms),
                                                 std::memory_order_relaxed);
            }

            computer->ScanBatchDists(id_count, codes.data, result_dists);
            return;
        }

        if (ctx != nullptr and ctx->stats != nullptr and id_count > 0) {
            ctx->stats->io_cnt.fetch_add(static_cast<uint32_t>(id_count),
                                         std::memory_order_relaxed);
        }
    }

    memset(result_dists, 0, sizeof(float) * id_count);
    int64_t i = 0;
    for (; i + 15 < id_count; i += 16) {
        for (int64_t j = 0; j < 16; ++j) {
            if (i + j + this->prefetch_stride_code_ < id_count) {
                this->io_->Prefetch(
                    static_cast<uint64_t>(idx[i + j + this->prefetch_stride_code_]) *
                        static_cast<uint64_t>(code_size_),
                    this->prefetch_depth_code_ * 64);
            }
        }
        bool release1 = false, release2 = false, release3 = false, release4 = false;
        bool release5 = false, release6 = false, release7 = false, release8 = false;
        bool release9 = false, release10 = false, release11 = false, release12 = false;
        bool release13 = false, release14 = false, release15 = false, release16 = false;
        const uint8_t *codes1 = nullptr, *codes2 = nullptr, *codes3 = nullptr, *codes4 = nullptr;
        const uint8_t *codes5 = nullptr, *codes6 = nullptr, *codes7 = nullptr, *codes8 = nullptr;
        const uint8_t *codes9 = nullptr, *codes10 = nullptr, *codes11 = nullptr, *codes12 = nullptr;
        const uint8_t *codes13 = nullptr, *codes14 = nullptr, *codes15 = nullptr, *codes16 = nullptr;
        auto release_batch = [&]() {
            if (release1 && codes1) this->io_->Release(codes1);
            if (release2 && codes2) this->io_->Release(codes2);
            if (release3 && codes3) this->io_->Release(codes3);
            if (release4 && codes4) this->io_->Release(codes4);
            if (release5 && codes5) this->io_->Release(codes5);
            if (release6 && codes6) this->io_->Release(codes6);
            if (release7 && codes7) this->io_->Release(codes7);
            if (release8 && codes8) this->io_->Release(codes8);
            if (release9 && codes9) this->io_->Release(codes9);
            if (release10 && codes10) this->io_->Release(codes10);
            if (release11 && codes11) this->io_->Release(codes11);
            if (release12 && codes12) this->io_->Release(codes12);
            if (release13 && codes13) this->io_->Release(codes13);
            if (release14 && codes14) this->io_->Release(codes14);
            if (release15 && codes15) this->io_->Release(codes15);
            if (release16 && codes16) this->io_->Release(codes16);
        };
        try {
            codes1 = this->GetCodesById(idx[i], release1);
            codes2 = this->GetCodesById(idx[i + 1], release2);
            codes3 = this->GetCodesById(idx[i + 2], release3);
            codes4 = this->GetCodesById(idx[i + 3], release4);
            codes5 = this->GetCodesById(idx[i + 4], release5);
            codes6 = this->GetCodesById(idx[i + 5], release6);
            codes7 = this->GetCodesById(idx[i + 6], release7);
            codes8 = this->GetCodesById(idx[i + 7], release8);
            codes9 = this->GetCodesById(idx[i + 8], release9);
            codes10 = this->GetCodesById(idx[i + 9], release10);
            codes11 = this->GetCodesById(idx[i + 10], release11);
            codes12 = this->GetCodesById(idx[i + 11], release12);
            codes13 = this->GetCodesById(idx[i + 12], release13);
            codes14 = this->GetCodesById(idx[i + 13], release14);
            codes15 = this->GetCodesById(idx[i + 14], release15);
            codes16 = this->GetCodesById(idx[i + 15], release16);
            {
                const auto* query_data = reinterpret_cast<const float*>(computer->buf_);
                uint64_t dim = this->quantizer_->GetDim();
                const auto* fcodes1 = reinterpret_cast<const float*>(codes1);
                const auto* fcodes2 = reinterpret_cast<const float*>(codes2);
                const auto* fcodes3 = reinterpret_cast<const float*>(codes3);
                const auto* fcodes4 = reinterpret_cast<const float*>(codes4);
                const auto* fcodes5 = reinterpret_cast<const float*>(codes5);
                const auto* fcodes6 = reinterpret_cast<const float*>(codes6);
                const auto* fcodes7 = reinterpret_cast<const float*>(codes7);
                const auto* fcodes8 = reinterpret_cast<const float*>(codes8);
                const auto* fcodes9 = reinterpret_cast<const float*>(codes9);
                const auto* fcodes10 = reinterpret_cast<const float*>(codes10);
                const auto* fcodes11 = reinterpret_cast<const float*>(codes11);
                const auto* fcodes12 = reinterpret_cast<const float*>(codes12);
                const auto* fcodes13 = reinterpret_cast<const float*>(codes13);
                const auto* fcodes14 = reinterpret_cast<const float*>(codes14);
                const auto* fcodes15 = reinterpret_cast<const float*>(codes15);
                const auto* fcodes16 = reinterpret_cast<const float*>(codes16);
                if (this->quantizer_->Metric() == MetricType::METRIC_TYPE_L2SQR) {
                    FP32ComputeL2SqrBatch16(query_data, dim,
                                            fcodes1, fcodes2, fcodes3, fcodes4,
                                            fcodes5, fcodes6, fcodes7, fcodes8,
                                            fcodes9, fcodes10, fcodes11, fcodes12,
                                            fcodes13, fcodes14, fcodes15, fcodes16,
                                            result_dists[i], result_dists[i + 1],
                                            result_dists[i + 2], result_dists[i + 3],
                                            result_dists[i + 4], result_dists[i + 5],
                                            result_dists[i + 6], result_dists[i + 7],
                                            result_dists[i + 8], result_dists[i + 9],
                                            result_dists[i + 10], result_dists[i + 11],
                                            result_dists[i + 12], result_dists[i + 13],
                                            result_dists[i + 14], result_dists[i + 15]);
                } else {
                    FP32ComputeIPBatch16(query_data, dim,
                                         fcodes1, fcodes2, fcodes3, fcodes4,
                                         fcodes5, fcodes6, fcodes7, fcodes8,
                                         fcodes9, fcodes10, fcodes11, fcodes12,
                                         fcodes13, fcodes14, fcodes15, fcodes16,
                                         result_dists[i], result_dists[i + 1],
                                         result_dists[i + 2], result_dists[i + 3],
                                         result_dists[i + 4], result_dists[i + 5],
                                         result_dists[i + 6], result_dists[i + 7],
                                         result_dists[i + 8], result_dists[i + 9],
                                         result_dists[i + 10], result_dists[i + 11],
                                         result_dists[i + 12], result_dists[i + 13],
                                         result_dists[i + 14], result_dists[i + 15]);
                    result_dists[i] = 1.0F - result_dists[i];
                    result_dists[i + 1] = 1.0F - result_dists[i + 1];
                    result_dists[i + 2] = 1.0F - result_dists[i + 2];
                    result_dists[i + 3] = 1.0F - result_dists[i + 3];
                    result_dists[i + 4] = 1.0F - result_dists[i + 4];
                    result_dists[i + 5] = 1.0F - result_dists[i + 5];
                    result_dists[i + 6] = 1.0F - result_dists[i + 6];
                    result_dists[i + 7] = 1.0F - result_dists[i + 7];
                    result_dists[i + 8] = 1.0F - result_dists[i + 8];
                    result_dists[i + 9] = 1.0F - result_dists[i + 9];
                    result_dists[i + 10] = 1.0F - result_dists[i + 10];
                    result_dists[i + 11] = 1.0F - result_dists[i + 11];
                    result_dists[i + 12] = 1.0F - result_dists[i + 12];
                    result_dists[i + 13] = 1.0F - result_dists[i + 13];
                    result_dists[i + 14] = 1.0F - result_dists[i + 14];
                    result_dists[i + 15] = 1.0F - result_dists[i + 15];
                }
            }
        } catch (...) {
            release_batch();
            throw;
        }
        release_batch();
    }
    for (; i + 7 < id_count; i += 8) {
        for (int64_t j = 0; j < 8; ++j) {
            if (i + j + this->prefetch_stride_code_ < id_count) {
                this->io_->Prefetch(
                    static_cast<uint64_t>(idx[i + j + this->prefetch_stride_code_]) *
                        static_cast<uint64_t>(code_size_),
                    this->prefetch_depth_code_ * 64);
            }
        }
        bool release1 = false, release2 = false, release3 = false, release4 = false;
        bool release5 = false, release6 = false, release7 = false, release8 = false;
        const uint8_t *codes1 = nullptr, *codes2 = nullptr, *codes3 = nullptr, *codes4 = nullptr;
        const uint8_t *codes5 = nullptr, *codes6 = nullptr, *codes7 = nullptr, *codes8 = nullptr;
        auto release_batch = [&]() {
            if (release1 && codes1) this->io_->Release(codes1);
            if (release2 && codes2) this->io_->Release(codes2);
            if (release3 && codes3) this->io_->Release(codes3);
            if (release4 && codes4) this->io_->Release(codes4);
            if (release5 && codes5) this->io_->Release(codes5);
            if (release6 && codes6) this->io_->Release(codes6);
            if (release7 && codes7) this->io_->Release(codes7);
            if (release8 && codes8) this->io_->Release(codes8);
        };
        try {
            codes1 = this->GetCodesById(idx[i], release1);
            codes2 = this->GetCodesById(idx[i + 1], release2);
            codes3 = this->GetCodesById(idx[i + 2], release3);
            codes4 = this->GetCodesById(idx[i + 3], release4);
            codes5 = this->GetCodesById(idx[i + 4], release5);
            codes6 = this->GetCodesById(idx[i + 5], release6);
            codes7 = this->GetCodesById(idx[i + 6], release7);
            codes8 = this->GetCodesById(idx[i + 7], release8);
            {
                const auto* query_data = reinterpret_cast<const float*>(computer->buf_);
                uint64_t dim = this->quantizer_->GetDim();
                const auto* fcodes1 = reinterpret_cast<const float*>(codes1);
                const auto* fcodes2 = reinterpret_cast<const float*>(codes2);
                const auto* fcodes3 = reinterpret_cast<const float*>(codes3);
                const auto* fcodes4 = reinterpret_cast<const float*>(codes4);
                const auto* fcodes5 = reinterpret_cast<const float*>(codes5);
                const auto* fcodes6 = reinterpret_cast<const float*>(codes6);
                const auto* fcodes7 = reinterpret_cast<const float*>(codes7);
                const auto* fcodes8 = reinterpret_cast<const float*>(codes8);
                if (this->quantizer_->Metric() == MetricType::METRIC_TYPE_L2SQR) {
                    FP32ComputeL2SqrBatch8(query_data, dim,
                                           fcodes1, fcodes2, fcodes3, fcodes4,
                                           fcodes5, fcodes6, fcodes7, fcodes8,
                                           result_dists[i], result_dists[i + 1],
                                           result_dists[i + 2], result_dists[i + 3],
                                           result_dists[i + 4], result_dists[i + 5],
                                           result_dists[i + 6], result_dists[i + 7]);
                } else {
                    FP32ComputeIPBatch8(query_data, dim,
                                        fcodes1, fcodes2, fcodes3, fcodes4,
                                        fcodes5, fcodes6, fcodes7, fcodes8,
                                        result_dists[i], result_dists[i + 1],
                                        result_dists[i + 2], result_dists[i + 3],
                                        result_dists[i + 4], result_dists[i + 5],
                                        result_dists[i + 6], result_dists[i + 7]);
                    result_dists[i] = 1.0F - result_dists[i];
                    result_dists[i + 1] = 1.0F - result_dists[i + 1];
                    result_dists[i + 2] = 1.0F - result_dists[i + 2];
                    result_dists[i + 3] = 1.0F - result_dists[i + 3];
                    result_dists[i + 4] = 1.0F - result_dists[i + 4];
                    result_dists[i + 5] = 1.0F - result_dists[i + 5];
                    result_dists[i + 6] = 1.0F - result_dists[i + 6];
                    result_dists[i + 7] = 1.0F - result_dists[i + 7];
                }
            }
        } catch (...) {
            release_batch();
            throw;
        }
        release_batch();
    }
    for (; i + 3 < id_count; i += 4) {
        for (int64_t j = 0; j < 4; ++j) {
            if (i + j + this->prefetch_stride_code_ < id_count) {
                this->io_->Prefetch(
                    static_cast<uint64_t>(idx[i + j + this->prefetch_stride_code_]) *
                        static_cast<uint64_t>(code_size_),
                    this->prefetch_depth_code_ * 64);
            }
        }
        bool release1 = false, release2 = false, release3 = false, release4 = false;
        const uint8_t* codes1 = nullptr;
        const uint8_t* codes2 = nullptr;
        const uint8_t* codes3 = nullptr;
        const uint8_t* codes4 = nullptr;
        auto release_batch = [&]() {
            if (release1 && codes1) {
                this->io_->Release(codes1);
            }
            if (release2 && codes2) {
                this->io_->Release(codes2);
            }
            if (release3 && codes3) {
                this->io_->Release(codes3);
            }
            if (release4 && codes4) {
                this->io_->Release(codes4);
            }
        };
        try {
            codes1 = this->GetCodesById(idx[i], release1);
            codes2 = this->GetCodesById(idx[i + 1], release2);
            codes3 = this->GetCodesById(idx[i + 2], release3);
            codes4 = this->GetCodesById(idx[i + 3], release4);
            computer->ComputeDistsBatch4(codes1,
                                         codes2,
                                         codes3,
                                         codes4,
                                         result_dists[i],
                                         result_dists[i + 1],
                                         result_dists[i + 2],
                                         result_dists[i + 3]);
        } catch (...) {
            release_batch();
            throw;
        }
        release_batch();
    }
    for (; i < id_count; ++i) {
        bool release = false;
        const uint8_t* codes = nullptr;
        try {
            codes = this->GetCodesById(idx[i], release);
            computer->ComputeDist(codes, result_dists + i);
        } catch (...) {
            if (release && codes) {
                this->io_->Release(codes);
            }
            throw;
        }
        if (release && codes) {
            this->io_->Release(codes);
        }
    }
}

template <typename QuantTmpl, typename IOTmpl>
float
FlattenDataCell<QuantTmpl, IOTmpl>::ComputePairVectors(InnerIdType id1, InnerIdType id2) {
    bool release1 = false, release2 = false;
    const uint8_t* codes1 = nullptr;
    const uint8_t* codes2 = nullptr;
    auto release_pair = [&]() {
        if (release1 && codes1) {
            this->io_->Release(codes1);
        }
        if (release2 && codes2) {
            this->io_->Release(codes2);
        }
    };
    try {
        codes1 = this->GetCodesById(id1, release1);
        codes2 = this->GetCodesById(id2, release2);
        auto result = this->quantizer_->Compute(codes1, codes2);
        release_pair();
        return result;
    } catch (...) {
        release_pair();
        throw;
    }
}

template <typename QuantTmpl, typename IOTmpl>
const uint8_t*
FlattenDataCell<QuantTmpl, IOTmpl>::GetCodesById(InnerIdType id, bool& need_release) const {
    return io_->Read(
        code_size_, static_cast<uint64_t>(id) * static_cast<uint64_t>(code_size_), need_release);
}

template <typename QuantTmpl, typename IOTmpl>
bool
FlattenDataCell<QuantTmpl, IOTmpl>::GetCodesById(InnerIdType id, uint8_t* codes) const {
    return io_->Read(
        code_size_, static_cast<uint64_t>(id) * static_cast<uint64_t>(code_size_), codes);
}

template <typename QuantTmpl, typename IOTmpl>
void
FlattenDataCell<QuantTmpl, IOTmpl>::Serialize(StreamWriter& writer) {
    FlattenInterface::Serialize(writer);
    this->io_->Serialize(writer);
    this->quantizer_->Serialize(writer);
}

template <typename QuantTmpl, typename IOTmpl>
void
FlattenDataCell<QuantTmpl, IOTmpl>::Deserialize(lvalue_or_rvalue<StreamReader> reader) {
    FlattenInterface::Deserialize(reader);
    this->io_->Deserialize(reader);
    this->quantizer_->Deserialize(reader);
}

template <typename QuantTmpl, typename IOTmpl>
void
FlattenDataCell<QuantTmpl, IOTmpl>::MergeOther(const FlattenInterfacePtr& other, InnerIdType bias) {
    auto ptr = std::dynamic_pointer_cast<FlattenDataCell<QuantTmpl, IOTmpl>>(other);
    if (ptr == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "Merge flatten datacell failed: not match type");
    }
    constexpr uint64_t BUFFER_SIZE = 1024 * 1024 * 10;
    uint64_t total_count = ptr->total_count_;
    uint64_t offset = static_cast<uint64_t>(bias) * static_cast<uint64_t>(code_size_);
    uint64_t read_count = 0;
    while (read_count < total_count) {
        bool need_release = false;
        uint64_t count = std::min(BUFFER_SIZE / this->code_size_, total_count - read_count);
        uint64_t size = count * this->code_size_;
        auto* buffer = ptr->io_->Read(size, read_count * this->code_size_, need_release);
        this->io_->Write(buffer, size, offset);
        if (need_release) {
            ptr->io_->Release(buffer);
        }
        offset += size;
        read_count += count;
    }
    this->total_count_ += total_count;
}

template <typename QuantTmpl, typename IOTmpl>
uint64_t
FlattenDataCell<QuantTmpl, IOTmpl>::GetMemoryUsage() const {
    uint64_t memory = sizeof(FlattenDataCell<QuantTmpl, IOTmpl>);
    if (IOTmpl::InMemory) {
        memory += this->io_->GetMemoryUsage();
    }
    memory += sizeof(QuantTmpl);
    return memory;
}

template <typename QuantTmpl, typename IOTmpl>
void
FlattenDataCell<QuantTmpl, IOTmpl>::Move(InnerIdType from, InnerIdType to) {
    bool need_release = false;
    const uint8_t* codes = this->GetCodesById(from, need_release);
    this->io_->Write(
        codes, code_size_, static_cast<uint64_t>(to) * static_cast<uint64_t>(code_size_));
    if (need_release) {
        this->io_->Release(codes);
    }
}

}  // namespace vsag
