/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <variant>
#include <memory>
#include <cudf/table/table.hpp>

#include "helper/helper.hpp"
#include "data/common.hpp"
#include "memory/memory_reservation.hpp"

namespace sirius {

using sirius::memory::Tier;

/**
 * @brief A class that represents the input(s) and output(s) of the different pipelines in a query. 
 * 
 * The DataBatch in Sirius represents a batch/chunk/row group of the data that is processed by/outputed by a task of a pipeline,
 * based on the "morsel driven" execution model used by many systems. The underlying data can be stored in different memory tiers
 * and in different formats as and is owned by the underlying IDataRepresentation rather than the DataBatch itself.
 */
class DataBatch {
public:
    /**
     * @brief Construct a new Data Batch object
     * 
     * @param batch_id Unique identifier for the data batch
     * @param data The actual data associated with this data batch
     */
    DataBatch(uint64_t batch_id, sirius::unique_ptr<IDataRepresentation> data) 
        : batch_id_(batch_id), data_(std::move(data)) {}
    
    // Move constructors
    DataBatch(DataBatch&& other) noexcept
        : batch_id_(other.batch_id_), data_(std::move(other.data_)) {
        other.batch_id_ = 0;
        other.data_ = nullptr;
    }

    DataBatch& operator=(DataBatch&& other) noexcept {
        if (this != &other) {
            batch_id_ = other.batch_id_;
            data_ = std::move(other.data_);
            other.batch_id_ = 0;
            other.data_ = nullptr;
        }
        return *this;
    }

    /**
     * @brief Get the tier that this data batch currently resides in
     * 
     * @return Tier The memory tier
     */
    Tier getCurrentTier() const {
        return data_->getCurrentTier();
    }

    /**
     * @brief Get this Data Batch's id
     * 
     * @return The unique identifier associated with this Data Batch
     */
    uint64_t getBatchId() const {
        return batch_id_;
    }

private:
    uint64_t batch_id_; // Unique identifier for the data batch
    sirius::unique_ptr<IDataRepresentation> data_; // Pointer to the actual data representation
};

} // namespace sirius