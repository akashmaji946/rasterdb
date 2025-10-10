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
#include "data_batch.hpp"
#include "helper/helper.hpp"

namespace sirius {

//! Thread-safe repository for storing and retrieving DataBatch objects
class DataRepository {
public:

    DataRepository() = default;

    void Initialize(size_t num_pipelines) {
        std::lock_guard<std::mutex> lock(mutex_);
        data_batches_.resize(num_pipelines);
    }

    // Add a new DataBatch to the repository at the specified pipeline_id and idx
    void AddNewDataBatch(size_t pipeline_id, size_t idx, sirius::unique_ptr<DataBatch> data_batch) {
        lock_guard<mutex> lock(mutex_);
        
        // Ensure the pipeline_id is valid
        if (pipeline_id >= data_batches_.size()) {
            throw std::out_of_range("Invalid pipeline_id");
        }
        
        // Ensure the inner vector is large enough
        if (idx >= data_batches_[pipeline_id].size()) {
            data_batches_[pipeline_id].resize(idx + 1);
        }
        
        // Store the data batch
        data_batches_[pipeline_id][idx] = std::move(data_batch);
    }

    // Get a DataBatch by pipeline_id and idx and transfer ownership
    sirius::unique_ptr<DataBatch> GetDataBatch(size_t pipeline_id, size_t idx) {
        lock_guard<mutex> lock(mutex_);
        
        // Check bounds
        if (pipeline_id >= data_batches_.size() || 
            idx >= data_batches_[pipeline_id].size()) {
            return nullptr;
        }
        
        // Transfer ownership and clear the slot
        auto result = std::move(data_batches_[pipeline_id][idx]);
        data_batches_[pipeline_id][idx] = nullptr;
        
        return result;
    }

private:
    // The data repository is organized as a 2D vector: outer vector indexed by pipeline_id, inner vector indexed by idx
    sirius::vector<sirius::vector<sirius::unique_ptr<DataBatch>>> data_batches_;
    mutex mutex_; // Mutex to protect access to data_batches
};

}