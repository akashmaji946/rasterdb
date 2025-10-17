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

#include "data/simple_data_repository_level.hpp"

namespace sirius {

void 
SimpleDataRepositoryLevel::AddNewDataBatch(sirius::unique_ptr<DataBatch> data_batch) {
    data_batches_.emplace_back(std::move(data_batch));
}

sirius::unique_ptr<DataBatch> 
SimpleDataRepositoryLevel::EvictDataBatch() {
    for (auto it = data_batches_.begin(); it != data_batches_.end(); ++it) {
        // if ((*it)->getBatchId() == data_batch_id) {
            auto evicted_batch = std::move(*it);
            data_batches_.erase(it);
            return evicted_batch;
        // }
    }
    throw std::invalid_argument("Data batch with the specified id does not exist");
}

std::vector<uint64_t> 
SimpleDataRepositoryLevel::GetDowngradableDataBatches(size_t num_data_batches) {
    std::vector<uint64_t> downgrable_batches;
    for (const auto &batch : data_batches_) {
        if (batch->getCurrentTier() != Tier::DISK) { // Only consider batches not already in the lowest tier
            downgrable_batches.push_back(batch->getBatchId());
            if (downgrable_batches.size() == num_data_batches) {
                break;
            }
        }
    }
    return downgrable_batches;
}

} // namespace sirius