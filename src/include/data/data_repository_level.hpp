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

/**
 * @brief Interface for a level in the data repository hierarchy.
 * 
 * Each level in the data repository is a thread-safe container to store the output of a specific pipeline in the query plan.
 * Thus, as a chunk of data goes through the various stages of the query plan, it gets stored in different levels of the data repository.
 * Thus each level doesn't have to reason about the query plan and the execution DAG when it comes to decision such as which DataBatch to downgrade.
 */
class IDataRepositoryLevel {
public:
    /**
     * @brief Method to add new data batch to the current level
     * 
     * @param data_batch The data batch to add to this level
     */
    virtual void AddNewDataBatch(sirius::unique_ptr<DataBatch> data_batch) = 0;

    /**
     * @brief Method to evict a data batch from the current level by its unique identifier
     * 
     * @param data_batch_id The unique identifier of the data batch to evict
     * @return sirius::unique_ptr<DataBatch> The evicted data batch
     * @throws std::invalid_argument if the data batch with the specified id does not exist
     */
    virtual sirius::unique_ptr<DataBatch> EvictDataBatch(uint64_t data_batch_id) = 0;

    /**
     * @brief Method to get an ordered (priority wise) list of data batch ids that can be downgraded to a lower tier
     * 
     * The ids returned by this method should be ordered in the order of priority for downgrading in that the id at index 0 should be given 
     * higher consideration for downgrading than the id at index 1 and so on. Each derived classs can implement its own logic for determining
     * the priority of downgrading batches in the current level
     * 
     * @param num_data_batches The number of data batches that should be returned (essentially returns the Top-K downgrable batches)
     * @return std::vector<uint64_t> The ordered list of data batch ids that can be downgraded. Its size = max(num_data_batches, # of downgradable batches)
     */
    virtual std::vector<uint64_t> GetDowngradableDataBatches(size_t num_data_batches) const = 0;
};

} // namespace sirius