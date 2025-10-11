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

#include <unordered_map>

#include "data_batch.hpp"
#include "helper/helper.hpp"
#include "data/data_repository_level.hpp"

namespace sirius {

/**
 * @brief A container for DataBatches produced and consumed by different tasks in the system. 
 * 
 * The DataRepository is primarily used to store DataBatches that are currently between tasks. Thus each task
 * outputs its result to the DataRepository and the subsequent tasks remove it from the DataRepository when they are
 * ready to work on it. 
 * 
 * The DataRepository is leveled container, where each level corresponds to the output a specific pipeline in a given query plan.
 * When deciding which DataBatch to downgrade to a lower memory tier, the DataRepository should consider where the pipeline is in
 * the overall query DAG and differ to the indvidual levels to determine which DataBatches within that level should be downgraded first. 
 */
class DataRepository {
public:
    /**
     * @brief Default constructor for the DataRepository
     */
    DataRepository() = default;

    /** 
     * @brief Method to add a new level to the DataRepository for a specific pipeline
     * 
     * Note that when this method is called, it moves the ownership of the level to the DataRepository and thus the level should not be used by anyone else
     * after this call.
     * 
     * @param pipeline_id The id of the pipeline for which the level is being added
     * @param level The level to add to the DataRepository
     * @throws std::invalid_argument if a level already exists for the specified pipeline_id
    */
    void AddNewLevel(size_t pipeline_id, sirius::unique_ptr<IDataRepositoryLevel> level);

    /**
     * @brief Add a new DataBatch to the repository
     * 
     * If a level was not previously initialized for the given pipeline_id, it will also be intitialized with the default IDataRepositoryLevel implementation.
     * Thus, it is recommended that AddNewLevel is called for each pipeline in the query plan before starting execution.
     * 
     * @param pipeline_id The id of the pipeline that is depositing the DataBatch into the repository
     * @param data_batch The DataBatch to add to the repository
     */
    void AddNewDataBatch(size_t pipeline_id, sirius::unique_ptr<DataBatch> data_batch);

    /**
     * @brief Evict a DataBatch from the repository
     * 
     * This method removes the DataBatch with the specified id from the level corresponding to the specified pipeline_id and returns it.
     * 
     * @param pipeline_id The id of the pipeline where the DataBatch currently resides
     * @param data_batch_id The unique identifier of the DataBatch to evict
     * @return sirius::unique_ptr<DataBatch> The evicted DataBatch
     * @throws std::invalid_argument if no level exists for the specified pipeline_id or if the data batch doesn't exist in the provided level
     */
    sirius::unique_ptr<DataBatch> EvictDataBatch(size_t pipeline_id, uint64_t data_batch_id);

private:
    sirius::unordered_map<size_t, sirius::unique_ptr<IDataRepositoryLevel>> levels_; // A map storing the different levels in the DataRepository
    mutex mutex_; // Mutex to protect access to data_batches
};

}