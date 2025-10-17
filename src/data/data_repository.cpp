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

#include "data/data_repository.hpp"
#include "data/simple_data_repository_level.hpp"

namespace sirius {

void DataRepository::AddNewLevel(size_t pipeline_id, sirius::unique_ptr<IDataRepositoryLevel> level) {
    if (levels_.find(pipeline_id) != levels_.end()) {
        throw std::invalid_argument("Level already exists for the specified pipeline_id");
    }
    levels_[pipeline_id] = std::move(level);
}

void DataRepository::AddNewDataBatch(size_t pipeline_id, sirius::unique_ptr<DataBatch> data_batch) {
    if (levels_.find(pipeline_id) == levels_.end()) {
        // If a level does not exist for the specified pipeline_id, create a default one
        levels_[pipeline_id] = sirius::make_unique<SimpleDataRepositoryLevel>();
    }
    levels_[pipeline_id]->Cast<SimpleDataRepositoryLevel>().AddNewDataBatch(std::move(data_batch));
}

sirius::unique_ptr<DataBatch> DataRepository::EvictDataBatch(size_t pipeline_id) {
    if (levels_.find(pipeline_id) == levels_.end()) {
        throw std::invalid_argument("No level exists for the specified pipeline_id");
    }
    return levels_[pipeline_id]->EvictDataBatch();
}

} // namespace sirius