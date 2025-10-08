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

#include "include/sirius_context.hpp"

namespace duckdb {
namespace sirius {

SiriusContext::SiriusContext() :
    gpu_pipeline_task_queue_(parallel::GPUPipelineTaskQueue()),
    downgrade_task_queue_(parallel::DowngradeTaskQueue()),
    data_repository_(duckdb::make_shared_ptr<DataRepository>()),
    gpu_pipeline_executor_(parallel::GPUPipelineExecutor(parallel::TaskExecutorConfig(1, 0), data_repository_)),
    downgrade_executor_(parallel::DowngradeExecutor(parallel::TaskExecutorConfig(1, 0), data_repository_)),
    task_creator_(TaskCreator(data_repository_, gpu_pipeline_task_queue_)),
    downgrade_task_creator_(parallel::DowngradeTaskCreator(parallel::TaskExecutorConfig(1, 0), data_repository_)) {
}

} // namespace sirius
} // namespace duckdb
