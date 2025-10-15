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
#include "helper/helper.hpp"

namespace sirius {

SiriusContext::SiriusContext() :
    data_repository_(DataRepository()),
    gpu_pipeline_executor_(parallel::GPUPipelineExecutor(parallel::TaskExecutorConfig(1, 0), data_repository_)),
    downgrade_executor_(parallel::DowngradeExecutor(parallel::TaskExecutorConfig(1, 0), data_repository_)),
    duckdb_scan_executor_(parallel::DuckDBScanExecutor(parallel::TaskExecutorConfig(1, 0), data_repository_)),
    task_creator_(TaskCreator(data_repository_, gpu_pipeline_executor_, duckdb_scan_executor_)),
    downgrade_task_creator_(data_repository_, downgrade_executor_) {}

} // namespace sirius
