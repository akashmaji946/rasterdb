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
#include "config.hpp"

namespace sirius {

sirius_context::sirius_context() :
    _data_repo_mgr(),
    _gpu_pipeline_executor(parallel::task_executor_config{Config::NUM_GPU_PIPELINE_EXECUTOR_THREADS, false}),
    _downgrade_executor(parallel::task_executor_config{Config::NUM_DOWNGRADE_EXECUTOR_THREADS, false}, _data_repo_mgr),
    _duckdb_scan_executor(parallel::task_executor_config{Config::NUM_DUCKDB_SCAN_EXECUTOR_THREADS, false}, _data_repo_mgr),
    _sirius_task_creator(_data_repo_mgr, _gpu_pipeline_executor, _duckdb_scan_executor),
    _downgrade_task_creator(_data_repo_mgr, _downgrade_executor) {}

} // namespace sirius
