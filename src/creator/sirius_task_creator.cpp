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

#include "creator/sirius_task_creator.hpp"
#include "gpu_executor.hpp"
#include "scan/duckdb_scan_executor.hpp"
#include "pipeline/gpu_pipeline_executor.hpp"
#include "gpu_pipeline_hashmap.hpp"
#include "gpu_physical_table_scan.hpp"

namespace sirius {

sirius_task_creator::sirius_task_creator(
    data_repository_manager& data_repo_mgr,
    parallel::gpu_pipeline_executor& gpu_pipeline_exec,
    parallel::duckdb_scan_executor& duckdb_scan_exec
    ) : 
    itask_creator(), _data_repo_mgr(data_repo_mgr),
        _gpu_pipeline_exec(gpu_pipeline_exec),
        _duckdb_scan_exec(duckdb_scan_exec),
        _task_completion_msg_queue(), 
        _coordinator(nullptr) {
}

sirius::unique_ptr<parallel::duckdb_scan_task>
sirius_task_creator::create_scan_task(size_t pipeline_idx) {
    // create new scan task
    auto it = _gpu_pipeline_hashmap->_scan_metadata_map.find(pipeline_idx);
    if (it == _gpu_pipeline_hashmap->_scan_metadata_map.end()) {
        throw std::runtime_error("No scan metadata found for pipeline id " + std::to_string(pipeline_idx));
    }
    auto& scan_metadata = it->second;

    sirius::shared_ptr<parallel::duckdb_scan_global_source_state> global_state = 
        sirius::make_shared<parallel::duckdb_scan_global_source_state>(_data_repo_mgr, _task_completion_msg_queue);

    sirius::shared_ptr<parallel::duckdb_scan_local_source_state> local_state = 
        sirius::make_shared<parallel::duckdb_scan_local_source_state>(get_next_task_id(), pipeline_idx, scan_metadata);

    sirius::unique_ptr<parallel::duckdb_scan_task> scan_task = 
        sirius::make_unique<parallel::duckdb_scan_task>(
            std::move(local_state),
            std::move(global_state)  // global state can be initialized inside the task
        );
    return scan_task;
}

sirius::unique_ptr<parallel::gpu_pipeline_task>
sirius_task_creator::create_pipeline_task(size_t pipeline_idx, sirius::vector<sirius::unique_ptr<data_batch_view>> batch_views) {
    // create pipeline tasks for each pipeline in the _gpu_pipeline_hashmap that can consume the data batch
    auto pipeline = _gpu_pipeline_hashmap->_vec[pipeline_idx];
    sirius::shared_ptr<parallel::gpu_pipeline_task_global_state> global_state = 
        sirius::make_shared<parallel::gpu_pipeline_task_global_state>(_data_repo_mgr, _task_completion_msg_queue);

    sirius::shared_ptr<parallel::gpu_pipeline_task_local_state> local_state = 
        sirius::make_shared<parallel::gpu_pipeline_task_local_state>(get_next_task_id(), pipeline_idx, pipeline, std::move(batch_views));

    // create new pipeline task
    sirius::unique_ptr<parallel::gpu_pipeline_task> pipeline_task = 
        sirius::make_unique<parallel::gpu_pipeline_task>(
            std::move(local_state),
            std::move(global_state)  // global state can be initialized inside the task
        );
    return pipeline_task;
}

void 
sirius_task_creator::worker_loop() {
    wait();  // wait for signal from B
    // TODO: Implement the worker loop
    _coordinator->Signal();  // signal B
}

bool
sirius_task_creator::scan_repository(size_t pipeline_idx) {
    // TODO: Implement the scan repository logic
    return false;
}

void
sirius_task_creator::schedule_duckdb_scan(sirius::unique_ptr<parallel::duckdb_scan_task> scan_task) {
    _duckdb_scan_exec.schedule(std::move(scan_task));
}

void
sirius_task_creator::schedule_pipeline_task(sirius::unique_ptr<parallel::gpu_pipeline_task> pipeline_task) {
    _gpu_pipeline_exec.schedule(std::move(pipeline_task));
}

} // namespace sirius