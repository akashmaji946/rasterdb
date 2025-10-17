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
#include "pipeline/gpu_pipeline_executor.hpp"
#include "scan/duckdb_scan_executor.hpp"
#include "parallel/task_executor.hpp"
#include "parallel/task.hpp"
#include "parallel/task_creator.hpp"
#include "gpu_pipeline_hashmap.hpp"
#include "task_completion.hpp"

namespace sirius {

class TaskCreator : public ITaskCreator {
public:
    TaskCreator(DataRepository& data_repository,
        parallel::GPUPipelineExecutor &gpu_pipeline_executor,
        parallel::DuckDBScanExecutor& duckdb_scan_executor);
          
    // Destructor
    ~TaskCreator() = default;

    // Non-copyable but movable
    TaskCreator(const TaskCreator&) = delete;
    TaskCreator& operator=(const TaskCreator&) = delete;
    TaskCreator(TaskCreator&&) = default;
    TaskCreator& operator=(TaskCreator&&) = default;

    // scan the data repository for new data batches and submit pipeline tasks
    void ScanRepository(size_t pipeline_idx);

    void WorkerLoop() override;

    void SetCoordinator(duckdb::GPUExecutor* coordinator) {
        coordinator_ = coordinator;
    }

    void SetGPUPipelineHashMap(sirius::shared_ptr<GPUPipelineHashMap> gpu_pipeline_hashmap) {
        gpu_pipeline_hashmap_ = gpu_pipeline_hashmap;
    }

    sirius::unique_ptr<parallel::DuckDBScanTask> CreateScanTask(size_t pipeline_idx);
    sirius::unique_ptr<parallel::GPUPipelineTask> CreatePipelineTask(size_t pipeline_idx, 
        sirius::vector<sirius::unique_ptr<DataBatch>> data_batches);

    // submit scan task to scan executor
    void ScheduleDuckDBScan (sirius::unique_ptr<parallel::DuckDBScanTask> scan_getsize_task);

    // submit pipeline task to pipeline executor
    void SchedulePipelineTask(sirius::unique_ptr<parallel::GPUPipelineTask> gpu_pipeline_task);

    void Run();

    void Signal();

    void Wait();

    uint64_t GetNextTaskId() {
        return next_task_id_++;
    }

private:
    TaskCompletionMessageQueue task_completion_message_queue_;
    DataRepository& data_repository_;
    parallel::GPUPipelineExecutor& gpu_pipeline_executor_;
    parallel::DuckDBScanExecutor& duckdb_scan_executor_;
    duckdb::GPUExecutor* coordinator_; // reference to the coordinator (GPUExecutor)
    sirius::shared_ptr<GPUPipelineHashMap> gpu_pipeline_hashmap_;

    sirius::atomic<uint64_t> next_task_id_ = 0; // Atomic counter for generating unique task IDs
	sirius::mutex mtx;
	std::condition_variable cv;
	bool ready = false;
};

} // namespace sirius