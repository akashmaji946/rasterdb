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
#include "parallel/task_executor.hpp"
#include "memory/memory_reservation.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "data/data_repository.hpp"

namespace duckdb {
namespace sirius {
namespace parallel {

/**
 * GPU-specific task executor that inherits from ITaskExecutor and uses
 * GPUPipelineTaskQueue as its scheduler. Manages a pool of threads to
 * execute GPU pipeline tasks with specialized GPU resource management.
 */
class GPUPipelineExecutor : public ITaskExecutor {
public:
    /**
     * Constructor that creates a GPUPipelineExecutor with a GPUPipelineTaskQueue scheduler
     * @param config Configuration for the task executor (thread count, retry policy, etc.)
     * @param data_repository Optional data repository for data access
     * @param reservation_manager Reference to the memory reservation manager
     */
    explicit GPUPipelineExecutor(
        TaskExecutorConfig config,
        duckdb::shared_ptr<DataRepository> data_repository = nullptr)
        : ITaskExecutor(duckdb::make_uniq<GPUPipelineTaskQueue>(), config),
          data_repository_(std::move(data_repository)) {}

    // Destructor
    ~GPUPipelineExecutor() override = default;

    // Non-copyable but movable
    GPUPipelineExecutor(const GPUPipelineExecutor&) = delete;
    GPUPipelineExecutor& operator=(const GPUPipelineExecutor&) = delete;
    GPUPipelineExecutor(GPUPipelineExecutor&&) = default;
    GPUPipelineExecutor& operator=(GPUPipelineExecutor&&) = default;

    /**
     * Schedule a GPU pipeline task for execution
     * @param gpu_task The GPU pipeline task to schedule
     */
    void ScheduleGPUTask(duckdb::unique_ptr<GPUPipelineTask> gpu_task) {
        // Convert to ITask and use parent's Schedule method
        Schedule(std::move(gpu_task));
    }

    /**
     * Get the data repository used by this executor
     * @return Shared pointer to the data repository, may be nullptr
     */
    duckdb::shared_ptr<DataRepository> GetDataRepository() const {
        return data_repository_;
    }

    /**
     * Override the Schedule method to provide GPU-specific scheduling logic
     * @param task The task to schedule
     */
    void Schedule(duckdb::unique_ptr<ITask> task) override;

private:
    // Helper method to safely cast ITask to GPUPipelineTask
    GPUPipelineTask* CastToGPUPipelineTask(ITask* task);

    // push the output data batch to the Data Repository
    void PushPipelineOutput(duckdb::unique_ptr<DataBatch> data_batch, size_t pipeline_id, size_t idx);

private:
    // GPU-specific resources
    duckdb::shared_ptr<DataRepository> data_repository_;
};

} // namespace parallel
} // namespace sirius
} // namespace duckdb