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
        DataRepository& data_repository)
        : ITaskExecutor(sirius::make_unique<GPUPipelineTaskQueue>(), config),
          data_repository_(data_repository) {}

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
    void ScheduleGPUTask(sirius::unique_ptr<GPUPipelineTask> gpu_task) {
        // Convert to ITask and use parent's Schedule method
        Schedule(std::move(gpu_task));
    }

    /**
     * Override the Schedule method to provide GPU-specific scheduling logic
     * @param task The task to schedule
     */
    void Schedule(sirius::unique_ptr<ITask> task) override;

private:
    // Helper method to safely cast ITask to GPUPipelineTask
    GPUPipelineTask* CastToGPUPipelineTask(ITask* task);

private:
    // GPU-specific resources
    DataRepository& data_repository_;
};

} // namespace parallel
} // namespace sirius