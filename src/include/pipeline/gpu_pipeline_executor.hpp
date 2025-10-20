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
 * @brief Executor specialized for executing GPU pipeline operations.
 * 
 * This executor inherits from ITaskExecutor and uses a GPUPipelineTaskQueue for
 * task scheduling. It manages a pool of threads dedicated to executing GPU pipeline
 * tasks with specialized GPU resource management.
 */
class GPUPipelineExecutor : public ITaskExecutor {
public:
    /**
     * @brief Constructs a new GPUPipelineExecutor with task execution configuration
     * 
     * @param config Configuration for the task executor (thread count, retry policy, etc.)
     * @param data_repository Reference to the data repository for accessing and storing data batches
     */
    explicit GPUPipelineExecutor(
        TaskExecutorConfig config,
        DataRepository& data_repository)
        : ITaskExecutor(sirius::make_unique<GPUPipelineTaskQueue>(), config),
          data_repository_(data_repository) {}

    /**
     * @brief Destructor for the GPUPipelineExecutor.
     */
    ~GPUPipelineExecutor() override = default;

    // Non-copyable but movable
    GPUPipelineExecutor(const GPUPipelineExecutor&) = delete;
    GPUPipelineExecutor& operator=(const GPUPipelineExecutor&) = delete;
    GPUPipelineExecutor(GPUPipelineExecutor&&) = default;
    GPUPipelineExecutor& operator=(GPUPipelineExecutor&&) = default;

    /**
     * @brief Schedules a GPU pipeline task for execution
     * 
     * This is a type-safe wrapper that converts the GPU task to the base ITask
     * interface and delegates to the parent's Schedule method.
     * 
     * @param gpu_task The GPU pipeline task to schedule for execution
     */
    void ScheduleGPUTask(sirius::unique_ptr<GPUPipelineTask> gpu_task) {
        // Convert to ITask and use parent's Schedule method
        Schedule(std::move(gpu_task));
    }

    /**
     * @brief Schedules a task for execution with GPU-specific logic
     * 
     * Overrides the base class Schedule method to provide specialized scheduling
     * behavior for GPU pipeline operations, including resource allocation and
     * GPU context management.
     * 
     * @param task The task to schedule (must be a GPUPipelineTask)
     */
    void Schedule(sirius::unique_ptr<ITask> task) override;

    /**
     * @brief Main worker loop for executing GPU pipeline tasks
     * 
     * Each worker thread runs this loop to continuously pull and execute GPU
     * pipeline tasks from the queue. Handles GPU-specific operations including
     * kernel launches, memory transfers, and synchronization.
     * 
     * @param worker_id The unique identifier for this worker thread
     */
    void WorkerLoop(int worker_id) override;

    /**
     * @brief Starts the executor and initializes worker threads
     * 
     * Initializes the thread pool and begins accepting tasks for execution.
     */
    void Start() override;

    /**
     * @brief Stops the executor and cleanly shuts down worker threads
     * 
     * Stops accepting new tasks and waits for all worker threads to complete
     * their current tasks before shutting down.
     */
    void Stop() override;

private:
    /**
     * @brief Safely casts ITask to GPUPipelineTask with type validation
     * 
     * @param task The ITask pointer to cast
     * @return GPUPipelineTask* The casted GPUPipelineTask pointer
     * @throws std::bad_cast if the task is not of type GPUPipelineTask
     */
    GPUPipelineTask* CastToGPUPipelineTask(ITask* task);

private:
    DataRepository& data_repository_;  ///< Reference to the data repository for accessing and storing data batches
};

} // namespace parallel
} // namespace sirius