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
#include "spilling/downgrade_task.hpp"
#include "data/data_repository.hpp"

namespace sirius {
namespace parallel {

/**
 * @brief Executor specialized for performing memory downgrade operations across tier hierarchies.
 * 
 * The DowngradeExecutor inherits from ITaskExecutor and manages a pool of threads dedicated
 * to executing downgrade tasks that move data between different memory tiers (e.g., GPU to CPU,
 * CPU to disk). It uses a DowngradeTaskQueue for task scheduling and coordination.
 * 
 * While similar to GPUPipelineExecutor in its threading model, it includes specialized
 * memory management logic required for efficient spilling operations and tier management.
 */
class DowngradeExecutor : public ITaskExecutor {
public:
    /**
     * @brief Constructs a new DowngradeExecutor with task execution configuration
     * 
     * @param config Configuration for the task executor (thread count, retry policy, etc.)
     * @param data_repository Reference to the data repository for accessing and storing data batches
     */
    explicit DowngradeExecutor(
        TaskExecutorConfig config,
        DataRepository& data_repository)
        : ITaskExecutor(sirius::make_unique<DowngradeTaskQueue>(), config),
          data_repository_(data_repository) {

          }

    /**
     * @brief Destructor for the DowngradeExecutor.
     */
    ~DowngradeExecutor() override = default;

    // Non-copyable but movable
    DowngradeExecutor(const DowngradeExecutor&) = delete;
    DowngradeExecutor& operator=(const DowngradeExecutor&) = delete;
    DowngradeExecutor(DowngradeExecutor&&) = default;
    DowngradeExecutor& operator=(DowngradeExecutor&&) = default;

    /**
     * @brief Schedules a downgrade task for execution
     * 
     * This is a type-safe wrapper that converts the downgrade task to the base ITask
     * interface and delegates to the parent's Schedule method.
     * 
     * @param downgrade_task The downgrade task to schedule for execution
     */
    void ScheduleDowngradeTask(sirius::unique_ptr<DowngradeTask> downgrade_task) {
        // Convert to ITask and use parent's Schedule method
        Schedule(std::move(downgrade_task));
    }

    /**
     * @brief Schedules a task for execution with downgrade-specific logic
     * 
     * Overrides the base class Schedule method to provide specialized scheduling
     * behavior for downgrade operations, including memory tier validation and
     * resource management.
     * 
     * @param task The task to schedule (must be a DowngradeTask)
     */
    void Schedule(sirius::unique_ptr<ITask> task) override;

    /**
     * @brief Main worker loop for executing downgrade tasks
     * 
     * Each worker thread runs this loop to continuously pull and execute downgrade
     * tasks from the queue. Includes specialized error handling and resource cleanup
     * for memory tier operations.
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
     * @brief Safely casts ITask to DowngradeTask with type validation
     * 
     * @param task The base task pointer to cast
     * @return DowngradeTask* The casted DowngradeTask pointer
     * @throws std::bad_cast if the task is not of type DowngradeTask
     */
    DowngradeTask* CastToDowngradeTask(ITask* task);

private:
    DataRepository& data_repository_;  ///< Reference to the data repository for accessing data during downgrade operations
};

} // namespace parallel
} // namespace sirius