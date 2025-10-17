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
 * @brief Executor that inherits from ITaskExecutor to perform downgrade tasks
 * 
 * The DowngradeExecutor manages a pool of threads to perform downgrade tasks as well as DowngradeTaskQueue
 * that actually manages the scheduling of the tasks.  
 * 
 * While it is relatively similar to the GPUPipelineExecutor in many of its mechanisms, it also specialized memory management 
 * logic needed for spilling operations.
 */
class DowngradeExecutor : public ITaskExecutor {
public:
    /**
     * @brief Constructor that creates a DowngradeExecutor with a DowngradeTaskQueue scheduler
     * 
     * @param config Configuration for the task executor (thread count, retry policy, etc.)
     * @param reservation_manager Reference to the memory reservation manager
     * @param data_repository Optional data repository for data access
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
     * @brief Schedule a downgrade task for execution
     * 
     * @param downgrade_task The downgrade task to schedule
     */
    void ScheduleDowngradeTask(sirius::unique_ptr<DowngradeTask> downgrade_task) {
        // Convert to ITask and use parent's Schedule method
        Schedule(std::move(downgrade_task));
    }

    /**
     * @brief Override the Schedule method to provide downgrade-specific scheduling logic
     * 
     * @param task The task to schedule
     */
    void Schedule(sirius::unique_ptr<ITask> task) override;

    /**
     * @brief Main worker loop for executing tasks
     * 
     * @param worker_id The identifier for the worker thread
     */
    void WorkerLoop(int worker_id) override;

    // Start worker threads
    void Start() override;

    // Stop accepting new tasks, and join worker threads.
    void Stop() override;

private:
    /**
     * @brief Helper method to safely cast ITask to DowngradeTask
     * 
     * @param task The base task to cast
     * @return Pointer to the DowngradeTask
     */
    DowngradeTask* CastToDowngradeTask(ITask* task);

private:
    DataRepository& data_repository_; // The data repository to access the data for downgrading
};

} // namespace parallel
} // namespace sirius