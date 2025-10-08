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

namespace duckdb {
namespace sirius {
namespace parallel {

/**
 * Downgrade-specific task executor that inherits from ITaskExecutor and uses
 * DowngradeTaskQueue as its scheduler. Manages a pool of threads to
 * execute downgrade tasks with specialized memory management for spilling operations.
 */
class DowngradeExecutor : public ITaskExecutor {
public:
    /**
     * Constructor that creates a DowngradeExecutor with a DowngradeTaskQueue scheduler
     * @param config Configuration for the task executor (thread count, retry policy, etc.)
     * @param reservation_manager Reference to the memory reservation manager
     * @param data_repository Optional data repository for data access
     */
    explicit DowngradeExecutor(
        TaskExecutorConfig config,
        duckdb::shared_ptr<DataRepository> data_repository = nullptr)
        : ITaskExecutor(duckdb::make_uniq<DowngradeTaskQueue>(), config),
          data_repository_(std::move(data_repository)) {

          }

    // Destructor
    ~DowngradeExecutor() override = default;

    // Non-copyable but movable
    DowngradeExecutor(const DowngradeExecutor&) = delete;
    DowngradeExecutor& operator=(const DowngradeExecutor&) = delete;
    DowngradeExecutor(DowngradeExecutor&&) = default;
    DowngradeExecutor& operator=(DowngradeExecutor&&) = default;

    /**
     * Schedule a downgrade task for execution
     * @param downgrade_task The downgrade task to schedule
     */
    void ScheduleDowngradeTask(duckdb::unique_ptr<DowngradeTask> downgrade_task) {
        // Convert to ITask and use parent's Schedule method
        Schedule(std::move(downgrade_task));
    }

    /**
     * Get the data repository used by this executor
     * @return Shared pointer to the data repository, may be nullptr
     */
    duckdb::shared_ptr<DataRepository> GetDataRepository() const {
        return data_repository_;
    }

    /**
     * Override the Schedule method to provide downgrade-specific scheduling logic
     * @param task The task to schedule
     */
    void Schedule(duckdb::unique_ptr<ITask> task) override;

private:
    // Helper method to safely cast ITask to DowngradeTask
    DowngradeTask* CastToDowngradeTask(ITask* task);

    // push the downgraded data batch to the Data Repository
    void PushDowngradeOutput(duckdb::unique_ptr<DataBatch> data_batch, size_t pipeline_id, size_t idx);

private:
    // Downgrade-specific resources
    duckdb::shared_ptr<DataRepository> data_repository_;
};

} // namespace parallel
} // namespace sirius
} // namespace duckdb