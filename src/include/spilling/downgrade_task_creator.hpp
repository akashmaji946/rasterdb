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
#include "spilling/downgrade_executor.hpp"

namespace sirius {
namespace parallel {

/**
 * Downgrade-specific task creator that inherits from ITaskExecutor and uses
 * DowngradeTaskQueue as its scheduler. Manages a pool of threads to
 * schedule downgrade tasks
 */
class DowngradeTaskCreator : public ITaskExecutor {
public:
    /**
     * Constructor that creates a DowngradeTaskCreator with a DowngradeTaskQueue scheduler
     * @param config Configuration for the task executor (thread count, retry policy, etc.)
     * @param data_repository Optional data repository for data access
     */
    explicit DowngradeTaskCreator(
        TaskExecutorConfig config,
        DataRepository& data_repository)
        : ITaskExecutor(sirius::make_unique<DowngradeTaskQueue>(), config),
          data_repository_(data_repository) {}

    // Destructor
    ~DowngradeTaskCreator() override = default;

    // Non-copyable but movable
    DowngradeTaskCreator(const DowngradeTaskCreator&) = delete;
    DowngradeTaskCreator& operator=(const DowngradeTaskCreator&) = delete;
    DowngradeTaskCreator(DowngradeTaskCreator&&) = default;
    DowngradeTaskCreator& operator=(DowngradeTaskCreator&&) = default;

    /**
     * Schedule a downgrade task for execution
     * @param downgrade_task The downgrade task to schedule
     */
    void ScheduleDowngradeTask(sirius::unique_ptr<DowngradeTask> downgrade_task) {
        // Convert to ITask and use parent's Schedule method
        Schedule(std::move(downgrade_task));
    }

    /**
     * Override the Schedule method to provide downgrade-specific scheduling logic
     * @param task The task to schedule
     */
    void Schedule(sirius::unique_ptr<ITask> task) override;

private:
    // Helper method to safely cast ITask to DowngradeTask
    DowngradeTask* CastToDowngradeTask(ITask* task);

private:
    // Downgrade-specific resources
    DataRepository& data_repository_;
};

} // namespace parallel
} // namespace sirius