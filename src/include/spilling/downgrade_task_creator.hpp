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
class DowngradeTaskCreator {
public:
    /**
     * Constructor that creates a DowngradeTaskCreator with a DowngradeTaskQueue scheduler
     * @param config Configuration for the task executor (thread count, retry policy, etc.)
     * @param data_repository Optional data repository for data access
     */
    DowngradeTaskCreator(
        DataRepository& data_repository, DowngradeExecutor &downgrade_executor)
        : data_repository_(data_repository), downgrade_executor_(downgrade_executor) {}

    /**
     * Schedule a downgrade task for execution
     * @param downgrade_task The downgrade task to schedule
     */
    void Schedule(sirius::unique_ptr<DowngradeTask> downgrade_task);

private:
    // Downgrade-specific resources
    DataRepository& data_repository_;
    DowngradeExecutor& downgrade_executor_;
};

} // namespace parallel
} // namespace sirius