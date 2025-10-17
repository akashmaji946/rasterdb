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
#include "parallel/task_creator.hpp"

namespace sirius {

/**
 * @brief Task creator specialized for managing downgrade operations in memory tier hierarchies.
 * 
 * This class inherits from ITaskCreator and is responsible for creating and scheduling
 * downgrade tasks that move data between different memory tiers (e.g., GPU to CPU,
 * CPU to disk). It coordinates with the DowngradeExecutor to manage the execution
 * of these memory management operations.
 */
class DowngradeTaskCreator : public ITaskCreator {
public:
    /**
     * @brief Constructs a new DowngradeTaskCreator object
     * 
     * @param data_repository Reference to the data repository for accessing and storing data batches
     * @param downgrade_executor Reference to the downgrade executor for task execution
     */
    DowngradeTaskCreator(
        DataRepository& data_repository, parallel::DowngradeExecutor &downgrade_executor)
        : ITaskCreator(), data_repository_(data_repository), downgrade_executor_(downgrade_executor) {}

    /**
     * @brief Destructor for the DowngradeTaskCreator
     */
    ~DowngradeTaskCreator() = default;

    // Non-copyable but movable
    DowngradeTaskCreator(const DowngradeTaskCreator&) = delete;
    DowngradeTaskCreator& operator=(const DowngradeTaskCreator&) = delete;
    DowngradeTaskCreator(DowngradeTaskCreator&&) = default;
    DowngradeTaskCreator& operator=(DowngradeTaskCreator&&) = default;

    /**
     * @brief Main worker loop for the downgrade task creator
     * 
     * This method continuously monitors for memory pressure and creates downgrade
     * tasks as needed to move data from higher-tier to lower-tier memory.
     */
    void WorkerLoop() override;

    /**
     * @brief Schedules a downgrade task for execution
     * 
     * @param downgrade_task The downgrade task to schedule for execution
     */
    void Schedule(sirius::unique_ptr<parallel::DowngradeTask> downgrade_task);

private:
    DataRepository& data_repository_;                    ///< Reference to the data repository for data access
    parallel::DowngradeExecutor& downgrade_executor_;    ///< Reference to the downgrade executor for task execution
};

} // namespace sirius