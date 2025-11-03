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
#include "creator/task_creator.hpp"
#include "gpu_pipeline_hashmap.hpp"
#include "task_completion.hpp"

namespace sirius {

/**
 * @brief Main task creator responsible for coordinating task scheduling across the system.
 * 
 * The sirius_task_creator serves as the central coordinator for task creation and scheduling
 * in the Sirius system. It monitors the data repository for new data batches,
 * creates appropriate tasks (scan or pipeline), and schedules them on the
 * corresponding executors. It also handles task completion notifications and
 * manages task dependencies.
 * 
 * This class implements the producer side of the producer-consumer pattern,
 * where it produces tasks that are consumed by various specialized executors.
 */
class sirius_task_creator : public itask_creator {
public:
    /**
     * @brief Constructs a new sirius_task_creator with required system components
     * 
     * @param data_repo_mgr Reference to the central data repository manager
     * @param gpu_pipeline_exec Reference to the GPU pipeline executor
     * @param duckdb_scan_exec Reference to the DuckDB scan executor
     */
    sirius_task_creator(data_repository_manager& data_repo_mgr,
        parallel::gpu_pipeline_executor &gpu_pipeline_exec,
        parallel::duckdb_scan_executor& duckdb_scan_exec);
          
    /**
     * @brief Destructor for sirius_task_creator
     */
    ~sirius_task_creator() = default;

    // Non-copyable but movable
    sirius_task_creator(const sirius_task_creator&) = delete;
    sirius_task_creator& operator=(const sirius_task_creator&) = delete;
    sirius_task_creator(sirius_task_creator&&) = default;
    sirius_task_creator& operator=(sirius_task_creator&&) = default;

    /**
     * @brief Scans the data repository for new data batches and creates pipeline tasks
     * 
     * @param pipeline_idx The index of the pipeline to scan for
     * @return bool True if new tasks were created, false otherwise
     */
    bool scan_repository(size_t pipeline_idx);

    /**
     * @brief Main worker loop for continuous task creation and scheduling
     * 
     * This method runs the main coordination loop that monitors for task completion
     * messages and creates new tasks as dependencies are satisfied.
     */
    void worker_loop() override;

    /**
     * @brief Sets the coordinator (GPUExecutor) for this task creator
     * 
     * @param coordinator Pointer to the GPU executor coordinator
     */
    void set_coordinator(duckdb::GPUExecutor* coordinator) {
        _coordinator = coordinator;
    }

    /**
     * @brief Sets the GPU pipeline hash map for pipeline lookups
     * 
     * @param pipeline_hashmap Shared pointer to the pipeline hash map
     */
    void set_gpu_pipeline_hashmap(sirius::shared_ptr<gpu_pipeline_hashmap> pipeline_hashmap) {
        _gpu_pipeline_hashmap = pipeline_hashmap;
    }

    /**
     * @brief Creates a new DuckDB scan task for the specified pipeline
     * 
     * @param pipeline_idx The index of the pipeline requiring a scan task
     * @return sirius::unique_ptr<parallel::duckdb_scan_task> The created scan task
     */
    sirius::unique_ptr<parallel::duckdb_scan_task> create_scan_task(size_t pipeline_idx);
    
    /**
     * @brief Creates a new GPU pipeline task with the provided data batches
     * 
     * @param pipeline_idx The index of the pipeline for this task
     * @param batch_views The data batch views to be processed by this task
     * @return sirius::unique_ptr<parallel::gpu_pipeline_task> The created pipeline task
     */
    sirius::unique_ptr<parallel::gpu_pipeline_task> create_pipeline_task(size_t pipeline_idx, 
        sirius::vector<sirius::unique_ptr<data_batch_view>> batch_views);

    /**
     * @brief Schedules a DuckDB scan task for execution
     * 
     * @param scan_task The scan task to schedule
     */
    void schedule_duckdb_scan(sirius::unique_ptr<parallel::duckdb_scan_task> scan_task);

    /**
     * @brief Schedules a GPU pipeline task for execution
     * 
     * @param gpu_pipeline_task The GPU pipeline task to schedule
     */
    void schedule_pipeline_task(sirius::unique_ptr<parallel::gpu_pipeline_task> gpu_pipeline_task);

private:
    task_completion_message_queue _task_completion_msg_queue;         ///< Queue for receiving task completion notifications
    data_repository_manager& _data_repo_mgr;                          ///< Reference to the central data repository manager
    parallel::gpu_pipeline_executor& _gpu_pipeline_exec;              ///< Reference to the GPU pipeline executor
    parallel::duckdb_scan_executor& _duckdb_scan_exec;                  ///< Reference to the DuckDB scan executor
    duckdb::GPUExecutor* _coordinator;                                ///< Pointer to the GPU executor coordinator
    sirius::shared_ptr<gpu_pipeline_hashmap> _gpu_pipeline_hashmap;     ///< Shared pointer to the GPU pipeline hash map
};

} // namespace sirius