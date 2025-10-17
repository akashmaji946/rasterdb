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
#include "parallel/task_creator.hpp"
#include "gpu_pipeline_hashmap.hpp"
#include "task_completion.hpp"

namespace sirius {

/**
 * @brief Main task creator responsible for coordinating task scheduling across the system.
 * 
 * The TaskCreator serves as the central coordinator for task creation and scheduling
 * in the Sirius system. It monitors the data repository for new data batches,
 * creates appropriate tasks (scan or pipeline), and schedules them on the
 * corresponding executors. It also handles task completion notifications and
 * manages task dependencies.
 * 
 * This class implements the producer side of the producer-consumer pattern,
 * where it produces tasks that are consumed by various specialized executors.
 */
class TaskCreator : public ITaskCreator {
public:
    /**
     * @brief Constructs a new TaskCreator with required system components
     * 
     * @param data_repository Reference to the central data repository
     * @param gpu_pipeline_executor Reference to the GPU pipeline executor
     * @param duckdb_scan_executor Reference to the DuckDB scan executor
     */
    TaskCreator(DataRepository& data_repository,
        parallel::GPUPipelineExecutor &gpu_pipeline_executor,
        parallel::DuckDBScanExecutor& duckdb_scan_executor);
          
    /**
     * @brief Destructor for TaskCreator
     */
    ~TaskCreator() = default;

    // Non-copyable but movable
    TaskCreator(const TaskCreator&) = delete;
    TaskCreator& operator=(const TaskCreator&) = delete;
    TaskCreator(TaskCreator&&) = default;
    TaskCreator& operator=(TaskCreator&&) = default;

    /**
     * @brief Scans the data repository for new data batches and creates pipeline tasks
     * 
     * @param pipeline_idx The index of the pipeline to scan for
     * @return bool True if new tasks were created, false otherwise
     */
    bool ScanRepository(size_t pipeline_idx);

    /**
     * @brief Main worker loop for continuous task creation and scheduling
     * 
     * This method runs the main coordination loop that monitors for task completion
     * messages and creates new tasks as dependencies are satisfied.
     */
    void WorkerLoop() override;

    /**
     * @brief Sets the coordinator (GPUExecutor) for this task creator
     * 
     * @param coordinator Pointer to the GPU executor coordinator
     */
    void SetCoordinator(duckdb::GPUExecutor* coordinator) {
        coordinator_ = coordinator;
    }

    /**
     * @brief Sets the GPU pipeline hash map for pipeline lookups
     * 
     * @param gpu_pipeline_hashmap Shared pointer to the GPU pipeline hash map
     */
    void SetGPUPipelineHashMap(sirius::shared_ptr<GPUPipelineHashMap> gpu_pipeline_hashmap) {
        gpu_pipeline_hashmap_ = gpu_pipeline_hashmap;
    }

    /**
     * @brief Creates a new DuckDB scan task for the specified pipeline
     * 
     * @param pipeline_idx The index of the pipeline requiring a scan task
     * @return sirius::unique_ptr<parallel::DuckDBScanTask> The created scan task
     */
    sirius::unique_ptr<parallel::DuckDBScanTask> CreateScanTask(size_t pipeline_idx);
    
    /**
     * @brief Creates a new GPU pipeline task with the provided data batches
     * 
     * @param pipeline_idx The index of the pipeline for this task
     * @param data_batches The data batches to be processed by this task
     * @return sirius::unique_ptr<parallel::GPUPipelineTask> The created pipeline task
     */
    sirius::unique_ptr<parallel::GPUPipelineTask> CreatePipelineTask(size_t pipeline_idx, 
        sirius::vector<sirius::unique_ptr<DataBatch>> data_batches);

    /**
     * @brief Schedules a DuckDB scan task for execution
     * 
     * @param scan_task The scan task to schedule
     */
    void ScheduleDuckDBScan(sirius::unique_ptr<parallel::DuckDBScanTask> scan_task);

    /**
     * @brief Schedules a GPU pipeline task for execution
     * 
     * @param gpu_pipeline_task The GPU pipeline task to schedule
     */
    void SchedulePipelineTask(sirius::unique_ptr<parallel::GPUPipelineTask> gpu_pipeline_task);

    /**
     * @brief Starts the task creator worker loop
     */
    void Run();

    /**
     * @brief Signals the task creator to wake up and check for new work
     */
    void Signal();

    /**
     * @brief Waits for the task creator to be signaled
     */
    void Wait();

    /**
     * @brief Generates a new unique task identifier
     * 
     * @return uint64_t A new unique task ID
     */
    uint64_t GetNextTaskId() {
        return next_task_id_++;
    }

private:
    TaskCompletionMessageQueue task_completion_message_queue_;         ///< Queue for receiving task completion notifications
    DataRepository& data_repository_;                                  ///< Reference to the central data repository
    parallel::GPUPipelineExecutor& gpu_pipeline_executor_;             ///< Reference to the GPU pipeline executor
    parallel::DuckDBScanExecutor& duckdb_scan_executor_;               ///< Reference to the DuckDB scan executor
    duckdb::GPUExecutor* coordinator_;                                 ///< Pointer to the GPU executor coordinator
    sirius::shared_ptr<GPUPipelineHashMap> gpu_pipeline_hashmap_;      ///< Shared pointer to the GPU pipeline hash map

    sirius::atomic<uint64_t> next_task_id_ = 0;                        ///< Atomic counter for generating unique task IDs
	sirius::mutex mtx;                                                 ///< Mutex for synchronization
	std::condition_variable cv;                                        ///< Condition variable for thread coordination
	bool ready = false;                                                ///< Flag indicating readiness state
};

} // namespace sirius