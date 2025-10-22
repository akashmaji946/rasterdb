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

// #pragma once
// #include "scan/duckdb_scan_task.hpp"
// #include "data/data_repository.hpp"
// #include "config.hpp"
// #include "operator/gpu_physical_table_scan.hpp"
// #include "duckdb/planner/filter/constant_filter.hpp"
// #include "duckdb/planner/filter/conjunction_filter.hpp"
// #include "duckdb/execution/execution_context.hpp"
// #include "parallel/task_executor.hpp"
// #include "helper/helper.hpp"

// namespace sirius {
// namespace parallel {

// /**
//  * @brief Executor specialized for performing DuckDB table scan operations.
//  *
//  * This executor manages a thread pool dedicated to executing DuckDB scan tasks that
//  * read data from DuckDB tables and convert it into DataBatch format for processing
//  * in the Sirius pipeline system. In the current implementation, we assume sufficient
//  * CPU memory is available during scanning operations.
//  */
// class DuckDBScanExecutor : public ITaskExecutor {
// public:
//     /**
//      * @brief Constructs a new DuckDBScanExecutor with task execution configuration
//      *
//      * @param config The task executor configuration (thread count, retry policy, etc.)
//      * @param data_repository Reference to the data repository for storing output data batches
//      */
//     explicit DuckDBScanExecutor(
//         TaskExecutorConfig config,
//         DataRepository& data_repository)
//         : ITaskExecutor(sirius::make_unique<DuckDBScanTaskQueue>(), config),
//           data_repository_(data_repository) {}

//     /**
//      * @brief Destructor for the DuckDBScanExecutor
//      */
//     ~DuckDBScanExecutor() override = default;

//     // Non-copyable but movable
//     DuckDBScanExecutor(const DuckDBScanExecutor&) = delete;
//     DuckDBScanExecutor& operator=(const DuckDBScanExecutor&) = delete;
//     DuckDBScanExecutor(DuckDBScanExecutor&&) = default;
//     DuckDBScanExecutor& operator=(DuckDBScanExecutor&&) = default;

//     /**
//      * @brief Schedules a DuckDB scan task for execution
//      *
//      * This is a type-safe wrapper that converts the scan task to the base ITask
//      * interface and delegates to the parent's Schedule method.
//      *
//      * @param scan_task The DuckDB scan task to schedule for execution
//      */
//     void ScheduleScanTask(sirius::unique_ptr<DuckDBScanTask> scan_task) {
//         // Convert to ITask and use parent's Schedule method
//         Schedule(std::move(scan_task));
//     }

//     /**
//      * @brief Schedules a task for execution with scan-specific logic
//      *
//      * Overrides the base class Schedule method to provide specialized scheduling
//      * behavior for DuckDB scan operations.
//      *
//      * @param task The task to schedule (must be a DuckDBScanTask)
//      */
//     void Schedule(sirius::unique_ptr<ITask> task) override;

//     /**
//      * @brief Main worker loop for executing scan tasks
//      *
//      * Each worker thread runs this loop to continuously pull and execute scan
//      * tasks from the queue. Handles DuckDB-specific operations and data conversion.
//      *
//      * @param worker_id The unique identifier for this worker thread
//      */
//     void WorkerLoop(int worker_id) override;

//     /**
//      * @brief Starts the executor and initializes worker threads
//      *
//      * Initializes the thread pool and begins accepting tasks for execution.
//      */
//     void Start() override;

//     /**
//      * @brief Stops the executor and cleanly shuts down worker threads
//      *
//      * Stops accepting new tasks and waits for all worker threads to complete
//      * their current tasks before shutting down.
//      */
//     void Stop() override;

// private:
//     DataRepository& data_repository_;  ///< Reference to the data repository for storing output
//     data batches
// };

// } // namespace parallel
// } // namespace sirius

#pragma once

// sirius
#include <parallel/task_executor.hpp>
#include <scan/duckdb_scan_task.hpp>
#include <scan/duckdb_scan_task_queue.hpp>

namespace sirius::parallel
{

//===--------------------------------------------------===//
// DuckDBScanTaskExecutor
//===--------------------------------------------------===//
class DuckDBScanTaskExecutor : public ITaskExecutor
{
public:
  //===----------Constructor----------===//
  explicit DuckDBScanTaskExecutor(TaskExecutorConfig config)
      : ITaskExecutor(sirius::make_unique<DuckDBScanTaskQueue>(), config)
  {}
};

} // namespace sirius::parallel