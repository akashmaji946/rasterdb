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
#include "scan/duckdb_scan_task.hpp"
#include "data/data_repository.hpp"
#include "config.hpp"
#include "operator/gpu_physical_table_scan.hpp"
#include "gpu_buffer_manager.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/execution/execution_context.hpp"
#include "duckdb/parallel/task_executor.hpp"
#include "helper/helper.hpp"

namespace duckdb {

/**
 * @brief Executor for performing DuckDB table scan operations 
 * 
 * This executor is just handing out the task to duckdb scheduler, and converting the duckdb output chunk to a data batch
 * We also don't need to have a separate task queue for managing the scan tasks since we are just handing off the work to duckdb's scheduler
 * 
 * In the current implementation, we assume that we will not run out of CPU memory while scanning the data from DuckDB.
 */
class DuckDBScanExecutor {
    /**
     * @brief Construct a new DuckDBScanExecutor object
     * 
     * @param executor The task executor to use for scheduling scan tasks
     * @param function The table function to scan data from
     * @param context The execution context for the scan operation
     * @param op The GPU physical table scan operator associated with this executor
     * @param data_repository The data repository to push the output data batches to
     */
    DuckDBScanExecutor(TaskExecutor &executor, TableFunction& function_p, ExecutionContext& context_p,
                       GPUPhysicalTableScan& op_p, ::sirius::DataRepository& data_repository) :
        task_executor_(executor), function_(function_p), context_(context_p), op_(op_p), data_repository_(data_repository) {}

    /**
     * @brief Destroy the DuckDBScanExecutor object
     */    
    ~DuckDBScanExecutor();

    /** 
     * @brief Creates a new scan task and schedules it to the duckdb task scheduler
     * 
     * See gpu_physical_table_scan.hpp for an example
     */
    void createAndScheduleTask();
    
    /**
     * @brief Method to inform the DuckDB scheduler to work on the scheduled scan task
     * 
     * See gpu_physical_table_scan.hpp for an example
     */
    void workOnTask();

     /**
     * @brief Method to mark that this task is completed
     * 
     * This method informs that TaskCreator that the current scan task is completed so that it can
     * schedule more scan tasks. 
     * This method should be called after pushing the output of this task to the Data Repository.
     */
    void MarkTaskCompletion();
    
    /**
     * @brief Method to push the output of a scan task to the Data Repository
     * 
     * @param data_batch The data batch to push
     * @param pipeline_id The id of the pipeline that produced this data batch
     */
    void pushScanOutput(::sirius::unique_ptr<::sirius::DataBatch> data_batch, size_t pipeline_id);

private:
    ::sirius::DataRepository& data_repository_; // The data repository to push the output data batches to
    TaskExecutor &task_executor_; // The task executor to use for scheduling scan tasks
    TableFunction& function_; // The table function to scan data from
    ExecutionContext& context_; // The execution context for the scan operation
    GPUPhysicalTableScan& op_; // The GPU physical table scan operator associated with this executor
};

} // namespace duckdb