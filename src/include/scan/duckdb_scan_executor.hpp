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
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/execution/execution_context.hpp"
#include "parallel/task_executor.hpp"
#include "helper/helper.hpp"

namespace sirius {
namespace parallel {

/**
 * @brief Executor for performing DuckDB table scan operations
 * 
 * In the current implementation, we assume that we will not run out of CPU memory while scanning the data from DuckDB.
 */
class DuckDBScanExecutor : public ITaskExecutor {
public:
    /**
     * @brief Construct a new DuckDBScanExecutor object with task executor configuration
     * 
     * @param config The task executor configuration
     * @param data_repository The data repository to push the output data batches to
     * @param function_p The table function to scan data from
     * @param context_p The execution context for the scan operation
     * @param op_p The GPU physical table scan operator associated with this executor
     */
    explicit DuckDBScanExecutor(
        TaskExecutorConfig config,
        DataRepository& data_repository, duckdb::TableFunction* function_p, duckdb::ExecutionContext* context_p,
                       duckdb::GPUPhysicalTableScan* op_p)
        : ITaskExecutor(sirius::make_unique<DuckDBScanTaskQueue>(), config),
          data_repository_(data_repository), function_(function_p), context_(context_p), op_(op_p) {}

    /**
     * @brief Construct a new DuckDBScanExecutor object with task executor configuration
     * 
     * @param config The task executor configuration
     * @param data_repository The data repository to push the output data batches to
     */
    explicit DuckDBScanExecutor(
        TaskExecutorConfig config,
        DataRepository& data_repository)
        : ITaskExecutor(sirius::make_unique<DuckDBScanTaskQueue>(), config),
          data_repository_(data_repository) {}

    /**
     * @brief Destructor for the GPUPipelineExecutor.
     */
    ~DuckDBScanExecutor() override = default;

    // Non-copyable but movable
    DuckDBScanExecutor(const DuckDBScanExecutor&) = delete;
    DuckDBScanExecutor& operator=(const DuckDBScanExecutor&) = delete;
    DuckDBScanExecutor(DuckDBScanExecutor&&) = default;
    DuckDBScanExecutor& operator=(DuckDBScanExecutor&&) = default;

    void SetExecutionContext(duckdb::ExecutionContext* context_p) {
        context_ = context_p;
    }

    void SetTableFunction(duckdb::TableFunction* function_p) {
        function_ = function_p;
    }

    void SetPhysicalTableScan(duckdb::GPUPhysicalTableScan* op_p) {
        op_ = op_p;
    }

    /**
     * @brief Schedule a DuckDB scan task for execution by converting it to ITask
     * 
     * @param scan_task The DuckDB scan task to schedule
     */
    void ScheduleScanTask(sirius::unique_ptr<DuckDBScanTask> scan_task) {
        // Convert to ITask and use parent's Schedule method
        Schedule(std::move(scan_task));
    }

    /**
     * @brief Override the Schedule method to provide GPU-specific scheduling logic
     * 
     * @param task The task to schedule
     */
    void Schedule(sirius::unique_ptr<ITask> task) override;

private:
    DataRepository& data_repository_; // The data repository to push the output data batches to
    duckdb::TableFunction* function_; // The table function to scan data from
    duckdb::ExecutionContext* context_; // The execution context for the scan operation
    duckdb::GPUPhysicalTableScan* op_; // The GPU physical table scan operator associated with this executor
};

} // namespace parallel
} // namespace sirius