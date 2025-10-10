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

// This executor is just handling out the task to duckdb scheduler, and converting the duckdb output chunk to a data batch
// TODO: one idea to make scan executor work is by having each thread continue calling 'function' until it accumulates 2GB of data
// then convert it into Data Batch, push it to repository, then continue scan to produce a new data batch, until the scan is done.
// For the first step, we assume that we will not run out of CPU memory.
class DuckDBScanExecutor {
    DuckDBScanExecutor(TaskExecutor &executor, TableFunction& function_p, ExecutionContext& context_p,
                       GPUPhysicalTableScan& op_p, ::sirius::DataRepository& data_repository) :
        task_executor_(executor), function_(function_p), context_(context_p), op_(op_p), data_repository_(data_repository) {}

    ~DuckDBScanExecutor();

    // Create a ScanTask and schedule it to duckdb task scheduler (see example in gpu_physical_table_scan.cpp)
    void createAndScheduleTask();

    // Tell DuckDB scheduler to work on the scheduled task until it's done (see example in gpu_physical_table_scan.cpp)
    void workOnTask();

    // Convert the output chunk from duckdb to a DataBatch
    void convertToDataBatch();

    // Push the output DataBatch to Data Repository
    void pushScanOutput(::sirius::unique_ptr<::sirius::DataBatch> data_batch, size_t pipeline_id, size_t idx);

private:
    ::sirius::DataRepository& data_repository_;
    TaskExecutor &task_executor_;
    TableFunction& function_;
    ExecutionContext& context_;
    GPUPhysicalTableScan& op_;
};

} // namespace duckdb