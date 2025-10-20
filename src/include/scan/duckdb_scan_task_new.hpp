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

// sirius
#include <data/data_repository.hpp>
#include <gpu_buffer_manager.hpp>
#include <helper/helper.hpp>
#include <parallel/task.hpp>
#include <task_completion.hpp>

// duckdb
#include <duckdb/common/types.hpp>
#include <duckdb/execution/operator/scan/physical_table_scan.hpp>
#include <duckdb/function/table_function.hpp>

// standard library
#include <atomic>
#include <climits>

namespace sirius::parallel
{

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define CEIL_DIV_8(x) (((x) + 7) >> 3)
#define MOD_8(x) ((x) & (7))
#define MUL_8(x) ((x) << 3)
#define DIV_8(x) ((x) >> 3)
#define MASK(x) ((1U << (x)) - 1U)

using idx_t = duckdb::idx_t;
class DuckDBScanExecutor; // Forward declaration

//===--------------------------------------------------===//
// DuckDBScanTaskGlobalState
//===--------------------------------------------------===//
class DuckDBScanTaskGlobalState
    : public ITaskGlobalState
    , public duckdb::GlobalSourceState
{
public:
  //----------Constructor----------//
  DuckDBScanTaskGlobalState(uint64_t pipeline_id,
                            duckdb::ClientContext& context,
                            const duckdb::PhysicalTableScan& op);

  //----------Methods----------//
  idx_t MaxThreads() override
  {
    return max_threads;
  }

  duckdb::optional_ptr<duckdb::TableFilterSet>
  GetTableFilters(const duckdb::PhysicalTableScan& op) const
  {
    return table_filters ? table_filters.get() : op.table_filters.get();
  }

  bool IsSourceDrained() const
  {
    return source_drained.load(std::memory_order_acquire);
  }

  void SetSourceDrained()
  {
    source_drained.store(true, std::memory_order_release);
  }

  //----------Fields----------//
  std::atomic<bool> source_drained{false};
  idx_t max_threads = 0;
  uint64_t pipeline_id;
  unique_ptr<duckdb::GlobalTableFunctionState> global_tf_state;
  unique_ptr<duckdb::TableFilterSet> table_filters;
};

//===--------------------------------------------------===//
// DuckDBScanTaskLocalState
//===--------------------------------------------------===//
class DuckDBScanTaskLocalState
    : public ITaskLocalState
    , public duckdb::LocalSourceState
{
public:
  // static constexpr size_t DEFAULT_TARGET_BYTES = 2ULL << 30; // ~2GB
  static constexpr size_t DEFAULT_TARGET_BYTES = 256ULL << 20; // 256MB
  static constexpr size_t DEFAULT_VARCHAR_SIZE = 256;

public:
  //----------Constructor/Destructor----------//
  DuckDBScanTaskLocalState(DataRepository& data_repository,
                           TaskCompletionMessageQueue& message_queue,
                           DuckDBScanExecutor& executor,
                           const DuckDBScanTaskGlobalState& gstate,
                           duckdb::ExecutionContext& context,
                           const duckdb::PhysicalTableScan& op);
  ~DuckDBScanTaskLocalState() override;

  //----------Fields----------//
  size_t num_columns;
  vector<duckdb::LogicalType> scanned_types; ///< Types of the scanned columns
  vector<size_t> column_sizes;               ///< Size of each DuckDB column in bytes
  size_t max_type_size = 0;                  ///< Maximum size of any single type in bytes

  duckdb::DataChunk chunk; ///< DataChunk buffer
  vector<uint8_t*> data_ptrs;
  vector<uint8_t*> mask_ptrs;
  vector<uint64_t*> offset_ptrs;
  vector<size_t> byte_offsets;  ///< Current byte offsets in data buffers
  size_t row_offset        = 0; ///< Current row offset in buffers
  size_t bytes_accumulated = 0; ///< Total bytes accumulated so far by the scan

  DataRepository& data_repository; ///< Reference to the data repository to push DataBatches to
  TaskCompletionMessageQueue&
    message_queue; ///< Message queue to notify TaskCreator about completion of the scan task
  DuckDBScanExecutor&
    executor; ///<  Reference to the calling executor to schedule additional scan tasks if necessary

  // Execution drivers
  unique_ptr<duckdb::LocalTableFunctionState> local_tf_state;
  duckdb::ExecutionContext& context;
  const duckdb::PhysicalTableScan& op;
};

//===--------------------------------------------------===//
// DuckDBScanTask
//===--------------------------------------------------===//
class DuckDBScanTask : public ITask
{
public:
  //----------Constructor----------//
  DuckDBScanTask(uint64_t task_id,
                 unique_ptr<DuckDBScanTaskLocalState> local_state,
                 shared_ptr<DuckDBScanTaskGlobalState> global_state)
      : task_id(task_id)
      , ITask(std::move(local_state), std::move(global_state))
  {}

  //----------Methods----------//
  void Execute() override;

  //----------Fields----------//
  uint64_t task_id;
};

} // namespace sirius::parallel