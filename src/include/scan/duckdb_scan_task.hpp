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
#include <scan/duckdb_physical_table_scan.hpp>
#include <sirius_context.hpp>
#include <task_completion.hpp>

// duckdb
#include <duckdb/common/types.hpp>
#include <duckdb/execution/operator/scan/physical_table_scan.hpp>
#include <duckdb/function/table_function.hpp>

// standard library
#include <atomic>
#include <climits>

namespace duckdb
{

// Simpler duckdb scan implementation
namespace experimental
{

// Forward declarations
class BatchSinkGlobalState;
class BatchSinkLocalState;
struct BatchBuilder ///< Dummy builder
{
  vector<LogicalType> types;
  size_t rows_accumulated  = 0;
  size_t bytes_accumulated = 0;
  explicit BatchBuilder(const vector<LogicalType>& types)
      : types(types)
  {}
  sirius::unique_ptr<sirius::DataBatch> MakeDataBatch(uint64_t batch_id) // dummy placeholder
  {
    return sirius::make_unique<sirius::DataBatch>(batch_id, nullptr);
  }
  void SliceChunk(DataChunk& chunk, BatchSinkGlobalState& g, BatchSinkLocalState& l);
  bool BatchIsReady(size_t target_bytes) const
  {
    return true;
  }
  void Reset()
  {
    rows_accumulated  = 0;
    bytes_accumulated = 0;
  }
};

struct BatchSinkGlobalState : public GlobalSinkState
{
  BatchSinkGlobalState(const vector<LogicalType>& types,
                       sirius::DataRepository& data_repository,
                       size_t target_bytes)
      : types(types)
      , data_repository(data_repository)
      , target_bytes(target_bytes)
  {}

  const vector<LogicalType> types;
  sirius::DataRepository& data_repository;
  const size_t target_bytes;
  std::atomic<uint64_t> batches_emitted{0};
};

struct BatchSinkLocalState : public LocalSinkState
{
  explicit BatchSinkLocalState(const vector<LogicalType>& types)
      : builder(types)
  {}

  BatchBuilder builder; /// TODO: initialize
};

class PhysicalBatchSink final : public PhysicalOperator
{
public:
  PhysicalBatchSink(vector<LogicalType> types,
                    idx_t estimated_cardinality,
                    unique_ptr<PhysicalOperator> child,
                    sirius::DataRepository& repo,
                    sirius::TaskCompletionMessageQueue& message_queue,
                    uint64_t pipeline_id,
                    uint64_t task_id,
                    size_t target_bytes)
      : PhysicalOperator(PhysicalOperatorType::EXTENSION, std::move(types), estimated_cardinality)
      , repo(repo)
      , message_queue(message_queue)
      , pipeline_id(pipeline_id)
      , task_id(task_id)
      , target_bytes(target_bytes)
  {
    children.push_back(std::move(child));
  }

  unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext& ctx) const override
  {
    return make_uniq<BatchSinkLocalState>(types);
  }
  unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext& ctx) const override
  {
    return make_uniq<BatchSinkGlobalState>(types, repo, target_bytes);
  }
  bool IsSink() const override
  {
    return true;
  }
  bool ParallelSink() const override
  {
    return true;
  }
  bool SinkOrderDependent() const override
  {
    return false;
  }

  // SINK, COMBINE, FINALIZE
  SinkResultType Sink(ExecutionContext&, DataChunk& chunk, OperatorSinkInput& input) const override
  {
    // Get global and local states
    auto& g = input.global_state.Cast<BatchSinkGlobalState>();
    auto& l = input.local_state.Cast<BatchSinkLocalState>();

    // Slice the chunk into columns and append to builder buffers
    l.builder.SliceChunk(chunk, g, l);

    if (l.builder.BatchIsReady(target_bytes))
    {
      PushDataBatch(g, l);
    }
    return duckdb::SinkResultType::NEED_MORE_INPUT;
  }

  SinkCombineResultType Combine(ExecutionContext&, OperatorSinkCombineInput& input) const override
  {
    // Get global and local states
    auto& g = input.global_state.Cast<BatchSinkGlobalState>();
    auto& l = input.local_state.Cast<BatchSinkLocalState>();

    // Push a data batch if there are remaining rows
    if (l.builder.rows_accumulated > 0)
    {
      PushDataBatch(g, l);
    }
    return SinkCombineResultType::FINISHED;
  }

  void PushDataBatch(BatchSinkGlobalState& g, BatchSinkLocalState& l) const
  {
    // Push data batch to the data repository
    const auto batch_id = g.data_repository.GetNextDataBatchId();
    auto batch          = l.builder.MakeDataBatch(batch_id);
    g.data_repository.AddNewDataBatch(pipeline_id, std::move(batch));

    // Send a message to the message queue that a batch is ready
    message_queue.EnqueueMessage(
      sirius::make_unique<sirius::TaskCompletionMessage>(task_id,
                                                         pipeline_id,
                                                         sirius::Source::SCAN));

    // Reset the builder for the next batch
    l.builder.Reset();

    // Update emitted batch count
    g.batches_emitted.fetch_add(1, std::memory_order_relaxed);
  }

  SinkFinalizeType
  Finalize(Pipeline&, Event&, ClientContext&, OperatorSinkFinalizeInput&) const override
  {
    // Nothing to do
    return duckdb::SinkFinalizeType::READY;
  }

private:
  sirius::TaskCompletionMessageQueue& message_queue;
  sirius::DataRepository& repo;
  size_t target_bytes;
  uint64_t pipeline_id;
  uint64_t task_id;
};

} // namespace experimental

} // namespace duckdb

namespace sirius::parallel
{

using idx_t = duckdb::idx_t;

namespace experimental
{
class DuckDBScanTaskGlobalState : public ITaskGlobalState
{
public:
  DuckDBScanTaskGlobalState() = default; // Empty
};
class DuckDBScanTaskLocalState : public ITaskLocalState
{
  static constexpr size_t DEFAULT_TARGET_BATCH_BYTES = 256ULL << 20; // 256MB
public:
  DuckDBScanTaskLocalState(uint64_t pipeline_id,
                           uint64_t task_id,
                           duckdb::unique_ptr<duckdb::PhysicalTableScan> op,
                           duckdb::ClientContext& context,
                           DataRepository& data_repository,
                           TaskCompletionMessageQueue& message_queue,
                           size_t target_batch_bytes = DEFAULT_TARGET_BATCH_BYTES)
      : pipeline_id(pipeline_id)
      , context(context)
      , data_repository(data_repository)
      , message_queue(message_queue)
      , target_batch_bytes(target_batch_bytes)
  {
    // Copy out what you need BEFORE moving
    duckdb::vector<duckdb::LogicalType> sink_types = op->types;
    duckdb::idx_t est_card                         = op->estimated_cardinality;

    //auto child = duckdb::make_uniq<duckdb::PhysicalOperator>(std::move(op));

    batch_sink = duckdb::make_uniq<duckdb::experimental::PhysicalBatchSink>(
      std::move(sink_types), // vector by value in sink ctor
      est_card,
      std::move(op),
      data_repository,
      message_queue,
      pipeline_id,
      task_id,
      target_batch_bytes);
  }

  uint64_t pipeline_id;
  duckdb::ClientContext& context;
  duckdb::unique_ptr<duckdb::experimental::PhysicalBatchSink> batch_sink;
  DataRepository& data_repository;
  TaskCompletionMessageQueue& message_queue;
  size_t target_batch_bytes;
};

class DuckDBScanTask : public ITask
{
public:
  DuckDBScanTask(sirius::unique_ptr<DuckDBScanTaskLocalState> local_state,
                 sirius::shared_ptr<DuckDBScanTaskGlobalState> global_state)
      : sirius::parallel::ITask(std::move(local_state), std::move(global_state))
  {}

  void Execute() override;
  void SetNumThreads(uint64_t num_threads)
  {
    auto& l     = local_state_->Cast<DuckDBScanTaskLocalState>();
    auto& sched = duckdb::TaskScheduler::GetScheduler(l.context);
    // The duckdb counts the thread executing Execute() as an external thread,
    // so we set external_threads = 1.
    sched.SetThreads(num_threads, 1);
  }
};
} // namespace experimental

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
                            const duckdb::DuckDBPhysicalTableScan& op);

  //----------Methods----------//
  idx_t MaxThreads() override
  {
    return max_threads;
  }

  duckdb::optional_ptr<duckdb::TableFilterSet>
  GetTableFilters(const duckdb::DuckDBPhysicalTableScan& op) const
  {
    return table_filters ? table_filters.get() : op.physical_table_scan_ptr->table_filters.get();
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
  static constexpr size_t DEFAULT_TARGET_BYTES  = 256ULL << 20; // 256MB
  static constexpr size_t DEFAULT_VARCHAR_BYTES = 256ULL;       // 256B

public:
  //===----------Constructor/Destructor----------===//
  DuckDBScanTaskLocalState(TaskCompletionMessageQueue& message_queue,
                           const DuckDBScanTaskGlobalState& gstate,
                           duckdb::ExecutionContext& context,
                           const duckdb::DuckDBPhysicalTableScan& pts);
  ~DuckDBScanTaskLocalState() override;

  //===----------Fields----------===//
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

  TaskCompletionMessageQueue&
    message_queue; ///< Message queue to notify TaskCreator about completion of the scan task

  // Execution drivers
  unique_ptr<duckdb::LocalTableFunctionState> local_tf_state;
  duckdb::ExecutionContext& context;
  const duckdb::DuckDBPhysicalTableScan& op;
};

//===--------------------------------------------------===//
// DuckDBScanTask
//===--------------------------------------------------===//
class DuckDBScanTask : public ITask
{
public:
  //===----------Constructor/Destructor----------===//
  DuckDBScanTask(uint64_t task_id,
                 unique_ptr<DuckDBScanTaskLocalState> local_state,
                 shared_ptr<DuckDBScanTaskGlobalState> global_state)
      : task_id(task_id)
      , ITask(std::move(local_state), std::move(global_state))
  {}
  ~DuckDBScanTask() override = default;

  //===----------Methods----------===//
  void Execute() override;

  //===----------Fields----------===//
  uint64_t task_id;
};

} // namespace sirius::parallel

// #pragma once
// #include "config.hpp"
// #include "operator/gpu_physical_table_scan.hpp"
// #include "duckdb/planner/filter/constant_filter.hpp"
// #include "duckdb/planner/filter/conjunction_filter.hpp"
// #include "duckdb/execution/execution_context.hpp"
// #include "gpu_pipeline_hashmap.hpp"
// #include "task_completion.hpp"
// #include "parallel/task_executor.hpp"
// #include "data/data_repository.hpp"
// #include "helper/helper.hpp"

// // The implementations have been based on the initial implementation in
// gpu_phyisical_table_scan.cpp by Yifei namespace sirius { namespace parallel {

// class DuckDBScanGlobalSourceState : public ITaskGlobalState {
// public:
//     explicit DuckDBScanGlobalSourceState(DataRepository& data_repository,
//     TaskCompletionMessageQueue& message_queue) :
//         data_repository_(data_repository), message_queue_(message_queue) {}
//     DataRepository& data_repository_;
//     TaskCompletionMessageQueue& message_queue_; // Message queue to notify TaskCreator about
//     completion of the
// };

// class DuckDBScanTask : public ITask {
// public:
//     /**
//      * @brief Construct a new DuckDBScanTask object
//      *
//      * @param scan_metadata Metadata required for the scan operation
//      * @param task_id A unique identifier for the task
//      * @param pipeline_id The id of the pipeline associated with this task
//      * @param local_state Local state specific to this task
//      * @param global_state Global state shared across scan tasks
//      */
//     DuckDBScanTask(uint64_t task_id,
//                     uint64_t pipeline_id,
//                     sirius::shared_ptr<DuckDBScanMetadata> scan_metadata,
//                     sirius::unique_ptr<ITaskLocalState> local_state,
//                     sirius::shared_ptr<ITaskGlobalState> global_state)
//         : ITask(std::move(local_state), std::move(global_state)),
//           scan_metadata_(scan_metadata),
//           task_id_(task_id),
//           pipeline_id_(pipeline_id) {}

//     /**
//      * @brief Method to actually execute the downgrade task
//      */
// 	void Execute() override;

//     /**
//      * @brief Method to convert the scanned data into a DataBatch
//      *
//      * This method transforms the raw data obtained from the DuckDB scan into a DataBatch
//      * that can be pushed to the Data Repository for further processing.
//      */
//     void ConvertToDataBatch();

//     /**
//      * @brief Method to mark that this task is completed
//      *
//      * This method informs that TaskCreator that the task is completed so that it can start
//      scheduling
//      * tasks that were dependent on this task. This method should be called after pushing the
//      output
//      * of this task to the Data Repository.
//      */
//     void MarkTaskCompletion();

//     /**
//      * @brief Method to push the resulting DataBatch to the Data Repository
//      *
//      * @param data_batch The data batch to push
//      * @param pipeline_id The identifier of the pipeline that produced this data batch
//      */
//     void PushToDataRepository(sirius::unique_ptr<sirius::DataBatch> data_batch, size_t
//     pipeline_id);

// private:
//   uint64_t task_id_;
//   uint64_t pipeline_id_;
//   sirius::shared_ptr<DuckDBScanMetadata> scan_metadata_;
//   ITaskGlobalState* g_state;
//   ITaskLocalState* l_state;
// };

// class DuckDBScanTaskQueue : public ITaskQueue {
// public:
//     /**
//      * @brief Construct a new DuckDBScanTaskQueue object
//      */
//     DuckDBScanTaskQueue() = default;

//     /**
//      * @brief Setups the task queue to start accepting and returning tasks
//      */
//     void Open() override {
//         sirius::lock_guard<sirius::mutex> lock(mutex_);
//         is_open_ = true;
//     }

//     /**
//      * @brief Closes the task queue from accepting new tasks or returning tasks
//      */
//     void Close() override {
//         sem_.release(); // signal that one item is available
//         sirius::lock_guard<sirius::mutex> lock(mutex_);
//         is_open_ = false;
//     }

//     /**
//      * @brief Push a new task to be scheduled.
//      *
//      * @param task The task to be scheduled
//      * @throws std::runtime_error If the scheduler is not currently accepting requests
//      */
//     void Push(sirius::unique_ptr<ITask> task) override {
//         // Convert ITask to DuckDBScanTask - since we know it's a DuckDBScanTask
//         auto duckdb_scan_task =
//         sirius::unique_ptr<DuckDBScanTask>(static_cast<DuckDBScanTask*>(task.release()));
//         Push(std::move(duckdb_scan_task));
//     }

//     /**
//      * @brief DuckDB scan specific push overload for type safety and convenience
//      *
//      * @param duckdb_scan_task The DuckDB scan task to be scheduled
//      * @throws std::runtime_error If the scheduler is not currently accepting requests
//      */
//     void Push(sirius::unique_ptr<DuckDBScanTask> duckdb_scan_task) {
//         EnqueueTask(std::move(duckdb_scan_task));
//     }

//     /**
//      * @brief Pull a task to execute.
//      *
//      * Note that this is a non blocking call and will return nullptr if no task is available. In
//      the future we should
//      * consider this call blocking.
//      *
//      * @return A unique pointer to the task to execute if there is one, nullptr otherwise
//      * @throws std::runtime_error If the scheduler is not currently stopped and thus not
//      returning tasks
//      */
//     sirius::unique_ptr<ITask> Pull() override {
//         // Delegate to DuckDB scan specific version and return as base type
//         auto duckdb_scan_task = PullScanTask();
//         return std::move(duckdb_scan_task);
//     }

//     /**
//      * @brief DuckDB scan specific pull method for type safety and convenience
//      *
//      * @return A unique pointer to the DuckDB scan task to execute, nullptr otherwise
//      * @throws std::runtime_error If the scheduler is not currently stopped and thus not
//      returning tasks
//      */
//     sirius::unique_ptr<DuckDBScanTask> PullScanTask() {
//         return DequeueTask();
//     }

//     /**
//      * @brief Enqueue a DuckDB scan task into the queue
//      *
//      * @param duckdb_scan_task The DuckDB scan task to enqueue
//      */
//     void EnqueueTask(sirius::unique_ptr<DuckDBScanTask> duckdb_scan_task) {
//         sirius::lock_guard<sirius::mutex> lock(mutex_);
//         if (duckdb_scan_task && is_open_) {
//             task_queue_.push(std::move(duckdb_scan_task));
//         }
//         sem_.release(); // signal that one item is available
//     }

//     /**
//      * @brief Dequeue a DuckDB scan task from the queue
//      *
//      * @return A unique pointer to the dequeued DuckDB scan task if there is a task, nullptr
//      otherwise
//      */
//     sirius::unique_ptr<DuckDBScanTask> DequeueTask() {
//         sem_.acquire(); // wait until there's something
//         sirius::lock_guard<sirius::mutex> lock(mutex_);
//         if (task_queue_.empty()) {
//             return nullptr;
//         }
//         auto task = std::move(task_queue_.front());
//         task_queue_.pop();
//         return task;
//     }

//     /**
//      * @brief Check if the task queue is empty
//      *
//      * @return true if the queue is empty, false otherwise
//      */
//     bool IsEmpty() const {
//         sirius::lock_guard<sirius::mutex> lock(mutex_);
//         return task_queue_.empty();
//     }

// private:
//     sirius::queue<sirius::unique_ptr<DuckDBScanTask>> task_queue_; // The underlying queue
//     storing the tasks bool is_open_ = false; // Whether the queue is open for accepting and
//     returning tasks mutable sirius::mutex mutex_;  // mutable to allow locking in const methods
//     std::counting_semaphore<> sem_{0}; // starts with 0 available permits
// };

// } // namespace parallel
// } // namespace sirius