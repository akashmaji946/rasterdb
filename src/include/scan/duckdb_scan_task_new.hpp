#pragma once

// sirius
#include <gpu_buffer_manager.hpp>
#include <parallel/task.hpp>
#include <parallel/task_queue.hpp>

// duckdb
#include <blockingconcurrentqueue.h>
#include <duckdb/common/types.hpp>
#include <duckdb/execution/operator/scan/physical_table_scan.hpp>
#include <duckdb/function/table_function.hpp>
#include <duckdb/parallel/concurrentqueue.hpp>

// standard library
#include <algorithm>
#include <atomic>
#include <climits>
#include <string>
#include <utility>
#include <vector>

namespace sirius::parallel
{

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define CEIL_DIV_8(x) (((x) + 7) >> 3)
#define MOD_8(x) ((x) & (7))
#define MUL_8(x) ((x) << 3)
#define DIV_8(x) ((x) >> 3)
#define MASK(x) ((1U << (x)) - 1U)

using idx_t = duckdb::idx_t;

//--------------------------------------------------//
// A concurrent queue for DuckDB scan tasks
//--------------------------------------------------//
class DuckDBScanTaskQueue : public ITaskQueue
{
public:
  DuckDBScanTaskQueue()           = default;
  ~DuckDBScanTaskQueue() override = default;

  void Open() override
  {
    is_open_.store(true, std::memory_order::release);
  }

  void Close() override
  {
    is_open_.store(false, std::memory_order::release);
  }

  // Use base-type tasks directly; just gate on is_open_
  void Push(sirius::unique_ptr<ITask> task) override
  {
    if (!is_open_.load(std::memory_order::acquire))
    {
      return; // ignore pushes when closed
    }
    task_queue_.enqueue(std::move(task));
  }

  // Wait until a task is available or the queue is closed.
  unique_ptr<ITask> Pull() override
  {
    unique_ptr<ITask> scan_task;
    while (true)
    {
      // Spin (for now -- will produce contention on is_open)
      if (task_queue_.try_dequeue(scan_task))
      {
        return scan_task;
      }
      // If closed, return
      if (!is_open_.load(std::memory_order::acquire))
      {
        return nullptr;
      }
    }
  }

private:
  std::atomic<bool> is_open_{false};
  duckdb_moodycamel::BlockingConcurrentQueue<sirius::unique_ptr<ITask>> task_queue_;
};

//--------------------------------------------------//
// DuckDB scan task global state
//--------------------------------------------------//
class DuckDBScanTaskGlobalState
    : public ITaskGlobalState
    , public duckdb::GlobalSourceState
{
public:
  //----------Constructor----------//
  explicit DuckDBScanTaskGlobalState(duckdb::ClientContext& context,
                                     const duckdb::PhysicalTableScan& op)
  {
    if (op.dynamic_filters && op.dynamic_filters->HasFilters())
    {
      table_filters = op.dynamic_filters->GetFinalTableFilters(op, op.table_filters.get());
    }
    if (op.function.init_global)
    {
      auto filters = table_filters ? *table_filters : GetTableFilters(op);
      duckdb::TableFunctionInitInput input(op.bind_data.get(),
                                           op.column_ids,
                                           op.projection_ids,
                                           filters,
                                           op.extra_info.sample_options);
      global_tf_state = op.function.init_global(context, input);
      max_threads     = global_tf_state ? global_tf_state->MaxThreads() : 1;
    }
    else
    {
      max_threads = 1;
    }

    // in_out_functions are not currently supported
    if (op.function.in_out_function)
    {
      std::string exception_msg = "[DuckDBScanTaskGlobalState] In-out functions not supported]";
      throw duckdb::NotImplementedException(exception_msg);
    }
  }

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
  unique_ptr<duckdb::GlobalTableFunctionState> global_tf_state;
  unique_ptr<duckdb::TableFilterSet> table_filters;
};

//--------------------------------------------------//
// DuckDB scan task local state
//--------------------------------------------------//
class DuckDBScanTaskLocalState
    : public ITaskLocalState
    , public duckdb::LocalSourceState
{
  static constexpr size_t DEFAULT_TARGET_BYTES = 2ULL << 30; // ~2GB
public:
  //----------Constructor----------//
  explicit DuckDBScanTaskLocalState(shared_ptr<DuckDBScanTaskQueue> task_queue,
                                    duckdb::ExecutionContext& context,
                                    DuckDBScanTaskGlobalState& gstate,
                                    const duckdb::PhysicalTableScan& op)
      : task_queue(std::move(task_queue))
      , context(context)
      , op(op)
      , num_columns(op.column_ids.size()) /// KEVIN: row_id column?
  {
    if (op.function.init_local)
    {
      duckdb::TableFunctionInitInput input(op.bind_data.get(),
                                           op.column_ids,
                                           op.projection_ids,
                                           gstate.GetTableFilters(op),
                                           op.extra_info.sample_options);
      local_tf_state = op.function.init_local(context, input, gstate.global_tf_state.get());
    }

    column_sizes.resize(num_columns);
    scanned_types.resize(num_columns);
    byte_offsets.resize(num_columns);
    for (duckdb::idx_t i = 0; i < num_columns; ++i)
    {
      scanned_types[i] = op.types[op.column_ids[i].GetPrimaryIndex()];
      column_sizes[i]  = duckdb::GetTypeIdSize(scanned_types[i].InternalType());
      byte_offsets[i]  = 0;
      max_type_size    = std::max(max_type_size, column_sizes[i]);
    }

    /// TODO: make memory reservations

    // Allocate buffers for Execute() (for now, just 2GB per column)
    data_ptrs   = gpu_buffer_manager->customCudaHostAlloc<uint8_t*>(num_columns);
    mask_ptrs   = gpu_buffer_manager->customCudaHostAlloc<uint8_t*>(num_columns);
    offset_ptrs = gpu_buffer_manager->customCudaHostAlloc<uint64_t*>(num_columns);
    for (duckdb::idx_t i = 0; i < num_columns; ++i)
    {
      // Data buffer
      data_ptrs[i] = gpu_buffer_manager->customCudaHostAlloc<uint8_t>(DEFAULT_TARGET_BYTES);

      // Validity mask buffer (round up to nearest byte)
      mask_ptrs[i] = gpu_buffer_manager->customCudaHostAlloc<uint8_t>(DIV_8(DEFAULT_TARGET_BYTES));

      // VARCHAR offset buffers
      if (scanned_types[i] == duckdb::LogicalTypeId::VARCHAR)
      {
        offset_ptrs[i] =
          gpu_buffer_manager->customCudaHostAlloc<uint64_t>(DIV_8(DEFAULT_TARGET_BYTES) + 1);
        offset_ptrs[i][0] = 0; // Initialize for running exclusive sum
      }
      else
      {
        offset_ptrs[i] = nullptr;
      }
    }
  }

  //----------Fields----------//
  size_t bytes_accumulated = 0;     ///< Total bytes accumulated so far by the scan
  std::vector<size_t> column_sizes; ///< Size of each DuckDB column in bytes
  size_t max_type_size = 0;         ///< Maximum size of any single type in bytes

  // Buffers for converting duckdb chunk to DataBatch
  duckdb::GPUBufferManager* gpu_buffer_manager = &duckdb::GPUBufferManager::GetInstance();
  uint8_t** data_ptrs                          = nullptr;
  uint8_t** mask_ptrs                          = nullptr;
  uint64_t** offset_ptrs                       = nullptr;
  std::vector<size_t> byte_offsets; ///< Current byte offsets in data buffers
  size_t row_offset = 0;            ///< Current row offset in buffers

  // Task queue for pushing extra scan tasks back into the queue
  shared_ptr<DuckDBScanTaskQueue> task_queue;

  size_t num_columns;
  std::vector<duckdb::LogicalType> scanned_types; ///< Types of the scanned columns
  duckdb::DataChunk chunk;                        ///< DataChunk buffer
  unique_ptr<duckdb::LocalTableFunctionState> local_tf_state;
  duckdb::ExecutionContext& context;   ///< For driving Execute()
  const duckdb::PhysicalTableScan& op; ///< For driving Execute()
};

//--------------------------------------------------//
// A DuckDB scan task
//--------------------------------------------------//
class DuckDBScanTask : public ITask
{
public:
  static constexpr size_t DEFAULT_TARGET_BYTES = 2ULL << 30; // 2GB

  //----------Constructor----------//
  DuckDBScanTask(unique_ptr<DuckDBScanTaskLocalState> local_state,
                 shared_ptr<DuckDBScanTaskGlobalState> global_state)
      : ITask(std::move(local_state), std::move(global_state))
  {}

  // The work function of the task
  void Execute() override
  {
    // Cast base task states to our concrete states
    auto& g = static_cast<DuckDBScanTaskGlobalState&>(*this->global_state_);
    auto& l = static_cast<DuckDBScanTaskLocalState&>(*this->local_state_);
    l.chunk.Initialize(duckdb::Allocator::Get(l.context.client), l.op.types);

    // Drive the table function directly until ~2GB (approximate) or source exhaustion
    bool is_finished = g.IsSourceDrained();
    while (!is_finished)
    {
      // Get a DataChunk
      duckdb::TableFunctionInput input(l.op.bind_data.get(),
                                       l.local_tf_state.get(),
                                       g.global_tf_state.get());
      l.op.function.function(l.context.client, input, l.chunk);
      if (l.chunk.size() == 0)
      {
        // Source is exhausted
        g.SetSourceDrained();
        is_finished = true;
        break;
      }

      size_t chunk_bytes = 0;
      for (duckdb::idx_t col = 0; col < l.chunk.ColumnCount(); ++col)
      {
        auto& vec = l.chunk.data[col];
        vec.Flatten(l.chunk.size());
        auto& validity = duckdb::FlatVector::Validity(vec);

        // DATA
        if (vec.GetType().id() == duckdb::LogicalTypeId::VARCHAR)
        {
          //----------VARCHAR----------//
          auto str_data = reinterpret_cast<duckdb::string_t*>(vec.GetData());
          for (idx_t row = 0; row < l.chunk.size(); ++row)
          {
            if (validity.RowIsValid(row))
            {
              auto const len = str_data[row].GetSize();
              chunk_bytes += len; // Data
              l.offset_ptrs[col][l.row_offset + row + 1] =
                l.offset_ptrs[col][l.row_offset + row] + len; // Prefix sum of offsets

              // Ensure that we have space in the data buffer for the data copy
              if (l.byte_offsets[col] + len > DEFAULT_TARGET_BYTES)
              {
                // For now, just throw an exception
                std::string error_msg = "[DuckDBScanTask] Insufficient space in data buffer for "
                                        "VARCHAR data copy";
                throw duckdb::InternalException(error_msg);
              }

              // Do the memcpy
              memcpy(l.data_ptrs[col] + l.byte_offsets[col], str_data[row].GetData(), len);
              l.byte_offsets[col] += len;
            }
          }
          chunk_bytes += sizeof(uint64_t) * l.chunk.size(); // Offsets
        }
        else
        {
          //----------NON-VARCHAR----------//
          memcpy(l.data_ptrs[col] + l.byte_offsets[col],
                 vec.GetData(),
                 l.column_sizes[col] * l.chunk.size());
          l.byte_offsets[col] += l.column_sizes[col] * l.chunk.size();
          chunk_bytes += l.column_sizes[col] * l.chunk.size();
        }

        // VALIDITY MASK
        if (validity.GetData())
        {
          auto const* src_valid = reinterpret_cast<const uint8_t*>(validity.GetData());
          auto* dst             = l.mask_ptrs[col];
          auto const cur_bit    = MOD_8(l.row_offset); ///< bit offset in tail byte
          auto const dst_byte   = DIV_8(l.row_offset); ///< dst start byte
          auto const nbits      = l.chunk.size();

          if (cur_bit == 0)
          {
            //----------Byte-Aligned----------//
            auto const full_bytes = DIV_8(nbits);
            auto const tail_bits  = MOD_8(nbits);
            memcpy(dst + dst_byte, src_valid, full_bytes);
            if (tail_bits > 0)
            {
              auto const tail_mask       = static_cast<uint8_t>(MASK(tail_bits));
              auto const tail            = src_valid[full_bytes] & tail_mask;
              dst[dst_byte + full_bytes] = (dst[dst_byte + full_bytes] & ~tail_mask) | tail;
            }
          }
          else
          {
            //----------Byte-Unaligned----------//
            /// KEVIN: is this code path ever entered in the single-threaded case?
            /// KEVIN: [potential optimization] use uint64_t chunks for bit manipulation
            auto const nbytes = CEIL_DIV_8(nbits);
            for (idx_t b = 0; b < nbytes; ++b)
            {
              auto src_byte          = src_valid[b];
              auto const cur_bits    = std::min<uint32_t>(CHAR_BIT, nbits - MUL_8(b));
              src_byte               = src_byte & static_cast<uint8_t>(MASK(cur_bits));
              auto const nlower_bits = std::min<uint32_t>(CHAR_BIT - cur_bit, cur_bits);
              auto const lower_mask  = static_cast<uint8_t>(MASK(nlower_bits) << cur_bit);
              auto const lower_bits =
                static_cast<uint8_t>((src_byte & MASK(nlower_bits)) << cur_bit);
              dst[dst_byte + b] = (dst[dst_byte + b] & ~lower_mask) | lower_bits;

              // There may be leftover bits to propagate to next byte
              auto const nupper_bits = cur_bits - nlower_bits;
              if (nupper_bits > 0)
              {
                auto const upper_mask = static_cast<uint8_t>(MASK(nupper_bits));
                auto const upper_bits =
                  static_cast<uint8_t>((src_byte >> nlower_bits) & upper_mask);
                dst[dst_byte + b + 1] = (dst[dst_byte + b + 1] & ~upper_mask) | upper_bits;
              }
            }
          }
        }
      }
      l.row_offset += l.chunk.size();
      l.bytes_accumulated += chunk_bytes;

      // Check termination condition (need to add null mask byte count)
      if (l.bytes_accumulated + CEIL_DIV_8(l.row_offset) + l.max_type_size * STANDARD_VECTOR_SIZE >=
          DEFAULT_TARGET_BYTES)
      {
        is_finished = true;
      }
      else
      {
        l.chunk.Reset();
      }
    } ///< End while loop

    // Add tasks back to the queue if the scan is not finished
    if (!g.IsSourceDrained())
    {
      auto new_task = sirius::make_unique<DuckDBScanTask>(
        sirius::make_unique<DuckDBScanTaskLocalState>(l.task_queue, l.context, g, l.op),
        this->global_state_);
      l.task_queue->Push(std::move(new_task));
    }

    // Copy data to perfectly sized buffers
    /// TODO: make memory reservations
    uint8_t** data_ptrs   = gpu_buffer_manager->customCudaHostAlloc<uint8_t*>(num_columns);
    uint8_t** mask_ptrs   = gpu_buffer_manager->customCudaHostAlloc<uint8_t*>(num_columns);
    uint64_t** offset_ptrs = gpu_buffer_manager->customCudaHostAlloc<uint64_t*>(num_columns);
    for(idx_t col = 0; col < num_columns; ++col)
    {
      // Data buffer
      data_ptrs[col] = gpu_buffer_manager->customCudaHostAlloc<uint8_t>(l.byte_offsets[col]);
      memcpy(data_ptrs[col], l.data_ptrs[col], l.byte_offsets[col]);

      // Validity mask buffer (round up to nearest byte)
      auto const mask_bytes = CEIL_DIV_8(l.row_offset);
      mask_ptrs[col]       = gpu_buffer_manager->customCudaHostAlloc<uint8_t>(mask_bytes);
      memcpy(mask_ptrs[col], l.mask_ptrs[col], mask_bytes);

      // VARCHAR offset buffers
      if (l.scanned_types[col] == duckdb::LogicalTypeId::VARCHAR)
      {
        auto const offset_bytes = (l.row_offset + 1) * sizeof(uint64_t);
        offset_ptrs[col]       = gpu_buffer_manager->customCudaHostAlloc<uint64_t>(l.row_offset + 1);
        memcpy(offset_ptrs[col], l.offset_ptrs[col], offset_bytes);
      }
      else
      {
        offset_ptrs[col] = nullptr;
      }
    }

    /// TODO: Push data as a data batch to the data repository
  }

private:
  duckdb::GPUBufferManager* gpu_buffer_manager = &duckdb::GPUBufferManager::GetInstance();
};

} // namespace sirius::parallel