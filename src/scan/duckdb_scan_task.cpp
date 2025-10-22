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

// sirius
#include <scan/duckdb_scan_task.hpp>
#include <scan/duckdb_scan_task_executor.hpp>
#include <task_completion.hpp>

// duckdb
// #include <duckdb/allocator/allocator.hpp> /// TODO

// standard library
#include <algorithm>
#include <iostream> /// DEBUG
#include <string>

namespace sirius::parallel
{

// Byte arithmetic and bit manipulation macros
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define CEIL_DIV_8(x) (((x) + 7) >> 3)
#define MOD_8(x) ((x) & (7))
#define MUL_8(x) ((x) << 3)
#define DIV_8(x) ((x) >> 3)
#define MASK(x) ((1U << (x)) - 1U)

//===--------------------------------------------------===//
// DuckDBScanTaskGlobalState
//===--------------------------------------------------===//
DuckDBScanTaskGlobalState::DuckDBScanTaskGlobalState(size_t pipeline_id,
                                                     duckdb::ClientContext& context,
                                                     const duckdb::DuckDBPhysicalTableScan& pts)
    : pipeline_id(pipeline_id)
{
  const auto& op = pts.physical_table_scan;
  if (op.dynamic_filters && op.dynamic_filters->HasFilters())
  {
    table_filters = op.dynamic_filters->GetFinalTableFilters(op, op.table_filters.get());
  }
  if (op.function.init_global)
  {
    auto filters = table_filters ? *table_filters : GetTableFilters(pts);
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

//===--------------------------------------------------===//
// DuckDBScanTaskLocalState
//===--------------------------------------------------===//
DuckDBScanTaskLocalState::DuckDBScanTaskLocalState(TaskCompletionMessageQueue& message_queue,
                                                   const DuckDBScanTaskGlobalState& gstate,
                                                   duckdb::ExecutionContext& context,
                                                   const duckdb::DuckDBPhysicalTableScan& op)
    : message_queue(message_queue)
    , context(context)
    , op(op)
{
  auto const& pts = op.physical_table_scan;
  num_columns     = pts.column_ids.size();
  if (pts.function.init_local)
  {
    duckdb::TableFunctionInitInput input(pts.bind_data.get(),
                                         pts.column_ids,
                                         pts.projection_ids,
                                         gstate.GetTableFilters(op),
                                         pts.extra_info.sample_options);
    local_tf_state = pts.function.init_local(context, input, gstate.global_tf_state.get());
  }

  // Allocate vectors
  scanned_types.resize(num_columns);
  column_sizes.resize(num_columns);
  data_ptrs.resize(num_columns);
  mask_ptrs.resize(num_columns);
  offset_ptrs.resize(num_columns);
  byte_offsets.resize(num_columns);

  // Initialize scanned types and column sizes
  for (duckdb::idx_t i = 0; i < num_columns; ++i)
  {
    scanned_types[i] = pts.types[pts.column_ids[i].GetPrimaryIndex()];
    if (scanned_types[i] == duckdb::LogicalTypeId::VARCHAR)
    {
      column_sizes[i] = DEFAULT_VARCHAR_BYTES;
    }
    else
    {
      column_sizes[i] = duckdb::GetTypeIdSize(scanned_types[i].InternalType());
    }
    byte_offsets[i] = 0;
    max_type_size   = std::max(max_type_size, column_sizes[i]);
  }

  // Allocate Execute() buffers
  /// TODO: make memory reservations
  for (duckdb::idx_t i = 0; i < num_columns; ++i)
  {
    // Data buffer
    data_ptrs[i] = new uint8_t[DEFAULT_TARGET_BYTES];

    // Validity mask buffer (conservatively assume 1B / row)
    mask_ptrs[i] = new uint8_t[CEIL_DIV_8(CEIL_DIV_8(DEFAULT_TARGET_BYTES))];

    // VARCHAR offset buffers
    if (scanned_types[i] == duckdb::LogicalTypeId::VARCHAR)
    {
      offset_ptrs[i]    = new uint64_t[CEIL_DIV_8(DEFAULT_TARGET_BYTES) + 1];
      offset_ptrs[i][0] = 0; // Initialize for running exclusive sum
    }
    else
    {
      offset_ptrs[i] = nullptr;
    }
  }
}

DuckDBScanTaskLocalState::~DuckDBScanTaskLocalState()
{
  for (idx_t i = 0; i < num_columns; ++i)
  {
    // Free data buffer
    delete[] data_ptrs[i];

    // Free validity mask buffer
    delete[] mask_ptrs[i];

    // Free VARCHAR offset buffers
    delete[] offset_ptrs[i];
  }
}

//===--------------------------------------------------===//
// DuckDBScanTask
//===--------------------------------------------------===//
void DuckDBScanTask::Execute()
{
  std::cout << "[DuckDBScanTask] Executing scan task...\n";

  // Cast base task states to our concrete states
  auto& g = this->global_state_->Cast<DuckDBScanTaskGlobalState>();
  auto& l = this->local_state_->Cast<DuckDBScanTaskLocalState>();
  l.chunk.Initialize(duckdb::Allocator::Get(l.context.client), l.op.physical_table_scan.types);

  // Drive the table function directly until ~2GB (approximate) or source exhaustion
  bool is_finished = g.IsSourceDrained();
  while (!is_finished)
  {
    std::cout << "[DuckDBScanTask] Loop start: pulling next data chunk...\n";
    // Get a DataChunk
    duckdb::TableFunctionInput input(l.op.physical_table_scan.bind_data.get(),
                                     l.local_tf_state.get(),
                                     g.global_tf_state.get());
    l.op.physical_table_scan.function.function(l.context.client, input, l.chunk);
    if (l.chunk.size() == 0)
    {
      // Source is exhausted
      g.SetSourceDrained();
      is_finished = true;
      break;
    }

    size_t chunk_bytes = 0;
    for (duckdb::idx_t col = 0; col < l.num_columns; ++col)
    {
      auto& vec = l.chunk.data[col];
      vec.Flatten(l.chunk.size());
      auto& validity = duckdb::FlatVector::Validity(vec);

      //===----------DATA----------===//
      if (vec.GetType().id() == duckdb::LogicalTypeId::VARCHAR)
      {
        //===----------VARCHAR----------===//
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
            if (l.byte_offsets[col] + len > DuckDBScanTaskLocalState::DEFAULT_TARGET_BYTES)
            {
              /// TODO: push current data batch and start a new one
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
        //===----------NON-VARCHAR----------===//
        memcpy(l.data_ptrs[col] + l.byte_offsets[col],
               vec.GetData(),
               l.column_sizes[col] * l.chunk.size());
        l.byte_offsets[col] += l.column_sizes[col] * l.chunk.size();
        chunk_bytes += l.column_sizes[col] * l.chunk.size();
      }

      //===----------VALIDITY MASK----------===//
      if (validity.GetData())
      {
        auto const* src_valid = reinterpret_cast<const uint8_t*>(validity.GetData());
        auto* dst             = l.mask_ptrs[col];
        auto const cur_bit    = MOD_8(l.row_offset); ///< bit offset in current tail byte
        auto const dst_byte   = DIV_8(l.row_offset); ///< dst start byte
        auto const nbits      = l.chunk.size();

        if (cur_bit == 0)
        {
          //===----------Byte-Aligned----------===//
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
          //===----------Byte-Unaligned----------===//
          /// KEVIN: [potential optimization] use uint64_t chunks for bit manipulation
          /// KEVIN: [potential optimization] use 2 buffers for aligned and unaligned writes (once
          /// unaligned is used once, most subsequent writes will be unaligned)
          auto const nbytes = CEIL_DIV_8(nbits);
          for (idx_t b = 0; b < nbytes; ++b)
          {
            auto src_byte          = src_valid[b];
            auto const cur_bits    = std::min<uint32_t>(CHAR_BIT, nbits - MUL_8(b));
            src_byte               = src_byte & static_cast<uint8_t>(MASK(cur_bits));
            auto const nlower_bits = std::min<uint32_t>(CHAR_BIT - cur_bit, cur_bits);
            auto const lower_mask  = static_cast<uint8_t>(MASK(nlower_bits) << cur_bit);
            auto const lower_bits = static_cast<uint8_t>((src_byte & MASK(nlower_bits)) << cur_bit);
            dst[dst_byte + b]     = (dst[dst_byte + b] & ~lower_mask) | lower_bits;

            // There may be leftover bits to propagate to next byte
            auto const nupper_bits = cur_bits - nlower_bits;
            if (nupper_bits > 0)
            {
              auto const upper_mask = static_cast<uint8_t>(MASK(nupper_bits));
              auto const upper_bits = static_cast<uint8_t>((src_byte >> nlower_bits) & upper_mask);
              dst[dst_byte + b + 1] = (dst[dst_byte + b + 1] & ~upper_mask) | upper_bits;
            }
          }
        }
      }
    }
    l.row_offset += l.chunk.size();
    l.bytes_accumulated += chunk_bytes;

    // Check termination condition (need to add null mask byte count)
    if (l.bytes_accumulated + l.num_columns * CEIL_DIV_8(l.row_offset) +
          l.max_type_size * STANDARD_VECTOR_SIZE >=
        DuckDBScanTaskLocalState::DEFAULT_TARGET_BYTES)
    {
      is_finished = true;
    }
    else
    {
      l.chunk.Reset();
    }
  } ///< End while loop

  auto& sirius_context = SiriusContext::GetInstance();

  // Add tasks back to the queue if the scan is not finished
  if (!g.IsSourceDrained())
  {
    auto& task_creator       = sirius_context.GetTaskCreator();
    auto& scan_task_executor = sirius_context.GetDuckDBScanTaskExecutor();
    auto new_task            = sirius::make_unique<DuckDBScanTask>(
      task_creator.GetNextTaskId(),
      sirius::make_unique<DuckDBScanTaskLocalState>(l.message_queue, g, l.context, l.op),
      std::dynamic_pointer_cast<DuckDBScanTaskGlobalState>(this->global_state_));
    scan_task_executor.Schedule(std::move(new_task));
  }

  // Copy data to correctly sized buffers
  vector<uint8_t*> new_data_ptrs(l.num_columns);
  vector<uint8_t*> new_mask_ptrs(l.num_columns);
  vector<uint64_t*> new_offset_ptrs(l.num_columns);
  for (idx_t col = 0; col < l.num_columns; ++col)
  {
    // Data buffer
    new_data_ptrs[col] = new uint8_t[l.byte_offsets[col]];
    memcpy(new_data_ptrs[col], l.data_ptrs[col], l.byte_offsets[col]);

    // Validity mask buffer (round up to nearest byte)
    auto const mask_bytes = CEIL_DIV_8(l.row_offset);
    new_mask_ptrs[col]    = new uint8_t[mask_bytes];
    memcpy(new_mask_ptrs[col], l.mask_ptrs[col], mask_bytes);

    // VARCHAR offset buffers
    if (l.scanned_types[col] == duckdb::LogicalTypeId::VARCHAR)
    {
      auto const offset_bytes = (l.row_offset + 1) * sizeof(uint64_t);
      new_offset_ptrs[col]    = new uint64_t[l.row_offset + 1];
      memcpy(new_offset_ptrs[col], l.offset_ptrs[col], offset_bytes);
    }
    else
    {
      new_offset_ptrs[col] = nullptr;
    }
  }

  // Push data as a data batch to the data repository (which will then own the buffers)
  /// TODO: create IDataRepresentation
  auto& data_repository = sirius_context.GetDataRepository();
  auto batch_id         = data_repository.GetNextDataBatchId();
  auto data_batch       = make_unique<DataBatch>(batch_id, nullptr);
  data_repository.AddNewDataBatch(g.pipeline_id, std::move(data_batch));

  // For now, free the buffers
  for (idx_t col = 0; col < l.num_columns; ++col)
  {
    delete[] new_data_ptrs[col];
    delete[] new_mask_ptrs[col];
    if (new_offset_ptrs[col])
    {
      delete[] new_offset_ptrs[col];
    }
  }

  // Notify TaskCreator of completion
  l.message_queue.EnqueueMessage(
    make_unique<TaskCompletionMessage>(task_id, g.pipeline_id, Source::SCAN));
}
} // namespace sirius::parallel

// #include "scan/duckdb_scan_executor.hpp"
// #include "scan/duckdb_scan_task.hpp"

// namespace sirius {
// namespace parallel {

// void DuckDBScanTask::Execute() {
//     // create IDataRepresentation
//     // create DataBatch
//     std::cout << "Scan Task is executing\n";
//     auto batch_id =
//     global_state_->Cast<DuckDBScanGlobalSourceState>().data_repository_.GetNextDataBatchId();
//     sirius::unique_ptr<sirius::DataBatch> data_batch =
//     sirius::make_unique<sirius::DataBatch>(batch_id, nullptr);
//     // push DataBatch to data_repository_
//     global_state_->Cast<DuckDBScanGlobalSourceState>().data_repository_.AddNewDataBatch(pipeline_id_,
//     std::move(data_batch));
//     // notify TaskCreator about task completion
//     MarkTaskCompletion();
// }

// void DuckDBScanTask::MarkTaskCompletion() {
//     // notify TaskCreator about task completion
//     std::cout << "Marking Scan Task Completion\n";
//     auto message = sirius::make_unique<sirius::TaskCompletionMessage>(task_id_, pipeline_id_,
//     sirius::Source::SCAN);
//     global_state_->Cast<DuckDBScanGlobalSourceState>().message_queue_.EnqueueMessage(std::move(message));
// }

// } // namespace parallel
// } // namespace sirius