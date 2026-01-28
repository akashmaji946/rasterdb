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
#include <data/data_batch_utils.hpp>
#include <helper/utils.hpp>
#include <memory/sirius_memory_reservation_manager.hpp>
#include <op/scan/duckdb_scan_task.hpp>

// cucascade
#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/memory/host_table.hpp>
#include <cucascade/memory/memory_reservation.hpp>

// duckdb
#include <duckdb/common/types.hpp>
#include <duckdb/function/table_function.hpp>

// standard library
#include <limits>
#include <numeric>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// duckdb_scan_task_global_state
//===----------------------------------------------------------------------===//
duckdb_scan_task_global_state::duckdb_scan_task_global_state(
  duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline,
  duckdb_scan_executor& scan_exec,
  duckdb::ClientContext& client_ctx,
  sirius_physical_table_scan* scan_op)
  : _pipeline(std::move(pipeline)),
    _sirius_ctx(client_ctx.registered_state->Get<duckdb::SiriusContext>("sirius_state")),
    _max_threads(scan_exec.get_num_threads()),
    _scan_executor(scan_exec),
    _op(*scan_op)
{
  // Initialize global table function state
  if (_op.function.init_global) {
    duckdb::TableFunctionInitInput tf_input(_op.bind_data.get(),
                                            _op.column_ids,
                                            _op.projection_ids,
                                            nullptr,
                                            _op.extra_info.sample_options);
    _global_tf_state = _op.function.init_global(client_ctx, tf_input);
  }

  // We do not support in_out_functions
  if (_op.function.in_out_function) {
    throw duckdb::NotImplementedException(
      "In-out table functions are not supported in sirius table scans.");
  }

  // For caching reasons, we do not push table filters into the scan
  if (_op.dynamic_filters) {
    throw duckdb::NotImplementedException(
      "Dynamic table filters are not supported in sirius table scans.");
  }
}

//===----------------------------------------------------------------------===//
// duckdb_scan_task_local_state::column_builder
//===----------------------------------------------------------------------===//
duckdb_scan_task_local_state::column_builder::column_builder(duckdb::LogicalType t,
                                                             size_t default_varchar_size)
  : type(t)
{
  type_size = t.InternalType() == duckdb::PhysicalType::VARCHAR
                ? default_varchar_size
                : duckdb::GetTypeIdSize(t.InternalType());
}

void duckdb_scan_task_local_state::column_builder::initialize_accessors(
  size_t estimated_num_rows,
  size_t byte_offset,
  std::unique_ptr<multiple_blocks_allocation>& allocation)
{
  assert(allocation != nullptr);
  assert(!allocation->get_blocks().empty());
  assert(utils::mod_8(byte_offset) == 0);  // byte_offset must be 8B-aligned

  if (type.InternalType() == duckdb::PhysicalType::VARCHAR) {
    // Initialize offset accessor
    offset_blocks_accessor.initialize(byte_offset, allocation);
    // Write the initial offset value of 0
    offset_blocks_accessor.set_current(0, allocation);
    // Initialize data accessor
    auto const data_byte_offset = byte_offset + (estimated_num_rows + 1) * sizeof(int64_t);
    data_blocks_accessor.initialize(data_byte_offset, allocation);
    // Initialize mask accessor
    total_data_bytes_allocated  = utils::align_8(estimated_num_rows * type_size);
    auto const mask_byte_offset = data_byte_offset + total_data_bytes_allocated;
    mask_blocks_accessor.initialize(mask_byte_offset, allocation);
  } else {
    // Fixed-width column
    data_blocks_accessor.initialize(byte_offset, allocation);
    auto const mask_byte_offset = utils::align_8(byte_offset + estimated_num_rows * type_size);
    mask_blocks_accessor.initialize(mask_byte_offset, allocation);
  }
}

bool duckdb_scan_task_local_state::column_builder::sufficient_space_for_column(
  duckdb::Vector& vec, duckdb::ValidityMask const& validity, size_t num_rows)
{
  // This method should be called only on variable-length data
  assert(type.InternalType() == duckdb::PhysicalType::VARCHAR);

  size_t data_bytes    = 0;
  auto const* str_data = reinterpret_cast<duckdb::string_t const*>(vec.GetData());
  for (size_t row = 0; row < num_rows; ++row) {
    if (validity.RowIsValid(row)) { data_bytes += str_data[row].GetSize(); }
  }
  return data_bytes + total_data_bytes <= total_data_bytes_allocated;
}

void duckdb_scan_task_local_state::column_builder::process_mask_for_column(
  duckdb::ValidityMask const& validity,
  size_t num_rows,
  size_t row_offset,
  std::unique_ptr<multiple_blocks_allocation>& allocation)
{
  uint64_t constexpr bit_width = std::numeric_limits<uint64_t>::digits;

  auto const* src_valid = validity.GetData();
  auto const cur_bit    = utils::mod<bit_width>(row_offset);
  auto const num_bits   = num_rows;

  if (cur_bit == 0) {
    //===----------Byte Aligned Case----------===//
    auto const full_words = utils::div<bit_width>(num_rows);
    auto const tail_bits  = utils::mod<bit_width>(num_rows);
    if (src_valid == nullptr) {
      //===----------All Valid----------===//
      // Set all bits in the mask to valid
      if (full_words != 0) {
        mask_blocks_accessor.memset(FULL_MASK, full_words * sizeof(uint64_t), allocation);
      }
      if (tail_bits != 0) {
        auto const tail_mask = utils::make_mask<uint64_t>(tail_bits);
        mask_blocks_accessor.set_current(tail_mask, allocation);
      }
      return;
    }
    //===----------Some Invalid----------===//
    size_t valid_count = 0;
    if (full_words != 0) {
      valid_count =
        mask_blocks_accessor.memcpy_from_with_popcount(src_valid, full_words, allocation);
    }
    if (tail_bits != 0) {
      auto const tail_mask = utils::make_mask<uint64_t>(tail_bits);
      auto const tail      = src_valid[full_words] & tail_mask;
      mask_blocks_accessor.set_current(tail, allocation);
      valid_count += std::popcount(tail);
    }
    null_count += num_rows - valid_count;
    return;
  }

  //===----------Byte Unaligned Case----------===//
  if (src_valid == nullptr) {
    //===----------All Valid----------===//
    auto const bits_in_current_word = std::min(bit_width - cur_bit, num_bits);
    auto const remaining_bits =
      num_bits - bits_in_current_word;  // Remaining bits after filling current word
    auto const full_words = utils::div<bit_width>(remaining_bits);
    auto const tail_bits  = utils::mod<bit_width>(remaining_bits);

    // Set bits in the current byte
    auto const current_word_mask =
      static_cast<uint64_t>(utils::make_mask<uint64_t>(bits_in_current_word) << cur_bit);
    auto const current_word = mask_blocks_accessor.get_current(allocation);
    mask_blocks_accessor.set_current(current_word | current_word_mask, allocation);
    if (bits_in_current_word + cur_bit == bit_width) { mask_blocks_accessor.advance(); }
    if (full_words != 0) {
      // Set full words to all valid
      mask_blocks_accessor.memset(FULL_MASK, full_words * sizeof(uint64_t), allocation);
    }
    if (tail_bits != 0) {
      // Set tail bits
      auto const tail_mask = utils::make_mask<uint64_t>(tail_bits);
      mask_blocks_accessor.set_current(tail_mask, allocation);
    }
    return;
  }
  //===----------Some Invalid----------===//
  // THIS CODE PATH IS PERFORMANCE HOTSPOT
  auto const full_words = utils::div<bit_width>(num_bits);
  auto const tail_bits  = utils::mod<bit_width>(num_bits);
  auto const cur_shift  = static_cast<uint64_t>(cur_bit);
  auto const upper_mask = utils::make_mask<uint64_t>(cur_shift);
  auto const next_shift = static_cast<uint64_t>(bit_width - cur_bit);
  auto const lower_mask = utils::make_mask<uint64_t>(next_shift);

  auto* block_words =
    reinterpret_cast<uint64_t*>(allocation->get_blocks()[mask_blocks_accessor.block_index] +
                                mask_blocks_accessor.offset_in_block);
  uint64_t upper_bits    = block_words[0] & upper_mask;
  size_t words_processed = 0;
  while (words_processed < full_words) {
    auto const words_to_process = std::min(
      full_words - words_processed,
      (allocation->block_size() - mask_blocks_accessor.offset_in_block) / sizeof(uint64_t));
    for (size_t w = 0; w < words_to_process; ++w) {
      auto const src_word = src_valid[words_processed + w];
      null_count += bit_width - std::popcount(src_word);
      auto const lower_bits  = static_cast<uint64_t>((src_word & lower_mask) << cur_shift);
      auto const word_to_set = upper_bits | lower_bits;
      block_words[w]         = word_to_set;
      upper_bits             = static_cast<uint64_t>((src_word >> next_shift) & upper_mask);
    }
    words_processed += words_to_process;
    mask_blocks_accessor.offset_in_block += words_to_process * sizeof(uint64_t);
    if (mask_blocks_accessor.offset_in_block == allocation->block_size()) {
      // Advance to next block
      mask_blocks_accessor.block_index++;
      mask_blocks_accessor.offset_in_block = 0;
      block_words =
        reinterpret_cast<uint64_t*>(allocation->get_blocks()[mask_blocks_accessor.block_index]);
    }
  }
  if (tail_bits != 0) {
    auto const src_block = src_valid[full_words] & utils::make_mask<uint64_t>(tail_bits);
    null_count += tail_bits - std::popcount(src_block);
    auto const lower_bits  = static_cast<uint64_t>((src_block & lower_mask) << cur_shift);
    auto const word_to_set = upper_bits | lower_bits;
    mask_blocks_accessor.set_current(word_to_set, allocation);
    if (tail_bits >= next_shift) {
      upper_bits = static_cast<uint64_t>((src_block >> next_shift) & upper_mask);
      mask_blocks_accessor.advance();
      mask_blocks_accessor.set_current(upper_bits, allocation);
    }
  }
}

void duckdb_scan_task_local_state::column_builder::process_column(
  duckdb::Vector& vec,
  duckdb::ValidityMask const& validity,
  size_t num_rows,
  size_t row_offset,
  std::unique_ptr<multiple_blocks_allocation>& allocation)
{
  // PRECONDITION: Vector must be flattened
  if (type.InternalType() == duckdb::PhysicalType::VARCHAR) {
    size_t data_bytes    = 0;
    auto const* str_data = reinterpret_cast<duckdb::string_t const*>(vec.GetData());
    for (size_t row = 0; row < num_rows; ++row) {
      auto const prev_offset = offset_blocks_accessor.get_current(allocation);
      offset_blocks_accessor.advance();
      if (validity.RowIsValid(row)) {
        auto const& str = str_data[row];
        auto const len  = str.GetSize();
        offset_blocks_accessor.set_current(prev_offset + len, allocation);

        // Copy string data
        data_blocks_accessor.memcpy_from(str.GetData(), len, allocation);

        // Update data bytes
        data_bytes += len;
      } else {
        offset_blocks_accessor.set_current(prev_offset, allocation);
      }
    }
    total_data_bytes += data_bytes;
  } else {
    // Fixed-width column
    auto const data_bytes = type_size * num_rows;
    data_blocks_accessor.memcpy_from(vec.GetData(), data_bytes, allocation);
    total_data_bytes += data_bytes;
  }

  process_mask_for_column(validity, num_rows, row_offset, allocation);
}

metadata_node duckdb_scan_task_local_state::column_builder::make_metadata_node(
  size_t num_rows) const
{
  if (type.InternalType() == duckdb::PhysicalType::VARCHAR) {
    // VARCHAR column
    return make_string_metadata_node(
      static_cast<cudf::size_type>(num_rows),
      static_cast<cudf::size_type>(null_count),
      static_cast<int64_t>(data_blocks_accessor.initial_byte_offset),
      static_cast<int64_t>(mask_blocks_accessor.initial_byte_offset),
      static_cast<int64_t>(offset_blocks_accessor.initial_byte_offset));
  } else {
    return make_flat_metadata_node(type,
                                   static_cast<cudf::size_type>(num_rows),
                                   static_cast<cudf::size_type>(null_count),
                                   static_cast<int64_t>(data_blocks_accessor.initial_byte_offset),
                                   static_cast<int64_t>(mask_blocks_accessor.initial_byte_offset));
  }
}

//===----------------------------------------------------------------------===//
// duckdb_scan_task_local_state
//===----------------------------------------------------------------------===//
//===----------Constructor----------===//
duckdb_scan_task_local_state::duckdb_scan_task_local_state(
  duckdb_scan_task_global_state& g_state,
  duckdb::ExecutionContext& exec_ctx,
  size_t approximate_batch_size,
  size_t default_varchar_size,
  std::unique_ptr<duckdb::LocalTableFunctionState> existing_local_tf_state)
  : _batch_size(approximate_batch_size),
    _default_varchar_size(default_varchar_size),
    _exec_ctx(exec_ctx)
{
  auto const& op = g_state._op;
  _num_columns   = op.projection_ids.size();

  // Reuse existing local table function state if provided
  if (existing_local_tf_state) {
    _local_tf_state = std::move(existing_local_tf_state);
  } else {
    g_state.increment_local_states();
  }

  // Estimate the number of rows per bach and adjust batch size accordingly
  estimate_rows_per_batch(op);

  // Make the memory reservation request
  auto& mem_res_mgr = g_state._sirius_ctx->get_memory_manager();
  _reservation      = mem_res_mgr.request_reservation(_res_request, _batch_size);

  // Make the allocation
  auto& mem_space = _reservation->get_memory_space();
  auto* allocator =
    mem_space.get_memory_resource_as<cucascade::memory::fixed_size_host_memory_resource>();
  if (allocator == nullptr) {
    throw std::runtime_error(
      "[duckdb_scan_task_local_state] Failed to get fixed_size_host_memory_resource allocator for "
      "HOST memory space with device id " +
      std::to_string(_reservation->device_id()) + ".");
  }
  _allocation = allocator->allocate_multiple_blocks(approximate_batch_size, _reservation.get());

  // Initialize the column builders
  initialize_builders();

  // Initialize local table function state (will skip if local_tf_state already set)
  initialize_local_table_function_state(op, exec_ctx, g_state._global_tf_state.get());
}

size_t duckdb_scan_task_local_state::get_last_byte_offset() const
{
  auto const& last_builder = _column_builders.back();
  auto last_byte_offset    = last_builder.mask_blocks_accessor.get_current_global_byte_offset();
  if (utils::mod_8(_row_offset) != 0) {
    last_byte_offset++;  // Round up to next byte if mask is partially filled
  }
  return std::min(last_byte_offset, _allocation->size_bytes());
}

void duckdb_scan_task_local_state::estimate_rows_per_batch(sirius_physical_table_scan const& op)
{
  assert(num_columns <= op.column_ids.size());

  // Construct column builders and collect the VARCHAR column indices
  // Also compute the base row size in bytes to estimate the rows per batch
  size_t base_row_bytes = 0;
  _column_builders.reserve(_num_columns);
  for (size_t i = 0; i < _num_columns; ++i) {
    auto const col_type = op.returned_types[op.column_ids[i].GetPrimaryIndex()];
    _column_builders.emplace_back(col_type, _default_varchar_size);
    base_row_bytes += _column_builders.back().type_size;
    if (col_type.InternalType() == duckdb::PhysicalType::VARCHAR) {
      _varchar_indices.push_back(i);
      base_row_bytes += sizeof(int64_t);
    }
  }
  base_row_bytes += utils::ceil_div_8(_num_columns);  // Mask bytes
  _estimated_rows_per_batch = std::max<size_t>(
    _batch_size / base_row_bytes, STANDARD_VECTOR_SIZE);  // Ensure at least 1 vector can fit

  auto bytes_for_rows = [this](size_t rows) {
    size_t bytes = 0;
    for (auto const& builder : _column_builders) {
      if (builder.type.InternalType() == duckdb::PhysicalType::VARCHAR) {
        auto const offset_bytes = (rows + 1) * sizeof(int64_t);
        auto const data_bytes   = utils::align_8(rows * _default_varchar_size);
        auto const mask_bytes   = utils::align_8(utils::ceil_div_8(rows));
        bytes += offset_bytes + data_bytes + mask_bytes;
      } else {
        auto const data_bytes = utils::align_8(rows * builder.type_size);
        auto const mask_bytes = utils::align_8(utils::ceil_div_8(rows));
        bytes += data_bytes + mask_bytes;
      }
    }
    return bytes;
  };

  // Revise the batch size allocation based on the number of estimated rows
  _batch_size = bytes_for_rows(_estimated_rows_per_batch);
}

void duckdb_scan_task_local_state::initialize_builders()
{
  size_t byte_offset = 0;
  for (size_t i = 0; i < _num_columns; ++i) {
    _column_builders[i].initialize_accessors(_estimated_rows_per_batch, byte_offset, _allocation);
    // Update byte_offset for next column
    if (_column_builders[i].type.InternalType() == duckdb::PhysicalType::VARCHAR) {
      // VARCHAR column (offsets + data + mask)
      auto const offset_bytes = (_estimated_rows_per_batch + 1) * sizeof(int64_t);
      auto const data_bytes   = utils::align_8(_estimated_rows_per_batch * _default_varchar_size);
      auto const mask_bytes   = utils::align_8(utils::ceil_div_8(_estimated_rows_per_batch));
      byte_offset += offset_bytes + data_bytes + mask_bytes;
    } else {
      // Fixed-width column (data + mask)
      auto const data_bytes =
        utils::align_8(_estimated_rows_per_batch * _column_builders[i].type_size);
      auto const mask_bytes = utils::align_8(utils::ceil_div_8(_estimated_rows_per_batch));
      byte_offset += data_bytes + mask_bytes;
    }
  }
}

void duckdb_scan_task_local_state::initialize_local_table_function_state(
  sirius_physical_table_scan const& op,
  duckdb::ExecutionContext& exec_ctx,
  duckdb::GlobalTableFunctionState* global_tf_state)
{
  // Note: local_tf_state might already be set if it was moved from a previous task
  // Only create a new one if it doesn't exist
  if (!_local_tf_state && op.function.init_local) {
    duckdb::TableFunctionInitInput tf_input(
      op.bind_data.get(), op.column_ids, op.projection_ids, nullptr, op.extra_info.sample_options);
    _local_tf_state = op.function.init_local(exec_ctx, tf_input, global_tf_state);
  }
}

std::shared_ptr<cucascade::data_batch> duckdb_scan_task_local_state::make_data_batch()
{
  using data_batch                = cucascade::data_batch;
  using host_table_allocation     = cucascade::memory::host_table_allocation;
  using host_table_representation = cucascade::host_table_representation;

  // Create metadata nodes for each column and assemble metadata buffer
  std::vector<metadata_node> column_metadata;
  column_metadata.reserve(_num_columns);
  for (auto const& builder : _column_builders) {
    column_metadata.push_back(builder.make_metadata_node(_row_offset));
  }
  auto metadata = std::make_unique<std::vector<uint8_t>>(pack_metadata_from_nodes(column_metadata));

  // Make the host table allocation
  auto const sz = get_last_byte_offset();
  // auto const sz = allocation->size_bytes();
  auto table_allocation =
    std::make_unique<host_table_allocation>(std::move(_allocation), std::move(metadata), sz);

  // Make the host table representation
  auto table =
    std::make_unique<host_table_representation>(std::move(table_allocation), _host_space);

  // Create the data batch and return
  return std::make_shared<data_batch>(get_next_batch_id(), std::move(table));
}

//===----------------------------------------------------------------------===//
// DuckDB Scan Task
//===----------------------------------------------------------------------===//
bool duckdb_scan_task::get_next_chunk(duckdb_scan_task_local_state& l_state,
                                      duckdb_scan_task_global_state& g_state)
{
  // Reset the chunk before calling the table function to ensure it starts empty
  l_state._chunk.Reset();

  duckdb::TableFunctionInput tf_input(
    g_state._op.bind_data.get(), l_state._local_tf_state.get(), g_state._global_tf_state.get());

  g_state._op.function.function(l_state._exec_ctx.client, tf_input, l_state._chunk);

  if (l_state._chunk.size() == 0) {
    if (!l_state._local_state_drained) {
      l_state._local_state_drained = true;
      g_state.decrement_local_states();
    }
    return false;
  }
  return true;
}

bool duckdb_scan_task::chunk_fits(duckdb_scan_task_local_state& l_state)
{
  // Loop over the VARCHAR columns and check if they fit in the allocated buffers
  for (auto varchar_idx : l_state._varchar_indices) {
    auto& vec = l_state._chunk.data[varchar_idx];
    vec.Flatten(l_state._chunk.size());
    auto const& validity = duckdb::FlatVector::Validity(l_state._chunk.data[varchar_idx]);
    if (!l_state._column_builders[varchar_idx].sufficient_space_for_column(
          vec, validity, l_state._chunk.size())) {
      return false;
    }
  }
  return true;
}

void duckdb_scan_task::process_chunk(duckdb_scan_task_local_state& l_state)
{
  for (size_t i = 0; i < l_state._num_columns; ++i) {
    auto& vec = l_state._chunk.data[i];
    vec.Flatten(l_state._chunk.size());
    auto const& validity = duckdb::FlatVector::Validity(vec);
    l_state._column_builders[i].process_column(
      vec, validity, l_state._chunk.size(), l_state._row_offset, l_state._allocation);
  }
  l_state._row_offset += l_state._chunk.size();
}

void duckdb_scan_task::execute()
{
  // Cast base task states to DuckDB scan task states
  auto& l_state = this->_local_state->cast<duckdb_scan_task_local_state>();
  auto& g_state = this->_global_state->cast<duckdb_scan_task_global_state>();

  // Initialize the data chunk
  l_state._chunk.Initialize(duckdb::Allocator::Get(l_state._exec_ctx.client),
                            g_state._op.returned_types);

  // Enter the scan loop to accumulate a data batch
  while (get_next_chunk(l_state, g_state)) {
    // We know a priori that the fixed-width columns and masks will fit in the allocated buffers.
    // For variable-length columns, we need to check that we have enough space.
    // If there isn't enough space, we just throw an exception for now.
    /// FUTURE WORK: push the current data batch into a new scan task.
    if (!chunk_fits(l_state)) {
      std::string err_msg =
        "[duckdb_scan_task]: current chunk does not fit in the allocated buffers.";
      throw std::runtime_error(err_msg);
    }

    // Process the chunk into the column builders
    process_chunk(l_state);

    // Termination condition
    if (STANDARD_VECTOR_SIZE + l_state._row_offset >= l_state._estimated_rows_per_batch) { break; }
  }

  // Add tasks back to the queue if the local scan state is not finished
  if (!l_state._local_state_drained) {
    // Create a new local state, passing the existing local_tf_state to continue the scan
    // This ensures DuckDB continues scanning from the current position rather than starting over
    auto new_local_state =
      std::make_unique<duckdb_scan_task_local_state>(g_state,
                                                     l_state._exec_ctx,
                                                     l_state._batch_size,
                                                     l_state._default_varchar_size,
                                                     std::move(l_state._local_tf_state));

    // Create a new reference to the global state
    auto const new_task_id = g_state._sirius_ctx->get_task_creator().get_next_task_id();
    auto shared_global_state =
      std::static_pointer_cast<duckdb_scan_task_global_state>(this->_global_state);
    auto next_task = std::make_unique<duckdb_scan_task>(
      new_task_id, _data_repo, std::move(new_local_state), shared_global_state);
    g_state._scan_executor.schedule(std::move(next_task));
  }

  // Make data batch and push to repository
  if (l_state._row_offset > 0) { _data_repo->add_data_batch(l_state.make_data_batch()); }
}

}  // namespace sirius::op::scan
