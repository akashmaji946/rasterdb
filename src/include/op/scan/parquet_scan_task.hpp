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
#include <config.hpp>
#include <memory/multiple_blocks_allocation_accessor.hpp>
#include <op/sirius_physical_table_scan.hpp>
#include <parallel/task.hpp>
#include <sirius_context.hpp>

// cucascade
#include <cucascade/data/data_repository.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation.hpp>

// duckdb
#include <duckdb/main/client_context.hpp>

// cudf
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_schema.hpp>

// standard library
#include <atomic>
#include <memory>
#include <vector>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// Parquet Scan Task Global State
//===----------------------------------------------------------------------===//
class parquet_scan_task_global_state : public parallel::itask_global_state {
  using hybrid_scan_reader = cudf::io::parquet::experimental::hybrid_scan_reader;

 public:
  /**
   * @brief Struct representing a range of row groups assigned to a scan task.
   */
  struct row_group_range {
    size_t start_row_group;
    size_t row_group_count;
    size_t reserved_uncompressed_bytes;
    size_t reserved_compressed_bytes;
  };

  //===----------Constructor----------===//
  parquet_scan_task_global_state(
    sirius_physical_table_scan const* scan_op,
    duckdb::ClientContext const& client_ctx,
    size_t approximate_batch_size = duckdb::Config::DEFAULT_SCAN_TASK_BATCH_SIZE);

  //===----------Methods----------===//
  [[nodiscard]] duckdb::SiriusContext& get_sirius_context() { return *_sirius_ctx; }
  [[nodiscard]] std::string const& get_file_path() const { return _file_path; }
  [[nodiscard]] cudf::io::parquet_reader_options const& get_options() const
  {
    return _reader_options;
  }
  [[nodiscard]] size_t get_num_row_group_partitions() const { return _row_group_partitions.size(); }
  [[nodiscard]] size_t get_next_rg_partition_idx()
  {
    return _next_rg_partition.fetch_add(1, std::memory_order_relaxed);
  }
  [[nodiscard]] row_group_range const& get_row_group_partition(size_t idx) const
  {
    return _row_group_partitions[idx];
  }
  [[nodiscard]] std::unique_ptr<cudf::io::parquet::experimental::hybrid_scan_reader> make_reader()
    const
  {
    return std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(
      cudf::host_span<uint8_t const>(_footer_buffer->data(), _footer_buffer->size()),
      _reader_options);
  }

 private:
  void make_selected_column_indices(sirius_physical_table_scan const& scan_op);
  void accumulate_row_group_byte_sizes(cudf::io::parquet::FileMetaData const& file_metadata);
  void partition_row_groups(cudf::io::parquet::FileMetaData const& file_metadata);

  //===----------Fields----------===//
  duckdb::shared_ptr<duckdb::SiriusContext> _sirius_ctx;  ///< The Sirius context
  size_t _approximate_batch_size;  ///< Target approximate batch size for scan tasks
  bool _is_projected;              ///< Whether projection is applied

  std::string _file_path;  ///< The parquet file path
  std::unique_ptr<cudf::io::datasource::buffer>
    _footer_buffer;                                  ///< The parquet file footer metadata
  cudf::io::parquet_reader_options _reader_options;  ///< Parquet reader options

  std::vector<size_t> _row_group_uncompressed_bytes;   ///< Per-row-group uncompressed bytes
  std::vector<size_t> _row_group_compressed_bytes;     ///< Per-row-group compressed bytes
  std::vector<row_group_range> _row_group_partitions;  ///< Row-group partitions for tasks
  std::vector<size_t> _selected_column_indices;        ///< Column indices to read (projection)

  std::atomic<size_t> _next_rg_partition{0};  ///< Number of local states created
};

//===----------------------------------------------------------------------===//
// Parquet Scan Task Local State
//===----------------------------------------------------------------------===//

class parquet_scan_task_local_state : public parallel::itask_local_state {
  using multiple_blocks_allocation =
    cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation;
  using multiple_blocks_allocation_accessor = memory::multiple_blocks_allocation_accessor<uint8_t>;

 public:
  //===----------Constructor----------===//
  parquet_scan_task_local_state(parquet_scan_task_global_state& g_state);

  //===----------Methods----------===//

  void read_range_into_allocation(size_t file_offset, size_t n_bytes);

  [[nodiscard]] cucascade::memory::reservation const& get_reservation() const
  {
    return *_reservation;
  };
  [[nodiscard]] std::unique_ptr<multiple_blocks_allocation> move_allocation()
  {
    return std::move(_allocation);
  };

  [[nodiscard]] cudf::host_span<cudf::size_type const> get_rg_span() const
  {
    return cudf::host_span<cudf::size_type const>(_rg_indices.data(), _rg_indices.size());
  };

  [[nodiscard]] std::vector<cudf::size_type> move_rg_indices() { return std::move(_rg_indices); }

 private:
  /**
   * @brief Reserve the next row-group range for this local state.
   *
   * @param[in] g_state The global state for the scan task.
   * @return true if row groups were reserved, false otherwise.
   * @note False is only returned if superfluous tasks were scheduled.
   */
  bool reserve_row_groups(parquet_scan_task_global_state& g_state);

  std::unique_ptr<cudf::io::datasource> _datasource;  ///< The cudf datasource for the input file
  std::unique_ptr<cucascade::memory::reservation>
    _reservation;  ///< Memory reservation for this local state's allocation
  std::unique_ptr<multiple_blocks_allocation>
    _allocation;  ///< The memory allocation into which parquet data is read
  multiple_blocks_allocation_accessor _data_blocks_accessor;  ///< Accessor for the allocation
  size_t _reserved_uncompressed_bytes =
    0;  ///< Number of uncompressed bytes reserved by the row group range
  size_t _reserved_compressed_bytes =
    0;  ///< Number of compressed bytes reserved by the row group range

  std::vector<cudf::size_type> _rg_indices;
};

//===----------------------------------------------------------------------===//
// Parquet Scan Task
//===----------------------------------------------------------------------===//

/**
 * @brief A scan task for stitching parquet row groups into a parquet blob data batch.
 */
class parquet_scan_task : public sirius::parallel::itask {
  using shared_data_repository = cucascade::shared_data_repository;

 public:
  //===----------Constructor----------===//
  /**
   * @brief Construct a new parquet_scan_task object.
   *
   * @param[in] task_id The unique ID of this task.
   * @param[in] data_repo The shared data repository to which the produced data batch will be
   * pushed.
   * @param[in] l_state The local state for this task.
   * @param[in] g_state The global state for this task.
   */
  parquet_scan_task(uint64_t task_id,
                    shared_data_repository* data_repo,
                    std::unique_ptr<parquet_scan_task_local_state> l_state,
                    std::shared_ptr<parquet_scan_task_global_state> g_state)
    : _task_id(task_id), _data_repo(data_repo), sirius::parallel::itask(std::move(l_state), g_state)
  {
  }

  //===----------Methods----------===//
  void execute() override;
  [[nodiscard]] uint64_t get_task_id() const { return _task_id; }

 private:
  shared_data_repository* _data_repo;  ///< The shared data repository to which to push batches
  uint64_t _task_id;                   ///< The unique ID of this task
};

}  // namespace sirius::op::scan
