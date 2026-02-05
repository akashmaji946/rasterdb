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

// cucascade
#include <cucascade/data/common.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_space.hpp>

// cudf
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/types.hpp>

// standard library
#include <memory>
#include <vector>

namespace sirius {

/**
 * @brief A host representation of Parquet data for use in a hybrid scan.
 *
 * This class encapsulates the necessary components to decompress a slice and/or projection of
 * Parquet data using cudf's hybrid scan capabilities.
 * See:
 * https://docs.rapids.ai/api/libcudf/stable/classcudf_1_1io_1_1parquet_1_1experimental_1_1hybrid__scan__reader
 */
class host_parquet_representation : public cucascade::idata_representation {
  using hybrid_scan_reader = cudf::io::parquet::experimental::hybrid_scan_reader;

 public:
  /**
   * @brief Constructs a host_parquet_representation.
   *
   * @param[in] memory_space The memory space to which the representation beloongs.
   * @param[in] column_chunks The fixed multiple blocks allocation containing the Parquet column
   * chunks.
   * @param[in] parquet_reader An instance hybrid scan Parquet reader for a given Parquet file.
   * @param[in] reader_options The Parquet reader options used to configure the hybrid scan reader
   * for materializing data.
   * @param[in] row_group_indices The row group indices of the row groups represented in the
   * multiple blocks allocation.
   * @param[in] column_chunk_byte_ranges The byte ranges in the multiple blocks allocation
   * representing the column chunks to be read.
   * @param[in] size_in_bytes The size of the representation in bytes
   */
  host_parquet_representation(cucascade::memory::memory_space* memory_space,
                              cucascade::memory::fixed_multiple_blocks_allocation column_chunks,
                              std::unique_ptr<hybrid_scan_reader> parquet_reader,
                              cudf::io::parquet_reader_options reader_options,
                              std::vector<cudf::size_type> row_group_indices,
                              std::vector<cudf::io::text::byte_range_info> column_chunk_byte_ranges,
                              size_t size_in_bytes)
    : idata_representation(*memory_space),
      _column_chunks(std::move(column_chunks)),
      _parquet_reader(std::move(parquet_reader)),
      _reader_options(std::move(reader_options)),
      _row_group_indices(std::move(row_group_indices)),
      _column_chunk_byte_ranges(std::move(column_chunk_byte_ranges)),
      size_in_bytes(size_in_bytes) {};

  [[nodiscard]] size_t get_size_in_bytes() const override { return size_in_bytes; }

  [[nodiscard]] cucascade::memory::fixed_multiple_blocks_allocation const& get_column_chunks() const
  {
    return _column_chunks;
  };

  [[nodiscard]] hybrid_scan_reader const& get_parquet_reader() const { return *_parquet_reader; };
  [[nodiscard]] std::unique_ptr<hybrid_scan_reader> move_parquet_reader()
  {
    return std::move(_parquet_reader);
  };

  [[nodiscard]] cudf::io::parquet_reader_options const& get_reader_options() const
  {
    return _reader_options;
  };

  [[nodiscard]] std::vector<cudf::size_type> const& get_row_group_indices() const
  {
    return _row_group_indices;
  };
  [[nodiscard]] std::vector<cudf::size_type>& get_row_group_indices()
  {
    return _row_group_indices;
  };

  [[nodiscard]] cudf::host_span<cudf::size_type const> get_rg_span() const
  {
    return cudf::host_span<cudf::size_type const>(_row_group_indices.data(),
                                                  _row_group_indices.size());
  };

  [[nodiscard]] std::vector<cudf::io::text::byte_range_info> const& get_column_chunk_byte_ranges()
    const
  {
    return _column_chunk_byte_ranges;
  };
  [[nodiscard]] std::vector<cudf::io::text::byte_range_info>& get_column_chunk_byte_ranges()
  {
    return _column_chunk_byte_ranges;
  };

 private:
  cucascade::memory::fixed_multiple_blocks_allocation _column_chunks;
  std::unique_ptr<hybrid_scan_reader> _parquet_reader;
  cudf::io::parquet_reader_options _reader_options;
  std::vector<cudf::size_type> _row_group_indices;
  std::vector<cudf::io::text::byte_range_info> _column_chunk_byte_ranges;
  size_t size_in_bytes;
};
}  // namespace sirius
