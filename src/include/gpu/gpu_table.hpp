/*
 * Copyright 2025, RasterDB Contributors.
 * GPU Table — holds columnar data on GPU as rasterdf device_buffers.
 */

#pragma once

#include "gpu_context.hpp"
#include "gpu_types.hpp"

#include <rasterdf/core/column.hpp>
#include <rasterdf/core/column_view.hpp>
#include <rasterdf/core/device_buffer.hpp>
#include <rasterdf/core/table.hpp>
#include <rasterdf/core/table_view.hpp>

#include <duckdb/common/types/data_chunk.hpp>
#include <duckdb/common/types/vector.hpp>

#include <memory>
#include <vector>

namespace rasterdb {
namespace gpu {

/// A single column residing on GPU memory.
struct gpu_column {
  rasterdf::device_buffer data;
  rasterdf::data_type type;
  rasterdf::size_type num_rows{0};

  /// Get a column_view for passing to dispatcher calls.
  rasterdf::column_view view() const {
    return rasterdf::column_view(type, num_rows, data.data(), 0, 0, 0);
  }

  VkDeviceAddress address() const { return data.data(); }
  size_t byte_size() const { return static_cast<size_t>(num_rows) * rdf_type_size(type.id); }
};

/// A table (collection of columns) residing on GPU memory.
class gpu_table {
public:
  gpu_table() = default;

  /// Upload a DuckDB DataChunk to GPU memory.
  /// Only numeric types are supported; strings/decimals throw NotImplementedException.
  static std::unique_ptr<gpu_table> from_data_chunks(
    gpu_context& ctx,
    const std::vector<duckdb::LogicalType>& types,
    const std::vector<std::unique_ptr<duckdb::DataChunk>>& chunks);

  /// Upload a single DataChunk to GPU (appends to existing data).
  void append_chunk(gpu_context& ctx, const duckdb::DataChunk& chunk,
                    const std::vector<duckdb::LogicalType>& types);

  /// Number of columns.
  size_t num_columns() const { return columns.size(); }

  /// Number of rows (from _num_rows, or inferred from first column).
  rasterdf::size_type num_rows() const {
    if (_num_rows > 0) return _num_rows;
    if (!columns.empty()) return columns[0].num_rows;
    return 0;
  }

  /// Set the row count explicitly.
  void set_num_rows(rasterdf::size_type n) { _num_rows = n; }

  /// Access a column.
  gpu_column& col(size_t i) { return columns[i]; }
  const gpu_column& col(size_t i) const { return columns[i]; }

  /// Get rasterdf table_view for all columns.
  rasterdf::table_view view() const;

  /// Column storage.
  std::vector<gpu_column> columns;

  /// Column types (DuckDB types, for result conversion).
  std::vector<duckdb::LogicalType> duckdb_types;

private:
  rasterdf::size_type _num_rows{0};
};

/// Download a gpu_column back to a flat host buffer.
/// Returns the number of bytes written.
size_t download_column(gpu_context& ctx, const gpu_column& col, void* dst, size_t dst_size);

/// Allocate a gpu_column of given type and size, uninitialized.
gpu_column allocate_column(gpu_context& ctx, rasterdf::data_type type, rasterdf::size_type num_rows);

/// Allocate a small device buffer (e.g., for counters, scalars).
rasterdf::device_buffer allocate_buffer(gpu_context& ctx, size_t bytes);

} // namespace gpu
} // namespace rasterdb
