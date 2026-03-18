/*
 * Copyright 2025, RasterDB Contributors.
 * GPU Table implementation — upload/download DuckDB data ↔ GPU device buffers.
 */

#include "gpu/gpu_table.hpp"
#include "log/logging.hpp"

#include <duckdb/common/types/vector.hpp>
#include <cstring>

namespace rasterdb {
namespace gpu {

gpu_column allocate_column(gpu_context& ctx, rasterdf::data_type type, rasterdf::size_type num_rows)
{
  size_t bytes = static_cast<size_t>(num_rows) * rdf_type_size(type.id);
  gpu_column col;
  col.type = type;
  col.num_rows = num_rows;
  col.data = rasterdf::device_buffer(ctx.workspace_mr(), bytes);
  return col;
}

rasterdf::device_buffer allocate_buffer(gpu_context& ctx, size_t bytes)
{
  return rasterdf::device_buffer(ctx.workspace_mr(), bytes);
}

size_t download_column(gpu_context& ctx, const gpu_column& col, void* dst, size_t dst_size)
{
  size_t bytes = col.byte_size();
  if (bytes > dst_size) {
    throw std::runtime_error("download_column: destination buffer too small");
  }
  if (bytes == 0) return 0;

  // Use device_buffer copy_to_host — needs non-const access for the staging copy
  // We create a temporary wrapper to call copy_to_host
  const_cast<rasterdf::device_buffer&>(col.data).copy_to_host(
    dst, bytes, ctx.device(), ctx.queue(), ctx.command_pool());
  return bytes;
}

/// Upload a flat host buffer to a new gpu_column.
static gpu_column upload_column(gpu_context& ctx, rasterdf::data_type type,
                                rasterdf::size_type num_rows, const void* host_data)
{
  gpu_column col = allocate_column(ctx, type, num_rows);
  size_t bytes = col.byte_size();
  if (bytes > 0 && host_data) {
    col.data.copy_from_host(host_data, bytes, ctx.device(), ctx.queue(), ctx.command_pool());
  }
  return col;
}

/// Extract flat data from a DuckDB Vector into a contiguous host buffer.
/// Returns the number of valid rows extracted.
static rasterdf::size_type flatten_vector(const duckdb::Vector& vec,
                                           rasterdf::size_type count,
                                           rasterdf::data_type rdf_type,
                                           std::vector<uint8_t>& out_buf)
{
  size_t elem_size = rdf_type_size(rdf_type.id);
  size_t offset = out_buf.size();
  out_buf.resize(offset + static_cast<size_t>(count) * elem_size);

  auto data_ptr = reinterpret_cast<const uint8_t*>(vec.GetData());
  std::memcpy(out_buf.data() + offset, data_ptr, static_cast<size_t>(count) * elem_size);
  return count;
}

std::unique_ptr<gpu_table> gpu_table::from_data_chunks(
  gpu_context& ctx,
  const std::vector<duckdb::LogicalType>& types,
  const std::vector<std::unique_ptr<duckdb::DataChunk>>& chunks)
{
  auto table = std::make_unique<gpu_table>();
  table->duckdb_types = types;

  if (chunks.empty()) {
    table->_num_rows = 0;
    table->columns.resize(types.size());
    for (size_t c = 0; c < types.size(); c++) {
      table->columns[c].type = to_rdf_type(types[c]);
      table->columns[c].num_rows = 0;
    }
    return table;
  }

  size_t num_cols = types.size();

  // First pass: compute total rows and validate types
  rasterdf::size_type total_rows = 0;
  std::vector<rasterdf::data_type> rdf_types(num_cols);
  for (size_t c = 0; c < num_cols; c++) {
    rdf_types[c] = to_rdf_type(types[c]); // throws for unsupported types
  }
  for (auto& chunk : chunks) {
    total_rows += static_cast<rasterdf::size_type>(chunk->size());
  }

  if (total_rows == 0) {
    table->_num_rows = 0;
    table->columns.resize(num_cols);
    for (size_t c = 0; c < num_cols; c++) {
      table->columns[c].type = rdf_types[c];
      table->columns[c].num_rows = 0;
    }
    return table;
  }

  // Second pass: flatten each column across all chunks, then upload to GPU
  table->columns.resize(num_cols);
  for (size_t c = 0; c < num_cols; c++) {
    std::vector<uint8_t> host_buf;
    host_buf.reserve(static_cast<size_t>(total_rows) * rdf_type_size(rdf_types[c].id));

    for (auto& chunk : chunks) {
      chunk->Flatten(); // ensure flat vectors
      flatten_vector(chunk->data[c], static_cast<rasterdf::size_type>(chunk->size()),
                     rdf_types[c], host_buf);
    }

    table->columns[c] = upload_column(ctx, rdf_types[c], total_rows, host_buf.data());
  }

  table->_num_rows = total_rows;
  RASTERDB_LOG_DEBUG("GPU table uploaded: {} rows x {} cols", total_rows, num_cols);
  return table;
}

void gpu_table::append_chunk(gpu_context& ctx, const duckdb::DataChunk& chunk,
                              const std::vector<duckdb::LogicalType>& types)
{
  (void)ctx; (void)chunk; (void)types;
  throw duckdb::NotImplementedException("gpu_table::append_chunk not yet implemented");
}

rasterdf::table_view gpu_table::view() const
{
  std::vector<rasterdf::column_view> views;
  views.reserve(columns.size());
  for (auto& col : columns) {
    views.push_back(col.view());
  }
  return rasterdf::table_view(views);
}

} // namespace gpu
} // namespace rasterdb
