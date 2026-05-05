/*
 * Copyright 2025, RasterDB Contributors.
 * Split from src/gpu/gpu_executor.cpp.
 */

#include "gpu/gpu_executor_internal.hpp"

namespace rasterdb {
namespace gpu {

// ============================================================================
// Convert gpu_table -> DuckDB QueryResult
// ============================================================================

duckdb::unique_ptr<duckdb::QueryResult> gpu_executor::to_query_result(
  std::unique_ptr<gpu_table> table,
  const duckdb::vector<duckdb::string>& names,
  const duckdb::vector<duckdb::LogicalType>& types)
{
  stage_timer t("  gpu_download+result");
  // Download full columns to CPU first, then build DataChunks
  size_t num_cols = table->num_columns();
  rasterdf::size_type total_rows = table->num_rows();
  RASTERDB_LOG_DEBUG("[RDB_DEBUG]     to_query_result: {} cols, {} rows",
                     num_cols, total_rows);
  for (size_t c = 0; c < num_cols; c++) {
    auto& col = table->col(c);
    RASTERDB_LOG_DEBUG("[RDB_DEBUG]       col[{}]: type_id={}, num_rows={}, is_host_only={}, host_data_sz={}",
                       c, static_cast<int>(col.type.id), col.num_rows,
                       col.is_host_only, col.host_data.size());
  }

  // Download all column data to host buffers
  auto t_dl = std::chrono::high_resolution_clock::now();
  std::vector<std::vector<uint8_t>> host_data(num_cols);
  size_t total_dl_bytes = 0;
  for (size_t c = 0; c < num_cols; c++) {
    auto& col = table->col(c);
    if (col.is_host_only) {
      // Column data already on CPU — use it directly (perf fix: skip GPU download)
      host_data[c] = col.host_data;
      RASTERDB_LOG_DEBUG("[RDB_DEBUG]     col[{}] host-only, {} bytes (no GPU download)",
                         c, host_data[c].size());
    } else {
      size_t bytes = col.byte_size();
      host_data[c].resize(bytes);
      if (bytes > 0) {
        download_column(_ctx, col, host_data[c].data(), bytes);
        total_dl_bytes += bytes;
      }
    }
  }
  auto t_dl_end = std::chrono::high_resolution_clock::now();
  RASTERDB_LOG_DEBUG("[RDB_DEBUG]     col_download: {} cols, {} bytes  {:8.2f} ms",
                     num_cols, total_dl_bytes,
                     std::chrono::duration<double, std::milli>(t_dl_end - t_dl).count());

  // Build ColumnDataCollection
  auto collection = duckdb::make_uniq<duckdb::ColumnDataCollection>(
    duckdb::Allocator::DefaultAllocator(), duckdb::vector<duckdb::LogicalType>(types));

  rasterdf::size_type chunk_size = STANDARD_VECTOR_SIZE;
  for (rasterdf::size_type offset = 0; offset < total_rows; offset += chunk_size) {
    rasterdf::size_type count = std::min(chunk_size, total_rows - offset);
    duckdb::DataChunk chunk;
    chunk.Initialize(duckdb::Allocator::DefaultAllocator(),
                     duckdb::vector<duckdb::LogicalType>(types));
    chunk.SetCardinality(count);

    for (size_t c = 0; c < num_cols; c++) {
      auto rdf_tid = table->col(c).type.id;
      auto duckdb_tid = types[c].id();

      // ── DICTIONARY32 → VARCHAR decode ──
      if (rdf_tid == rasterdf::type_id::DICTIONARY32 &&
          duckdb_tid == duckdb::LogicalTypeId::VARCHAR &&
          table->dictionaries.has_dict(c)) {
        const auto& dict = table->dictionaries.get(c);
        const auto* codes = reinterpret_cast<const int32_t*>(
            host_data[c].data() + static_cast<size_t>(offset) * sizeof(int32_t));
        auto& vec = chunk.data[c];
        for (rasterdf::size_type r = 0; r < count; r++) {
          const auto& str = dict.decode(codes[r]);
          vec.SetValue(r, duckdb::Value(str));
        }
        continue;
      }

      size_t rdf_elem_size = rdf_type_size(rdf_tid);
      size_t byte_offset = static_cast<size_t>(offset) * rdf_elem_size;
      const uint8_t* src = host_data[c].data() + byte_offset;
      auto dst = reinterpret_cast<uint8_t*>(chunk.data[c].GetData());

      // If the rasterdf type matches the DuckDB type in size, direct copy
      bool needs_cast = false;

      // Check for type mismatches that need widening/casting
      if (rdf_tid == rasterdf::type_id::INT32 && duckdb_tid == duckdb::LogicalTypeId::HUGEINT) {
        // int32 → hugeint (int128): widen each element
        for (rasterdf::size_type r = 0; r < count; r++) {
          int32_t val;
          std::memcpy(&val, src + r * sizeof(int32_t), sizeof(int32_t));
          duckdb::hugeint_t hval;
          hval.lower = static_cast<uint64_t>(val < 0 ? -static_cast<int64_t>(-val) : val);
          hval.upper = val < 0 ? -1 : 0;
          std::memcpy(dst + r * sizeof(duckdb::hugeint_t), &hval, sizeof(duckdb::hugeint_t));
        }
        needs_cast = true;
      } else if (rdf_tid == rasterdf::type_id::INT64 && duckdb_tid == duckdb::LogicalTypeId::HUGEINT) {
        // int64 → hugeint (int128): widen each element
        for (rasterdf::size_type r = 0; r < count; r++) {
          int64_t val;
          std::memcpy(&val, src + r * sizeof(int64_t), sizeof(int64_t));
          duckdb::hugeint_t hval;
          hval.lower = static_cast<uint64_t>(val);
          hval.upper = val < 0 ? -1 : 0;
          std::memcpy(dst + r * sizeof(duckdb::hugeint_t), &hval, sizeof(duckdb::hugeint_t));
        }
        needs_cast = true;
      } else if (rdf_tid == rasterdf::type_id::INT32 && duckdb_tid == duckdb::LogicalTypeId::BIGINT) {
        // int32 → bigint (int64): widen each element
        for (rasterdf::size_type r = 0; r < count; r++) {
          int32_t val;
          std::memcpy(&val, src + r * sizeof(int32_t), sizeof(int32_t));
          int64_t wide = static_cast<int64_t>(val);
          std::memcpy(dst + r * sizeof(int64_t), &wide, sizeof(int64_t));
        }
        needs_cast = true;
      } else if (rdf_tid == rasterdf::type_id::FLOAT32 && duckdb_tid == duckdb::LogicalTypeId::DOUBLE) {
        // float32 → double: widen each element
        for (rasterdf::size_type r = 0; r < count; r++) {
          float val;
          std::memcpy(&val, src + r * sizeof(float), sizeof(float));
          double wide = static_cast<double>(val);
          std::memcpy(dst + r * sizeof(double), &wide, sizeof(double));
        }
        needs_cast = true;
      } else if (rdf_tid == rasterdf::type_id::INT64 && duckdb_tid == duckdb::LogicalTypeId::BIGINT) {
        // int64 → bigint: same size, direct copy
        std::memcpy(dst, src, static_cast<size_t>(count) * rdf_elem_size);
        needs_cast = true;
      } else if (rdf_tid == rasterdf::type_id::FLOAT64 && duckdb_tid == duckdb::LogicalTypeId::DOUBLE) {
        // float64 → double: same size, direct copy
        std::memcpy(dst, src, static_cast<size_t>(count) * rdf_elem_size);
        needs_cast = true;
      }

      if (!needs_cast) {
        // Default: types match in size, direct copy
        std::memcpy(dst, src, static_cast<size_t>(count) * rdf_elem_size);
      }
    }

    collection->Append(chunk);
  }

  auto props = _client_ctx.GetClientProperties();
  return duckdb::make_uniq<duckdb::MaterializedQueryResult>(
    duckdb::StatementType::SELECT_STATEMENT,
    duckdb::StatementProperties(),
    duckdb::vector<duckdb::string>(names),
    std::move(collection), props);
}

} // namespace gpu
} // namespace rasterdb
