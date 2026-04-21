/*
 * Copyright 2025, RasterDB Contributors.
 *
 * Simplified Scan Executor Implementation
 */

#include "gpu/gpu_scan_executor.hpp"
#include "gpu/gpu_buffer_manager.hpp"
#include "log/logging.hpp"

#include <duckdb/main/client_context.hpp>
#include <duckdb/main/connection.hpp>
#include <duckdb/common/vector_operations/vector_operations.hpp>
#include <cstring>

namespace rasterdb::gpu {

gpu_scan_executor::gpu_scan_executor(gpu_context& ctx, int num_threads)
  : _ctx(ctx), _num_threads(num_threads > 0 ? num_threads : 4)
{}

std::unique_ptr<gpu_table> gpu_scan_executor::execute_scan(
  duckdb::ClientContext& client_ctx,
  duckdb::TableFunction& function,
  duckdb::FunctionData* bind_data,
  const duckdb::vector<duckdb::ColumnIndex>& column_ids,
  const duckdb::vector<duckdb::idx_t>& projection_ids,
  const duckdb::vector<duckdb::LogicalType>& scan_types,
  const std::string& table_name)
{
  // Convert DuckDB types to RDF types
  std::vector<rasterdf::data_type> rdf_types;
  for (auto& st : scan_types) {
    rdf_types.push_back(to_rdf_type(st));
  }
  size_t num_cols = rdf_types.size();

  // Initialize result table
  auto result = std::make_unique<gpu_table>();
  result->duckdb_types = scan_types;
  result->columns.resize(num_cols);
  rasterdf::size_type total_rows = 0;

  // Use direct table function call (like in gpu_executor.cpp)
  bool can_use_fast_path = function.function && !function.init_local;

  if (can_use_fast_path) {
    // Two-phase approach: first count rows, then allocate and fill
    // Phase 1: Scan to collect chunks
    std::vector<std::unique_ptr<duckdb::DataChunk>> all_chunks;
    {
      duckdb::TableFunctionInitInput init_input(
        bind_data, column_ids, projection_ids,
        nullptr, nullptr);

      auto global_state = function.init_global(client_ctx, init_input);
      duckdb::TableFunctionInput tf_input(bind_data, nullptr, global_state.get());

      while (true) {
        duckdb::DataChunk scan_chunk;
        scan_chunk.Initialize(duckdb::Allocator::DefaultAllocator(), scan_types);
        function.function(client_ctx, tf_input, scan_chunk);
        if (scan_chunk.size() == 0) break;
        scan_chunk.Flatten();

        // DuckDB seq_scan returns FLAT vectors whose storage points into the
        // table's buffer-managed pages.  Those pages may be evicted / reused
        // as subsequent chunks are fetched, silently invalidating earlier
        // chunks (manifests at scale: e.g. SF10 lineitem → wrong GROUP BY
        // keys).  Force a deep copy into a freshly-owned DataChunk so the
        // captured data survives until we upload it to the GPU staging
        // buffer.
        auto owned = duckdb::make_uniq<duckdb::DataChunk>();
        owned->Initialize(duckdb::Allocator::DefaultAllocator(), scan_types);
        owned->SetCardinality(scan_chunk.size());
        for (duckdb::idx_t c = 0; c < scan_types.size(); c++) {
          duckdb::VectorOperations::Copy(scan_chunk.data[c], owned->data[c],
                                         scan_chunk.size(), 0, 0);
        }

        total_rows += static_cast<rasterdf::size_type>(owned->size());
        all_chunks.push_back(std::move(owned));
      }
    }

    RASTERDB_LOG_DEBUG("Scan complete: {} rows in {} chunks", total_rows, all_chunks.size());

    if (total_rows == 0) {
      for (size_t c = 0; c < num_cols; c++) {
        result->columns[c].type = rdf_types[c];
        result->columns[c].num_rows = 0;
      }
      result->set_num_rows(0);
      return result;
    }

    // Phase 2: Allocate GPU columns and upload
    // Check if we can use zero-copy (reBAR) via GPUBufferManager
    if (GPUBufferManager::is_initialized()) {
      auto& bufMgr = GPUBufferManager::GetInstance();

      // Build column names for cache lookup
      std::vector<std::string> col_names;
      for (size_t c = 0; c < num_cols; c++) {
        col_names.push_back("col_" + std::to_string(c));
      }

      // Reset staging pointer
      bufMgr.cpuProcessingPointer.store(0, std::memory_order_relaxed);

      // Use from_buffer_manager for zero-copy if possible
      result = gpu_table::from_buffer_manager(_ctx, table_name, col_names, scan_types, all_chunks);
    } else {
      // Standard upload path
      result = gpu_table::from_data_chunks(_ctx, scan_types, all_chunks);
    }
  } else {
    // Fallback: use Connection::Query() for complex table functions
    auto conn = duckdb::make_uniq<duckdb::Connection>(*client_ctx.db);
    std::string col_list;
    for (size_t i = 0; i < column_ids.size(); i++) {
      if (i > 0) col_list += ", ";
      col_list += std::to_string(i);
    }
    if (col_list.empty()) col_list = "*";

    std::string scan_query = "SELECT " + col_list + " FROM \"" + table_name + "\"";
    auto query_result = conn->Query(scan_query);
    if (query_result->HasError()) {
      throw std::runtime_error("Scan query failed: " + query_result->GetError());
    }

    std::vector<std::unique_ptr<duckdb::DataChunk>> all_chunks;
    while (true) {
      auto chunk = query_result->Fetch();
      if (!chunk || chunk->size() == 0) break;
      total_rows += static_cast<rasterdf::size_type>(chunk->size());
      all_chunks.push_back(std::move(chunk));
    }

    if (total_rows == 0) {
      for (size_t c = 0; c < num_cols; c++) {
        result->columns[c].type = rdf_types[c];
        result->columns[c].num_rows = 0;
      }
      result->set_num_rows(0);
      return result;
    }

    // Use GPUBufferManager if available
    if (GPUBufferManager::is_initialized()) {
      auto& bufMgr = GPUBufferManager::GetInstance();
      std::vector<std::string> col_names;
      for (size_t c = 0; c < num_cols; c++) {
        col_names.push_back("col_" + std::to_string(c));
      }
      bufMgr.cpuProcessingPointer.store(0, std::memory_order_relaxed);
      result = gpu_table::from_buffer_manager(_ctx, table_name, col_names, scan_types, all_chunks);
    } else {
      result = gpu_table::from_data_chunks(_ctx, scan_types, all_chunks);
    }
  }

  RASTERDB_LOG_DEBUG("Scan executor returning table with {} rows x {} cols",
                     result->num_rows(), result->num_columns());
  for (size_t c = 0; c < result->num_columns(); c++) {
    RASTERDB_LOG_DEBUG("  col {}: {} rows, type={}", c, result->col(c).num_rows,
                       static_cast<int>(result->col(c).type.id));
  }

  return result;
}

} // namespace rasterdb::gpu
