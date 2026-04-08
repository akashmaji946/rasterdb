/*
 * Copyright 2025, RasterDB Contributors.
 *
 * Pipelined Scan Executor — multi-threaded DuckDB scan with direct GPU memory write.
 * Mirrors Sirius duckdb_scan_executor but simplified for RasterDB architecture.
 */

#pragma once

#include "gpu/gpu_context.hpp"
#include "gpu/gpu_table.hpp"

#include <duckdb/function/table_function.hpp>
#include <duckdb/common/types/data_chunk.hpp>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <future>

namespace rasterdb::gpu {

//===----------------------------------------------------------------------===//
// GPU Scan Executor — multi-threaded scan with direct GPU memory write
//===----------------------------------------------------------------------===//
class gpu_scan_executor {
 public:
  explicit gpu_scan_executor(gpu_context& ctx, int num_threads = 4);

  // Execute scan + upload
  std::unique_ptr<gpu_table> execute_scan(
    duckdb::ClientContext& client_ctx,
    duckdb::TableFunction& function,
    duckdb::FunctionData* bind_data,
    const duckdb::vector<duckdb::ColumnIndex>& column_ids,
    const duckdb::vector<duckdb::idx_t>& projection_ids,
    const duckdb::vector<duckdb::LogicalType>& scan_types,
    const std::string& table_name);

 private:
  gpu_context& _ctx;
  int _num_threads;
};

} // namespace rasterdb::gpu
