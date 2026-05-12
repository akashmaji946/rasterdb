/*
 * Copyright 2025, RasterDB Contributors.
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

/*
 * RasterDB DuckDB Extension — Vulkan/rasterdf GPU execution.
 * Executes SQL queries on GPU via rasterdf compute shaders.
 * Falls back to DuckDB CPU for unsupported operations.
 */

#include "duckdb/main/database.hpp"
#define DUCKDB_EXTENSION_MAIN

#include "config.hpp"
#include "duckdb/common/assert.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/query_result.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/planner.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"
#include "log/logging.hpp"
#include "rasterdb_extension.hpp"
#include "util/segfault_backtrace.hpp"
#include "gpu/gpu_context.hpp"
#include "gpu/gpu_buffer_manager.hpp"
#include "gpu/gpu_executor.hpp"
#include <duckdb/common/types/vector.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace duckdb {

bool RasterdbExtension::buffer_is_initialized = false;

// =========================================================================
// gpu_buffer_init(cache_size, processing_size) — pre-allocate GPU buffers.
// Mirrors Sirius's gpu_buffer_init('2GB', '2GB') API exactly.
// =========================================================================

struct GPUBufferInitFunctionData : public TableFunctionData {
  size_t cache_size = 0;
  size_t processing_size = 0;
  size_t pinned_memory_size = 0;
  bool finished = false;
};

static size_t parse_size_string(const string& size_str) {
  size_t multiplier = 1;
  string num_part, unit_part;
  size_t i = 0;
  while (i < size_str.length() && isspace(size_str[i])) i++;
  while (i < size_str.length() && (isdigit(size_str[i]) || size_str[i] == '.')) {
    num_part += size_str[i]; i++;
  }
  while (i < size_str.length() && isspace(size_str[i])) i++;
  unit_part = size_str.substr(i);
  double num_value = stod(num_part);
  if (unit_part == "B") multiplier = 1;
  else if (unit_part == "KB" || unit_part == "KiB") multiplier = 1024;
  else if (unit_part == "MB" || unit_part == "MiB") multiplier = 1024 * 1024;
  else if (unit_part == "GB" || unit_part == "GiB") multiplier = 1024ULL * 1024 * 1024;
  else if (unit_part == "TB" || unit_part == "TiB") multiplier = 1024ULL * 1024 * 1024 * 1024;
  else throw InvalidInputException("gpu_buffer_init: invalid size format '" + size_str + "'");
  return static_cast<size_t>(num_value * multiplier);
}

static unique_ptr<FunctionData> GPUBufferInitBind(ClientContext& context,
                                                  TableFunctionBindInput& input,
                                                  vector<LogicalType>& return_types,
                                                  vector<string>& names) {
  auto result = make_uniq<GPUBufferInitFunctionData>();
  string cache_str = input.inputs[0].ToString();
  string processing_str = input.inputs[1].ToString();
  result->cache_size = parse_size_string(cache_str);
  result->processing_size = parse_size_string(processing_str);
  result->pinned_memory_size = std::max(result->cache_size, result->processing_size);
  return_types.emplace_back(LogicalType(LogicalTypeId::BOOLEAN));
  names.emplace_back("Success");
  return std::move(result);
}

static void GPUBufferInitFunction(ClientContext& context,
                                  TableFunctionInput& data_p,
                                  DataChunk& output) {
  auto& data = data_p.bind_data->CastNoConst<GPUBufferInitFunctionData>();
  if (data.finished) return;

  if (!rasterdb::gpu::GPUBufferManager::is_initialized()) {
    RASTERDB_LOG_INFO("gpu_buffer_init: cache={}MB, processing={}MB, staging={}MB",
                      data.cache_size / (1024*1024),
                      data.processing_size / (1024*1024),
                      data.pinned_memory_size / (1024*1024));
    auto& bufMgr = rasterdb::gpu::GPUBufferManager::GetInstance(
        data.cache_size, data.processing_size, data.pinned_memory_size);
    (void)bufMgr;
    RasterdbExtension::buffer_is_initialized = true;
  } else {
    RASTERDB_LOG_WARN("GPUBufferManager already initialized");
  }
  output.SetCardinality(1);
  output.SetValue(0, 0, Value::BOOLEAN(true));
  data.finished = true;
}

// =========================================================================
// gpu_execution — execute SQL on GPU via rasterdf Vulkan compute.
// Falls back to DuckDB CPU for unsupported operators.
// =========================================================================

struct RasterDBQueryData : public TableFunctionData {
  unique_ptr<QueryResult> cpu_result;   // CPU fallback result
  std::unique_ptr<rasterdb::gpu::gpu_table> result_table; // Lazy GPU table stream
  idx_t chunk_offset = 0;               // Streaming offset
  unique_ptr<Connection> conn;
  string query;
  bool finished = false;
  bool attempted = false;

  // Zero-copy host pointers: point directly into GPUBufferManager's HOST_CACHED
  // download buffer after DMA. No heap allocation, no memcpy, no page faults.
  std::vector<const uint8_t*> host_column_ptrs;
  bool host_cache_ready = false;
};

static unique_ptr<FunctionData> GPUExecutionBind(ClientContext& context,
                                                  TableFunctionBindInput& input,
                                                  vector<LogicalType>& return_types,
                                                  vector<string>& names)
{
  auto result  = make_uniq<RasterDBQueryData>();
  result->conn = make_uniq<Connection>(*context.db);
  result->query = input.inputs[0].ToString();
  if (input.inputs[0].IsNull()) {
    throw BinderException("gpu_execution cannot be called with a NULL parameter");
  }

  // Parse and plan to get column names/types
  Parser parser(context.GetParserOptions());
  parser.ParseQuery(result->query);
  Planner planner(context);
  planner.CreatePlan(std::move(parser.statements[0]));

  for (auto& col : planner.names) { names.emplace_back(col); }
  for (auto& type : planner.types) { return_types.emplace_back(type); }

  return std::move(result);
}

static void GPUExecutionFunction(ClientContext& context,
                                  TableFunctionInput& data_p,
                                  DataChunk& output)
{
  auto& data = data_p.bind_data->CastNoConst<RasterDBQueryData>();
  static std::mutex gpu_execution_mutex;
  std::lock_guard<std::mutex> gpu_execution_lock(gpu_execution_mutex);
  if (data.finished) { return; }

  // First call: attempt GPU execution, fall back to CPU on failure
  if (!data.attempted) {
    data.attempted = true;

    // Try GPU execution if the GPU context is initialized
    if (rasterdb::gpu::gpu_context::is_initialized()) {
      try {
        auto t_total_start = std::chrono::high_resolution_clock::now();
        RASTERDB_LOG_DEBUG("[TIMER] === Query: {}{}",
                data.query.substr(0, 60), data.query.size() > 60 ? "..." : "");

        auto t0 = std::chrono::high_resolution_clock::now();
        // Parse + plan the query
        Parser parser(context.GetParserOptions());
        parser.ParseQuery(data.query);
        Planner planner(context);
        planner.CreatePlan(std::move(parser.statements[0]));

        // Use unoptimized plan to avoid spurious FILTER nodes from optimizer
        auto& plan = *planner.plan;

        // Resolve column bindings to flat indices (BoundColumnRef -> BoundRef)
        ColumnBindingResolver resolver;
        resolver.VisitOperator(plan);
        auto t1 = std::chrono::high_resolution_clock::now();
        RASTERDB_LOG_DEBUG("[TIMER] {:<30s} {:8.2f} ms", "  parse+plan",
                std::chrono::duration<double, std::milli>(t1 - t0).count());

        // Execute on GPU via rasterdf
        auto& gpu_ctx = rasterdb::gpu::gpu_context::instance();
        rasterdb::gpu::gpu_executor executor(gpu_ctx, context);
        data.result_table = executor.execute(plan);
        data.chunk_offset = 0;

        auto t_total_end = std::chrono::high_resolution_clock::now();
        RASTERDB_LOG_INFO("[TIMER] {:<30s} {:8.2f} ms", "TOTAL gpu_execute",
                std::chrono::duration<double, std::milli>(t_total_end - t_total_start).count());

        // Bulk-download all GPU result columns to host memory.
        // Uses the pre-allocated GPUBufferManager staging buffer to avoid
        // per-column VMA alloc/dealloc of huge staging buffers.
        // All workspace pool columns share ONE VkBuffer with different offsets.
        if (data.result_table && data.result_table->num_rows() > 0) {
          auto t_dl_start = std::chrono::high_resolution_clock::now();
          // Zero-copy DMA: batch all columns into download buffer in ONE submit,
          // return pointers directly into HOST_CACHED memory. No heap alloc, no memcpy.
          data.host_column_ptrs = rasterdb::gpu::batch_download_columns(gpu_ctx, *data.result_table);

          data.host_cache_ready = true;
          auto t_dl_end = std::chrono::high_resolution_clock::now();
          RASTERDB_LOG_INFO("[TIMER] {:<30s} {:8.2f} ms", "  bulk_download_to_host",
                  std::chrono::duration<double, std::milli>(t_dl_end - t_dl_start).count());
        }

        RASTERDB_LOG_INFO("RasterDB: query executed on GPU (Vulkan/rasterdf).");
      } catch (duckdb::NotImplementedException& e) {
        // Unsupported operator — fall back to CPU
        RASTERDB_LOG_INFO("RasterDB: GPU fallback to CPU — {}", e.what());
        data.result_table.reset();
      } catch (std::exception& e) {
        // Other GPU error — fall back to CPU
        RASTERDB_LOG_WARN("RasterDB: GPU error, falling back to CPU — {}", e.what());
        data.result_table.reset();
      }
    }

    // CPU fallback if GPU didn't produce a result
    if (!data.result_table) {
      RASTERDB_LOG_INFO("RasterDB: executing query on DuckDB CPU");
      data.cpu_result = data.conn->Query(data.query);
    }
  }

  // Fetch next chunk from whichever result we have
  if (data.result_table) {
    idx_t available = data.result_table->num_rows();
    if (data.chunk_offset >= available) {
      output.SetCardinality(0);
      data.finished = true;
      // Pointers into download buffer — no heap memory to release
      data.host_column_ptrs.clear();
      return;
    }
    
    idx_t chunk_size = duckdb::MinValue<idx_t>(STANDARD_VECTOR_SIZE, available - data.chunk_offset);
    output.SetCardinality(chunk_size);

    for (size_t c = 0; c < data.result_table->num_columns(); c++) {
      auto& col = data.result_table->col(c);
      auto rdf_tid = col.type.id;

      // STRING column: reconstruct DuckDB strings from downloaded offsets+chars
      if (rdf_tid == rasterdf::type_id::STRING) {
        const uint8_t* base = data.host_column_ptrs[c];
        size_t offsets_bytes = (static_cast<size_t>(col.num_rows) + 1) * sizeof(int32_t);
        const auto* offsets = reinterpret_cast<const int32_t*>(base);
        const auto* chars = reinterpret_cast<const char*>(base + offsets_bytes);
        auto str_dst = duckdb::FlatVector::GetData<duckdb::string_t>(output.data[c]);
        for (size_t r = 0; r < chunk_size; r++) {
          idx_t row = data.chunk_offset + r;
          int32_t str_start = offsets[row];
          int32_t str_end = offsets[row + 1];
          int32_t str_len = str_end - str_start;
          str_dst[r] = duckdb::StringVector::AddString(
              output.data[c], chars + str_start, static_cast<uint32_t>(str_len));
        }
        continue;
      }

      size_t elem_size = rasterdb::gpu::rdf_type_size(col.type.id);
      auto dst = duckdb::FlatVector::GetData(output.data[c]);

      // Source pointer: directly from HOST_CACHED download buffer (zero-copy)
      const uint8_t* src = data.host_column_ptrs[c] + data.chunk_offset * elem_size;

      auto duckdb_tid = output.data[c].GetType().id();

      // Determine if a type cast is needed between rdf column and DuckDB output
      bool types_match = false;
      if (rdf_tid == rasterdf::type_id::INT32 && duckdb_tid == duckdb::LogicalTypeId::INTEGER) types_match = true;
      if (rdf_tid == rasterdf::type_id::INT64 && duckdb_tid == duckdb::LogicalTypeId::BIGINT) types_match = true;
      if (rdf_tid == rasterdf::type_id::FLOAT32 && duckdb_tid == duckdb::LogicalTypeId::FLOAT) types_match = true;
      if (rdf_tid == rasterdf::type_id::FLOAT64 && duckdb_tid == duckdb::LogicalTypeId::DOUBLE) types_match = true;

      if (types_match) {
        // Direct memcpy from host cache — no GPU transfer needed
        std::memcpy(dst, src, chunk_size * elem_size);
      } else {
        // Type conversion from host cache
        if (duckdb_tid == duckdb::LogicalTypeId::HUGEINT) {
          if (rdf_tid == rasterdf::type_id::INT64) {
            for (size_t r = 0; r < chunk_size; r++) {
                int64_t val;
                std::memcpy(&val, src + r * sizeof(int64_t), sizeof(int64_t));
                duckdb::hugeint_t hval;
                hval.lower = static_cast<uint64_t>(val);
                hval.upper = val < 0 ? -1 : 0;
                std::memcpy(dst + r * sizeof(duckdb::hugeint_t), &hval, sizeof(duckdb::hugeint_t));
            }
          } else {
            for (size_t r = 0; r < chunk_size; r++) {
                int32_t val;
                std::memcpy(&val, src + r * sizeof(int32_t), sizeof(int32_t));
                duckdb::hugeint_t hval;
                hval.lower = static_cast<uint64_t>(val < 0 ? -static_cast<int64_t>(-val) : val);
                hval.upper = val < 0 ? -1 : 0;
                std::memcpy(dst + r * sizeof(duckdb::hugeint_t), &hval, sizeof(duckdb::hugeint_t));
            }
          }
        } else if (duckdb_tid == duckdb::LogicalTypeId::DOUBLE) {
          if (rdf_tid == rasterdf::type_id::INT64) {
            for (size_t r = 0; r < chunk_size; r++) {
                int64_t val;
                std::memcpy(&val, src + r * sizeof(int64_t), sizeof(int64_t));
                double dval = static_cast<double>(val);
                std::memcpy(dst + r * sizeof(double), &dval, sizeof(double));
            }
          } else if (rdf_tid == rasterdf::type_id::FLOAT32) {
            for (size_t r = 0; r < chunk_size; r++) {
                float val;
                std::memcpy(&val, src + r * sizeof(float), sizeof(float));
                double dval = static_cast<double>(val);
                std::memcpy(dst + r * sizeof(double), &dval, sizeof(double));
            }
          } else if (rdf_tid == rasterdf::type_id::INT32) {
            for (size_t r = 0; r < chunk_size; r++) {
                int32_t val;
                std::memcpy(&val, src + r * sizeof(int32_t), sizeof(int32_t));
                double dval = static_cast<double>(val);
                std::memcpy(dst + r * sizeof(double), &dval, sizeof(double));
            }
          } else {
            std::memcpy(dst, src, chunk_size * elem_size);
          }
        } else if (duckdb_tid == duckdb::LogicalTypeId::BIGINT) {
          if (rdf_tid == rasterdf::type_id::INT32) {
            for (size_t r = 0; r < chunk_size; r++) {
                int32_t val;
                std::memcpy(&val, src + r * sizeof(int32_t), sizeof(int32_t));
                int64_t wide = static_cast<int64_t>(val);
                std::memcpy(dst + r * sizeof(int64_t), &wide, sizeof(int64_t));
            }
          } else {
            std::memcpy(dst, src, chunk_size * elem_size);
          }
        } else {
          std::memcpy(dst, src, chunk_size * elem_size);
        }
      }
    }
    data.chunk_offset += chunk_size;
    return;
  } else if (data.cpu_result) {
    auto chunk = data.cpu_result->Fetch();
    if (!chunk || chunk->size() == 0) {
      output.SetCardinality(0);
      data.finished = true;
      return;
    }
    output.Reference(*chunk);
    return;
  }

  output.SetCardinality(0);
  data.finished = true;
}

// =========================================================================
// Config setters — minimal set for the extension skeleton
// =========================================================================

static void SetEnableDuckdbFallback(ClientContext&, SetScope, Value& parameter)
{
  Config::ENABLE_DUCKDB_FALLBACK = BooleanValue::Get(parameter);
}

// =========================================================================
// Extension registration
// =========================================================================

void RasterdbExtension::RegisterGPUFunctions(DatabaseInstance& instance)
{
  auto transaction = CatalogTransaction::GetSystemTransaction(instance);
  auto& catalog    = Catalog::GetSystemCatalog(instance);

  // gpu_execution(query) — the main entry point
  TableFunction gpu_execution("gpu_execution",
                              {LogicalType::VARCHAR},
                              GPUExecutionFunction,
                              GPUExecutionBind);
  CreateTableFunctionInfo gpu_execution_info(gpu_execution);
  catalog.CreateTableFunction(transaction, gpu_execution_info);

  // gpu_buffer_init(cache_size, processing_size) — pre-allocate GPU buffers
  TableFunction gpu_buffer_init("gpu_buffer_init",
                                {LogicalType::VARCHAR, LogicalType::VARCHAR},
                                GPUBufferInitFunction,
                                GPUBufferInitBind);
  CreateTableFunctionInfo gpu_buffer_init_info(gpu_buffer_init);
  catalog.CreateTableFunction(transaction, gpu_buffer_init_info);
}

void RasterdbExtension::InitialGPUConfigs(DBConfig& config)
{
  config.AddExtensionOption(
    "enable_duckdb_fallback",
    "Whether to fall back to DuckDB CPU execution on GPU error",
    LogicalType(LogicalTypeId::BOOLEAN),
    Value::BOOLEAN(Config::ENABLE_DUCKDB_FALLBACK),
    SetEnableDuckdbFallback);
}

static void LoadInternal(ExtensionLoader& loader)
{
  duckdb::InitGPULogger();
  rasterdb::util::install_segfault_backtrace_handler();

  auto& db     = loader.GetDatabaseInstance();
  auto& config = DBConfig::GetConfig(db);
  RasterdbExtension::InitialGPUConfigs(config);
  RasterdbExtension::RegisterGPUFunctions(db);

  // Initialize GPU context (Vulkan + rasterdf)
  try {
    rasterdb::gpu::gpu_context::initialize();
    // Ensure GPU context is destroyed before static destructors run
    std::atexit([]() { rasterdb::gpu::gpu_context::shutdown(); });

    // Auto-initialize BufferManager with 2GB defaults
    auto& bufMgr = rasterdb::gpu::GPUBufferManager::GetInstance(
        2048ULL * 1024 * 1024 * 2, 
        2048ULL * 1024 * 1024 * 2, 
        2048ULL * 1024 * 1024 * 2);
    (void)bufMgr;

    RasterdbExtension::buffer_is_initialized = true;
    RASTERDB_LOG_INFO("RasterDB extension loaded with Vulkan GPU support");
  } catch (std::exception& e) {
    RASTERDB_LOG_WARN("RasterDB: GPU init failed ({}), CPU-only mode", e.what());
    RasterdbExtension::buffer_is_initialized = false;
  }
}

void RasterdbExtension::Load(ExtensionLoader& loader) { LoadInternal(loader); }

std::string RasterdbExtension::Name() { return "rasterdb"; }

std::string RasterdbExtension::Version() const
{
#ifdef EXT_VERSION_RASTERDB
  return EXT_VERSION_RASTERDB;
#else
  return "0.1.0-dev";
#endif
}

}  // namespace duckdb

extern "C" {
DUCKDB_CPP_EXTENSION_ENTRY(rasterdb, loader) { duckdb::LoadInternal(loader); }
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
