/*
 * Copyright 2025, RasterDB Contributors.
 * Split from src/gpu/gpu_executor.cpp.
 */

#include "gpu/gpu_executor_internal.hpp"

namespace rasterdb {
namespace gpu {

std::unique_ptr<gpu_table> gpu_executor::execute_get(duckdb::LogicalGet& op)
{
  RASTERDB_LOG_DEBUG("GPU execute_get: {}", op.function.name);
  if (debug_logging_enabled()) {
    auto& cids = op.GetColumnIds();
    auto bindings = op.GetColumnBindings();

    RASTERDB_LOG_DEBUG("[RDB_DEBUG] GET '{}': table_filters={}",
                       op.function.name, op.table_filters.filters.size());

    std::ostringstream col_ids_line;
    col_ids_line << "[RDB_DEBUG]   column_ids(" << cids.size() << "):";
    for (auto& c : cids) {
      col_ids_line << ' ' << static_cast<size_t>(c.GetPrimaryIndex());
    }
    RASTERDB_LOG_DEBUG("{}", col_ids_line.str());

    std::ostringstream projections_line;
    projections_line << "[RDB_DEBUG]   projection_ids(" << op.projection_ids.size() << "):";
    for (auto& p : op.projection_ids) {
      projections_line << ' ' << static_cast<size_t>(p);
    }
    RASTERDB_LOG_DEBUG("{}", projections_line.str());

    std::ostringstream names_line;
    names_line << "[RDB_DEBUG]   names(" << op.names.size() << "):";
    for (size_t i = 0; i < op.names.size() && i < 20; i++) {
      names_line << ' ' << op.names[i];
    }
    RASTERDB_LOG_DEBUG("{}", names_line.str());

    std::ostringstream bindings_line;
    bindings_line << "[RDB_DEBUG]   bindings(" << bindings.size() << "):";
    for (auto& b : bindings) {
      bindings_line << " (" << static_cast<size_t>(b.table_index)
                    << "," << static_cast<size_t>(b.column_index) << ")";
    }
    RASTERDB_LOG_DEBUG("{}", bindings_line.str());
  }

  // Get the output types from the logical operator.
  // In unoptimized plans, op.types may be empty. Use returned_types + column_ids
  // to determine the actual projected types.
  auto& col_ids = op.GetColumnIds();
  duckdb::vector<duckdb::LogicalType> types;
  for (auto& cid : col_ids) {
    auto idx = cid.GetPrimaryIndex();
    if (idx < op.returned_types.size()) {
      types.push_back(op.returned_types[idx]);
    }
  }
  if (types.empty()) {
    types = op.returned_types; // fallback: all columns
  }

  // Validate all types are GPU-compatible (throws for strings etc.)
  for (auto& t : types) {
    to_rdf_type(t);
  }

  // Get table name for logging / cache lookup
  std::string table_name;
  auto table_entry = op.GetTable();
  if (table_entry) {
    table_name = table_entry->name;
  }
  if (table_name.empty()) {
    table_name = op.function.name;
  }

  // count(*) optimization: only scan 1 column, we just need the row count
  if (_scan_count_star_only && !types.empty()) {
    types.resize(1);
  }

  // ── Direct Table Function Scan (mirrors Sirius GetDataDuckDB) ────────
  // Build scan types from column_ids
  duckdb::vector<duckdb::LogicalType> scan_types;
  if (_scan_count_star_only && !col_ids.empty()) {
    auto idx = col_ids[0].GetPrimaryIndex();
    if (idx < op.returned_types.size()) {
      scan_types.push_back(op.returned_types[idx]);
    }
  } else {
    for (auto& cid : col_ids) {
      auto idx = cid.GetPrimaryIndex();
      if (idx < op.returned_types.size()) {
        scan_types.push_back(op.returned_types[idx]);
      }
    }
  }
  if (scan_types.empty()) {
    scan_types = op.returned_types;
  }

  // Build column name list for cache lookup
  std::vector<std::string> col_names;
  if (_scan_count_star_only && !col_ids.empty()) {
    auto col_idx = col_ids[0].GetPrimaryIndex();
    col_names.push_back(col_idx < op.names.size() ? op.names[col_idx] : op.names[0]);
  } else {
    for (size_t i = 0; i < col_ids.size(); i++) {
      auto col_idx = col_ids[i].GetPrimaryIndex();
      col_names.push_back(col_idx < op.names.size() ? op.names[col_idx] : op.names[0]);
    }
  }

  // ── Pipelined Scan + Direct Staging Write (mirrors Sirius) ──────────
  // Scan chunks from DuckDB and flatten directly into reBAR staging buffer
  // in a single pass, eliminating the separate upload step.
  rasterdf::size_type total_scanned = 0;
  std::unique_ptr<gpu_table> gpu_tbl;

  bool use_buffer_manager = GPUBufferManager::is_initialized();

  if (use_buffer_manager) {
    // ── FAST PATH: Pipelined scan with direct reBAR staging write ──
    // Scan chunks from DuckDB and flatten each chunk directly into pre-allocated
    // staging memory as it arrives. No separate upload step needed.
    // GPU reads directly from reBAR staging memory (host-visible device-local).
    auto& bufMgr = GPUBufferManager::GetInstance();
    // NOTE: cpuProcessingPointer is reset per-query in execute(), NOT here.
    // Resetting per-scan would overwrite previous tables' staging data.

    size_t num_cols = scan_types.size();
    std::vector<rasterdf::data_type> rdf_types(num_cols);
    for (size_t c = 0; c < num_cols; c++) {
      rdf_types[c] = to_rdf_type(scan_types[c]);
    }

    // Check cache hits first
    gpu_tbl = std::make_unique<gpu_table>();
    gpu_tbl->duckdb_types = types;
    gpu_tbl->columns.resize(num_cols);

    bool all_cached = true;
    for (size_t c = 0; c < num_cols; c++) {
      if (bufMgr.checkIfColumnCached(table_name, col_names[c])) {
        auto* cached = bufMgr.getCachedColumn(table_name, col_names[c]);
        gpu_tbl->columns[c].type = cached->type;
        gpu_tbl->columns[c].num_rows = static_cast<rasterdf::size_type>(cached->num_rows);
        gpu_tbl->columns[c].cached_address = bufMgr.gpuCacheAddress() + cached->gpu_offset;
        gpu_tbl->columns[c].cached_buffer = bufMgr.gpuCacheBuffer();
      } else {
        all_cached = false;
      }
    }

    if (all_cached) {
      // All columns cached — skip scan entirely
      total_scanned = gpu_tbl->columns[0].num_rows;
      RASTERDB_LOG_DEBUG("[TIMER]   cpu_scan                         0.00 ms (all cached)");
      RASTERDB_LOG_DEBUG("[TIMER]   scan: {} {} rows x {} cols",
                         table_name, total_scanned, types.size());
      RASTERDB_LOG_DEBUG("[TIMER]   gpu_upload                        0.00 ms (cached)");
    } else {
      // Pre-allocate staging buffers for each uncached column (max estimate)
      // We'll use the scan to determine actual row count, then set metadata
      struct col_staging_info {
        size_t staging_off;
        uint8_t* staging_dst;
        size_t write_pos;
      };
      std::vector<col_staging_info> staging(num_cols);

      // Estimate staging size: use _scan_limit, estimated_cardinality, or fallback
      size_t STAGING_CHUNK_ROWS;
      if (_scan_limit > 0) {
        STAGING_CHUNK_ROWS = static_cast<size_t>(_scan_limit);
      } else if (op.has_estimated_cardinality && op.estimated_cardinality > 0) {
        // Use plan estimate with 20% headroom
        STAGING_CHUNK_ROWS = static_cast<size_t>(op.estimated_cardinality * 1.2) + 1024;
      } else {
        STAGING_CHUNK_ROWS = 20000000; // 20M rows fallback
      }
      for (size_t c = 0; c < num_cols; c++) {
        if (!bufMgr.checkIfColumnCached(table_name, col_names[c])) {
          size_t col_bytes = STAGING_CHUNK_ROWS * rdf_type_size(rdf_types[c].id);
          staging[c].staging_dst = bufMgr.customVkHostAlloc<uint8_t>(col_bytes);
          staging[c].staging_off = static_cast<size_t>(staging[c].staging_dst - bufMgr.cpuProcessing);
          staging[c].write_pos = 0;
        }
      }

      {
        stage_timer t_scan("  cpu_scan");

        // Create proper ExecutionContext for init_local (like Sirius)
        duckdb::ThreadContext thread_ctx(_client_ctx);
        duckdb::ExecutionContext exec_ctx(_client_ctx, thread_ctx, nullptr);

        duckdb::TableFunctionInitInput init_input(
          op.bind_data.get(), col_ids, op.projection_ids,
          nullptr /* no filters — we handle them on GPU */,
          op.extra_info.sample_options);

        auto global_state = op.function.init_global(_client_ctx, init_input);

        duckdb::unique_ptr<duckdb::LocalTableFunctionState> local_state;
        if (op.function.init_local) {
          local_state = op.function.init_local(exec_ctx, init_input, global_state.get());
        }

        duckdb::TableFunctionInput tf_input(
          op.bind_data.get(), local_state.get(), global_state.get());

        // ── Pipelined scan loop: read chunk, flatten directly into staging ──
        // Respect _scan_limit: stop scanning once we have enough rows
        rasterdf::size_type scan_row_limit = (_scan_limit > 0)
          ? static_cast<rasterdf::size_type>(_scan_limit) : std::numeric_limits<rasterdf::size_type>::max();

        while (total_scanned < scan_row_limit) {
          auto chunk = duckdb::make_uniq<duckdb::DataChunk>();
          chunk->Initialize(duckdb::Allocator::DefaultAllocator(), scan_types);
          op.function.function(_client_ctx, tf_input, *chunk);
          if (chunk->size() == 0) break;
          chunk->Flatten();

          auto chunk_rows = static_cast<rasterdf::size_type>(chunk->size());
          // Clamp to limit
          if (total_scanned + chunk_rows > scan_row_limit) {
            chunk_rows = scan_row_limit - total_scanned;
          }
          total_scanned += chunk_rows;

          // Flatten each column directly into staging (inline with scan)
          for (size_t c = 0; c < num_cols; c++) {
            if (staging[c].staging_dst) {
              size_t elem_size = rdf_type_size(rdf_types[c].id);
              size_t bytes = static_cast<size_t>(chunk_rows) * elem_size;
              auto data_ptr = reinterpret_cast<const uint8_t*>(chunk->data[c].GetData());
              std::memcpy(staging[c].staging_dst + staging[c].write_pos, data_ptr, bytes);
              staging[c].write_pos += bytes;
            }
          }
        }
      }
      RASTERDB_LOG_DEBUG("[TIMER]   scan: {} {} rows x {} cols",
                         table_name, total_scanned, types.size());

      // Set column metadata (staging addresses for GPU to read via reBAR zero-copy)
      for (size_t c = 0; c < num_cols; c++) {
        if (staging[c].staging_dst) {
          gpu_tbl->columns[c].type = rdf_types[c];
          gpu_tbl->columns[c].num_rows = total_scanned;
          gpu_tbl->columns[c].cached_address = bufMgr.cpuStagingAddress() + staging[c].staging_off;
          gpu_tbl->columns[c].cached_buffer = bufMgr.cpuStagingBuffer();
        }
      }
      gpu_tbl->set_num_rows(total_scanned);

      size_t total_bytes = 0;
      for (size_t c = 0; c < num_cols; c++) {
        total_bytes += gpu_tbl->col(c).byte_size();
      }
      RASTERDB_LOG_DEBUG("[RDB_DEBUG]     gpu_upload_detail: {} cols, {} bytes (zero-copy reBAR)",
                         gpu_tbl->num_columns(), total_bytes);
      RASTERDB_LOG_DEBUG("[TIMER]   gpu_upload                        0.00 ms (zero-copy)");
    }
  } else {
    // ── FALLBACK: Standard scan + device upload ──
    std::vector<std::unique_ptr<duckdb::DataChunk>> chunks;
    {
      stage_timer t_scan("  cpu_scan");

      duckdb::ThreadContext thread_ctx(_client_ctx);
      duckdb::ExecutionContext exec_ctx(_client_ctx, thread_ctx, nullptr);

      duckdb::TableFunctionInitInput init_input(
        op.bind_data.get(), col_ids, op.projection_ids,
        nullptr, op.extra_info.sample_options);

      auto global_state = op.function.init_global(_client_ctx, init_input);

      duckdb::unique_ptr<duckdb::LocalTableFunctionState> local_state;
      if (op.function.init_local) {
        local_state = op.function.init_local(exec_ctx, init_input, global_state.get());
      }

      duckdb::TableFunctionInput tf_input(
        op.bind_data.get(), local_state.get(), global_state.get());

      // Respect _scan_limit: stop scanning once we have enough rows
      rasterdf::size_type scan_row_limit = (_scan_limit > 0)
        ? static_cast<rasterdf::size_type>(_scan_limit) : std::numeric_limits<rasterdf::size_type>::max();

      while (total_scanned < scan_row_limit) {
        auto chunk = duckdb::make_uniq<duckdb::DataChunk>();
        chunk->Initialize(duckdb::Allocator::DefaultAllocator(), scan_types);
        op.function.function(_client_ctx, tf_input, *chunk);
        if (chunk->size() == 0) break;
        chunk->Flatten();
        auto chunk_rows = static_cast<rasterdf::size_type>(chunk->size());
        if (total_scanned + chunk_rows > scan_row_limit) {
          chunk_rows = scan_row_limit - total_scanned;
        }
        total_scanned += chunk_rows;
        chunks.push_back(std::move(chunk));
      }
    }
    RASTERDB_LOG_DEBUG("[TIMER]   scan: {} {} rows x {} cols",
                       table_name, total_scanned, types.size());

    // Debug: dump first 20 values from each column in CPU staging
    if (debug_logging_enabled() && total_scanned > 0 && !chunks.empty()) {
      size_t sample = std::min((size_t)20, (size_t)total_scanned);
      RASTERDB_LOG_DEBUG("[RDB_DEBUG] SCAN '{}' CPU staging first {}:", table_name, sample);
      for (size_t col_idx = 0; col_idx < types.size(); col_idx++) {
        std::ostringstream line;
        line << "[RDB_DEBUG]   col[" << col_idx << "] type="
             << types[col_idx].ToString() << ":";
        size_t row_idx = 0;
        for (auto& chunk : chunks) {
          if (row_idx >= sample) break;
          auto& vec = chunk->data[col_idx];
          auto chunk_rows = std::min((size_t)chunk->size(), sample - row_idx);
          if (types[col_idx] == duckdb::LogicalType::FLOAT) {
            auto floats = duckdb::FlatVector::GetData<float>(vec);
            for (size_t i = 0; i < chunk_rows; i++) {
              line << ' ' << floats[i];
            }
          } else if (types[col_idx] == duckdb::LogicalType::INTEGER) {
            auto ints = duckdb::FlatVector::GetData<int32_t>(vec);
            for (size_t i = 0; i < chunk_rows; i++) {
              line << ' ' << ints[i];
            }
          } else {
            line << " ?";
          }
          row_idx += chunk_rows;
          if (row_idx >= sample) break;
        }
        RASTERDB_LOG_DEBUG("{}", line.str());
      }
    }

    {
      stage_timer t_upload("  gpu_upload");
      gpu_tbl = gpu_table::from_data_chunks(_ctx, types, chunks);
      size_t total_bytes = 0;
      for (size_t c = 0; c < gpu_tbl->num_columns(); c++) {
        total_bytes += gpu_tbl->col(c).byte_size();
      }
      RASTERDB_LOG_DEBUG("[RDB_DEBUG]     gpu_upload_detail: {} cols, {} bytes (device copy)",
                         gpu_tbl->num_columns(), total_bytes);
    }
  }

  return gpu_tbl;
}

} // namespace gpu
} // namespace rasterdb
