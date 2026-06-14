/*
 * Copyright 2026, RasterDB Contributors.
 * Split from src/gpu/gpu_executor.cpp.
 */

#include "gpu/gpu_executor_internal.hpp"

#include <duckdb/planner/table_filter.hpp>
#include <duckdb/storage/statistics/numeric_stats.hpp>
#include <duckdb/storage/table_storage_info.hpp>

#include <future>
#include <limits>
#include <mutex>
#include <thread>

namespace rasterdb {
namespace gpu {

// Fast path: use multiple DuckDB local scan states and coalesce numeric columns
// directly into RasterDB's staging buffers. Set false to force the earlier
// single-thread scan path for operator-only comparisons against Sirius/cuDF.
static constexpr bool USE_RDB_PARALLEL_SCAN = true;
// Set to 1, 2, 3, ... to cap scan workers. 
// Leave at max() to use all available scan threads.
static constexpr size_t USE_RDB_PARALLEL_SCAN_THREADS = std::numeric_limits<size_t>::max();

static duckdb::unique_ptr<duckdb::TableFilterSet>
create_scan_filter_set(const duckdb::TableFilterSet& table_filters,
                       const duckdb::vector<duckdb::ColumnIndex>& column_ids)
{
  if (table_filters.filters.empty()) {
    return nullptr;
  }
  auto filter_set = duckdb::make_uniq<duckdb::TableFilterSet>();
  for (auto& entry : table_filters.filters) {
    duckdb::optional_idx column_index;
    for (duckdb::idx_t i = 0; i < column_ids.size(); i++) {
      if (entry.first == column_ids[i].GetPrimaryIndex()) {
        column_index = i;
        break;
      }
    }
    if (!column_index.IsValid()) {
      throw duckdb::InternalException("RasterDB GPU scan: could not remap table filter column index");
    }
    filter_set->filters[column_index.GetIndex()] = entry.second->Copy();
  }
  return filter_set;
}

static void attach_scan_i32_minmax(gpu_table& table,
                                   duckdb::optional_ptr<duckdb::TableCatalogEntry> table_entry,
                                   duckdb::ClientContext& client_ctx,
                                   const duckdb::vector<duckdb::ColumnIndex>& output_col_ids,
                                   const duckdb::vector<duckdb::LogicalType>& scan_types)
{
  if (!table_entry) {
    return;
  }
  auto count = std::min(table.num_columns(), output_col_ids.size());
  for (size_t c = 0; c < count && c < scan_types.size(); c++) {
    auto& col = table.columns[c];
    if (col.type.id != rasterdf::type_id::INT32) {
      continue;
    }
    try {
      auto stats = table_entry->GetStatistics(client_ctx, output_col_ids[c].GetPrimaryIndex());
      if (!stats || !duckdb::NumericStats::HasMinMax(*stats)) {
        continue;
      }
      auto min_value = duckdb::NumericStats::Min(*stats).DefaultCastAs(scan_types[c]);
      auto max_value = duckdb::NumericStats::Max(*stats).DefaultCastAs(scan_types[c]);
      col.i32_min = min_value.GetValueUnsafe<int32_t>();
      col.i32_max = max_value.GetValueUnsafe<int32_t>();
      col.has_i32_minmax = true;
    } catch (...) {
      col.has_i32_minmax = false;
    }
  }
}

std::unique_ptr<gpu_table> gpu_executor::execute_get(duckdb::LogicalGet& op)
{
  RASTERDB_LOG_DEBUG("GPU execute_get: {}", op.function.name);
  if (debug_logging_enabled()) {
    auto& cids    = op.GetColumnIds();
    auto bindings = op.GetColumnBindings();

    RASTERDB_LOG_DEBUG(
      "[RDB_DEBUG] GET '{}': table_filters={}", op.function.name, op.table_filters.filters.size());

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
      bindings_line << " (" << static_cast<size_t>(b.table_index) << ","
                    << static_cast<size_t>(b.column_index) << ")";
    }
    RASTERDB_LOG_DEBUG("{}", bindings_line.str());
  }

  auto& col_ids = op.GetColumnIds();
  duckdb::vector<duckdb::ColumnIndex> output_col_ids;
  if (_scan_count_star_only && !col_ids.empty()) {
    output_col_ids.push_back(col_ids[0]);
  } else if (!op.projection_ids.empty()) {
    for (auto proj_id : op.projection_ids) {
      if (proj_id >= col_ids.size()) {
        throw duckdb::InternalException(
          "RasterDB GPU scan: projection id %llu out of range for %llu column ids",
          static_cast<unsigned long long>(proj_id),
          static_cast<unsigned long long>(col_ids.size()));
      }
      output_col_ids.push_back(col_ids[proj_id]);
    }
  } else {
    output_col_ids = col_ids;
  }

  // Get the output types from the logical operator.
  // Optimized plans may scan more columns than they project. Downstream bound
  // references index the projected output schema, so all GPU table metadata must
  // follow output_col_ids/op.types rather than raw col_ids.
  duckdb::vector<duckdb::LogicalType> types;
  if (!op.types.empty() && op.types.size() == output_col_ids.size()) {
    types = op.types;
  } else {
  for (auto& cid : output_col_ids) {
    auto idx = cid.GetPrimaryIndex();
    if (idx < op.returned_types.size()) { types.push_back(op.returned_types[idx]); }
  }
  }
  if (types.empty()) {
    types = op.returned_types;  // fallback: all columns
  }

  // Validate all types are GPU-compatible (throws for strings etc.)
  for (auto& t : types) {
    to_rdf_type(t);
  }

  // Get table name for logging / cache lookup
  std::string table_name;
  auto table_entry = op.GetTable();
  if (table_entry) { table_name = table_entry->name; }
  if (table_name.empty()) { table_name = op.function.name; }

  // count(*) optimization: only scan 1 column, we just need the row count
  if (_scan_count_star_only && !types.empty()) { types.resize(1); }

  // ── Direct Table Function Scan (mirrors Sirius GetDataDuckDB) ────────
  // Build physical output scan types. TableFunctionInitInput still receives
  // raw col_ids + projection_ids so DuckDB can perform pushdown internally, but
  // chunks contain only projected output columns when projection_ids is set.
  duckdb::vector<duckdb::LogicalType> scan_types;
  if (!op.types.empty() && op.types.size() == output_col_ids.size()) {
    scan_types = op.types;
  } else {
    for (auto& cid : output_col_ids) {
      auto idx = cid.GetPrimaryIndex();
      if (idx < op.returned_types.size()) { scan_types.push_back(op.returned_types[idx]); }
    }
  }
  if (scan_types.empty()) { scan_types = op.returned_types; }

  // Build column name list for cache lookup
  std::vector<std::string> col_names;
  if (!op.names.empty()) {
    for (auto& cid : output_col_ids) {
      auto col_idx = cid.GetPrimaryIndex();
      col_names.push_back(col_idx < op.names.size() ? op.names[col_idx] : op.names[0]);
    }
  }

  // ── Pipelined Scan + Direct Staging Write (mirrors Sirius) ──────────
  // Scan chunks from DuckDB and flatten directly into reBAR staging buffer
  // in a single pass, eliminating the separate upload step.
  rasterdf::size_type total_scanned = 0;
  std::unique_ptr<gpu_table> gpu_tbl;

  bool use_buffer_manager = GPUBufferManager::is_initialized();

  // STRING columns cannot use the buffer manager zero-copy path (variable-width);
  // fall back to from_data_chunks which handles string flattening properly.
  if (use_buffer_manager) {
    for (auto& t : scan_types) {
      if (to_rdf_type(t).id == rasterdf::type_id::STRING) {
        use_buffer_manager = false;
        break;
      }
    }
  }

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
    gpu_tbl               = std::make_unique<gpu_table>();
    gpu_tbl->duckdb_types = scan_types;
    gpu_tbl->columns.resize(num_cols);

    bool all_cached = true;
    bool any_cached = false;
    for (size_t c = 0; c < num_cols; c++) {
      if (bufMgr.checkIfColumnCached(table_name, col_names[c])) {
        any_cached = true;
        auto* cached                       = bufMgr.getCachedColumn(table_name, col_names[c]);
        gpu_tbl->columns[c].type           = cached->type;
        gpu_tbl->columns[c].num_rows       = static_cast<rasterdf::size_type>(cached->num_rows);
        gpu_tbl->columns[c].cached_address = bufMgr.gpuCacheAddress() + cached->gpu_offset;
        gpu_tbl->columns[c].cached_buffer  = bufMgr.gpuCacheBuffer();
        gpu_tbl->columns[c].cached_offset  = cached->gpu_offset;
      } else {
        all_cached = false;
      }
    }

    if (all_cached) {
      // All columns cached — skip scan entirely
      total_scanned = gpu_tbl->columns[0].num_rows;
      RASTERDB_LOG_DEBUG("[TIMER]   cpu_scan                         0.00 ms (all cached)");
      RASTERDB_LOG_DEBUG(
        "[TIMER]   scan: {} {} rows x {} cols", table_name, total_scanned, types.size());
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

      // Size staging from actual base-table cardinality when available. Optimized
      // LogicalGet estimates can reflect filtered/joined cardinality, but this
      // table function scan still materializes the base scan output.
      size_t STAGING_CHUNK_ROWS;
      size_t table_rows = 0;
      if (table_entry) {
        try {
          auto storage_info = table_entry->GetStorageInfo(_client_ctx);
          if (storage_info.cardinality.IsValid()) {
            table_rows = storage_info.cardinality.GetIndex();
          }
        } catch (...) {
        }
      }
      if (_scan_limit > 0) {
        STAGING_CHUNK_ROWS = static_cast<size_t>(_scan_limit);
      } else if (table_rows > 0) {
        STAGING_CHUNK_ROWS = static_cast<size_t>(table_rows * 1.05) + 4096;
        RASTERDB_LOG_DEBUG("[RDB_DEBUG]   staging sized from table stats: {} rows",
                           STAGING_CHUNK_ROWS);
      } else if (op.has_estimated_cardinality && op.estimated_cardinality > 0) {
        STAGING_CHUNK_ROWS = static_cast<size_t>(op.estimated_cardinality * 2.0) + 4096;
      } else {
        STAGING_CHUNK_ROWS = 200000000;  // 200M rows fallback (safe up to ~SF30)
        RASTERDB_LOG_WARN(
          "[RDB_DEBUG]   using 200M row staging fallback — no table stats available");
      }
      for (size_t c = 0; c < num_cols; c++) {
        if (!bufMgr.checkIfColumnCached(table_name, col_names[c])) {
          size_t col_bytes       = STAGING_CHUNK_ROWS * rdf_type_size(rdf_types[c].id);
          staging[c].staging_dst = bufMgr.customVkHostAlloc<uint8_t>(col_bytes);
          staging[c].staging_off =
            static_cast<size_t>(staging[c].staging_dst - bufMgr.cpuProcessing);
          staging[c].write_pos = 0;
        }
      }

      {
        stage_timer t_scan("  cpu_scan");

        auto table_filters = create_scan_filter_set(op.table_filters, col_ids);
        duckdb::TableFunctionInitInput init_input(op.bind_data.get(),
                                                  col_ids,
                                                  op.projection_ids,
                                                  table_filters.get(),
                                                  op.extra_info.sample_options);

        auto global_state = op.function.init_global(_client_ctx, init_input);
        duckdb::idx_t max_threads = 1;
        if (global_state) {
          try {
            max_threads = std::max<duckdb::idx_t>(1, global_state->MaxThreads());
          } catch (...) {
            max_threads = 1;
          }
        }
        auto table_max_threads = static_cast<size_t>(std::max<duckdb::idx_t>(1, max_threads));
        auto hw_threads = static_cast<size_t>(
          std::max<unsigned>(1, std::thread::hardware_concurrency()));
        auto max_scan_threads = std::max<size_t>(1, std::min(table_max_threads, hw_threads));
        auto requested_scan_threads =
          std::max<size_t>(1, USE_RDB_PARALLEL_SCAN_THREADS);
        auto num_scan_threads = std::min(requested_scan_threads, max_scan_threads);
        if (!USE_RDB_PARALLEL_SCAN ||
            std::getenv("RASTERDB_DISABLE_PARALLEL_SCAN") != nullptr) {
          num_scan_threads = 1;
        }
        if (any_cached) {
          // Parallel scans append chunks in completion order. That is fine when
          // every projected column is scanned together, but partial-cache scans
          // would no longer align newly scanned columns with cached columns.
          num_scan_threads = 1;
        }

        rasterdf::size_type scan_row_limit = (_scan_limit > 0)
                                               ? static_cast<rasterdf::size_type>(_scan_limit)
                                               : std::numeric_limits<rasterdf::size_type>::max();
        std::atomic<size_t> write_rows{0};
        std::atomic<bool> staging_overflow{false};
        std::exception_ptr first_error;
        std::mutex error_mutex;

        auto scan_worker = [&](size_t /*worker_id*/) {
          try {
            duckdb::ThreadContext thread_ctx(_client_ctx);
            duckdb::ExecutionContext exec_ctx(_client_ctx, thread_ctx, nullptr);
            duckdb::unique_ptr<duckdb::LocalTableFunctionState> local_state;
            if (op.function.init_local) {
              local_state = op.function.init_local(exec_ctx, init_input, global_state.get());
            }

            duckdb::TableFunctionInput tf_input(
              op.bind_data.get(), local_state.get(), global_state.get());
            duckdb::DataChunk chunk;
            chunk.Initialize(duckdb::Allocator::DefaultAllocator(), scan_types);

            while (true) {
              chunk.Reset();
              op.function.function(_client_ctx, tf_input, chunk);
              if (chunk.size() == 0) {
                break;
              }
              chunk.Flatten();

              auto chunk_rows = static_cast<size_t>(chunk.size());
              auto base_row = write_rows.fetch_add(chunk_rows, std::memory_order_relaxed);
              if (base_row >= static_cast<size_t>(scan_row_limit) ||
                  base_row >= STAGING_CHUNK_ROWS) {
                break;
              }
              auto writable_rows = std::min(chunk_rows,
                                            static_cast<size_t>(scan_row_limit) - base_row);
              if (base_row + writable_rows > STAGING_CHUNK_ROWS) {
                writable_rows = STAGING_CHUNK_ROWS - base_row;
                staging_overflow.store(true, std::memory_order_relaxed);
              }
              if (writable_rows == 0) {
                break;
              }

              for (size_t c = 0; c < num_cols; c++) {
                if (staging[c].staging_dst) {
                  auto elem_size = rdf_type_size(rdf_types[c].id);
                  copy_duckdb_vector_to_rdf(chunk.data[c],
                                            writable_rows,
                                            scan_types[c],
                                            rdf_types[c],
                                            staging[c].staging_dst + base_row * elem_size);
                }
              }
            }
          } catch (...) {
            std::lock_guard<std::mutex> guard(error_mutex);
            if (!first_error) {
              first_error = std::current_exception();
            }
          }
        };

        if (num_scan_threads <= 1) {
          scan_worker(0);
        } else {
          std::vector<std::future<void>> futures;
          futures.reserve(num_scan_threads);
          for (size_t t = 0; t < num_scan_threads; t++) {
            futures.push_back(std::async(std::launch::async, scan_worker, t));
          }
          for (auto& f : futures) {
            f.get();
          }
        }
        if (first_error) {
          std::rethrow_exception(first_error);
        }
        total_scanned = static_cast<rasterdf::size_type>(std::min(
            {write_rows.load(std::memory_order_relaxed), static_cast<size_t>(scan_row_limit), STAGING_CHUNK_ROWS}));
            
        if (staging_overflow.load(std::memory_order_relaxed)) {
          RASTERDB_LOG_WARN("Staging buffer overflow during parallel scan; result truncated to {} rows",
                            total_scanned);
        }
        RASTERDB_LOG_DEBUG(
          "[RDB_DEBUG]   scan_threads={} requested={} max={} table_max={} hw={} parallel={}",
          num_scan_threads,
          requested_scan_threads,
          max_scan_threads,
          table_max_threads,
          hw_threads,
          USE_RDB_PARALLEL_SCAN);
      }
      RASTERDB_LOG_DEBUG(
        "[TIMER]   scan: {} {} rows x {} cols", table_name, total_scanned, types.size());

      // Set column metadata — staging IS in VRAM (reBAR device-local+host-visible)
      // GPU reads at full VRAM bandwidth, no DMA copy needed.
      for (size_t c = 0; c < num_cols; c++) {
        if (staging[c].staging_dst) {
          gpu_tbl->columns[c].type           = rdf_types[c];
          gpu_tbl->columns[c].num_rows       = total_scanned;
          gpu_tbl->columns[c].cached_address = bufMgr.cpuStagingAddress() + staging[c].staging_off;
          gpu_tbl->columns[c].cached_buffer  = bufMgr.cpuStagingBuffer();
          gpu_tbl->columns[c].cached_offset  = staging[c].staging_off;
        }
      }
      gpu_tbl->set_num_rows(total_scanned);

      size_t total_bytes = 0;
      for (size_t c = 0; c < num_cols; c++) {
        total_bytes += gpu_tbl->col(c).byte_size();
      }
      RASTERDB_LOG_DEBUG("[RDB_DEBUG]     gpu_upload_detail: {} cols, {} bytes (reBAR zero-copy VRAM)",
                         gpu_tbl->num_columns(),
                         total_bytes);
      RASTERDB_LOG_DEBUG("[TIMER]   gpu_upload                        0.00 ms (reBAR zero-copy)");
    }
  } else {
    // ── FALLBACK: Standard scan + device upload ──
    std::vector<std::unique_ptr<duckdb::DataChunk>> chunks;
    {
      stage_timer t_scan("  cpu_scan");

      duckdb::ThreadContext thread_ctx(_client_ctx);
      duckdb::ExecutionContext exec_ctx(_client_ctx, thread_ctx, nullptr);

      auto table_filters = create_scan_filter_set(op.table_filters, col_ids);
      duckdb::TableFunctionInitInput init_input(
        op.bind_data.get(), col_ids, op.projection_ids, table_filters.get(), op.extra_info.sample_options);

      auto global_state = op.function.init_global(_client_ctx, init_input);

      duckdb::unique_ptr<duckdb::LocalTableFunctionState> local_state;
      if (op.function.init_local) {
        local_state = op.function.init_local(exec_ctx, init_input, global_state.get());
      }

      duckdb::TableFunctionInput tf_input(
        op.bind_data.get(), local_state.get(), global_state.get());

      // Respect _scan_limit: stop scanning once we have enough rows
      rasterdf::size_type scan_row_limit = (_scan_limit > 0)
                                             ? static_cast<rasterdf::size_type>(_scan_limit)
                                             : std::numeric_limits<rasterdf::size_type>::max();

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
    RASTERDB_LOG_DEBUG(
      "[TIMER]   scan: {} {} rows x {} cols", table_name, total_scanned, types.size());

    // Debug: dump first 20 values from each column in CPU staging
    if (debug_logging_enabled() && total_scanned > 0 && !chunks.empty()) {
      size_t sample = std::min((size_t)20, (size_t)total_scanned);
      RASTERDB_LOG_DEBUG("[RDB_DEBUG] SCAN '{}' CPU staging first {}:", table_name, sample);
      for (size_t col_idx = 0; col_idx < types.size(); col_idx++) {
        std::ostringstream line;
        line << "[RDB_DEBUG]   col[" << col_idx << "] type=" << types[col_idx].ToString() << ":";
        size_t row_idx = 0;
        for (auto& chunk : chunks) {
          if (row_idx >= sample) break;
          auto& vec       = chunk->data[col_idx];
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
      gpu_tbl            = gpu_table::from_data_chunks(_ctx, scan_types, chunks);
      size_t total_bytes = 0;
      for (size_t c = 0; c < gpu_tbl->num_columns(); c++) {
        total_bytes += gpu_tbl->col(c).byte_size();
      }
      RASTERDB_LOG_DEBUG("[RDB_DEBUG]     gpu_upload_detail: {} cols, {} bytes (device copy)",
                         gpu_tbl->num_columns(),
                         total_bytes);
    }
  }

  attach_scan_i32_minmax(*gpu_tbl, table_entry, _client_ctx, output_col_ids, scan_types);
  return gpu_tbl;
}

}  // namespace gpu
}  // namespace rasterdb
