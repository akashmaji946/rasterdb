/*
 * Copyright 2025, RasterDB Contributors.
 * GPU Executor — walks DuckDB logical plan, executes via rasterdf Vulkan compute.
 *
 * Strategy:
 *   1. Materialize base table data via DuckDB CPU scan (no reimpl of I/O)
 *   2. Upload to GPU as device_buffers
 *   3. Execute compute (filter, project, aggregate, order, limit) on GPU
 *   4. Download results back to DuckDB DataChunks
 *   5. Unsupported ops throw NotImplementedException -> CPU fallback
 *
 * File decomposition (mirrors Sirius src/op/ pattern):
 *   gpu_executor.cpp           — this file: includes + core dispatch
 *   gpu_executor_internal.hpp  — shared utilities (stage_timer, unwrap_cast, constants)
 *   gpu_executor_scan.inl      — execute_get (LOGICAL_GET)
 *   gpu_executor_filter.inl    — execute_filter, evaluate_comparison, apply_filter_mask
 *   gpu_executor_project.inl   — execute_projection, evaluate_binary_op
 *   gpu_executor_aggregate.inl — execute_aggregate, execute_ungrouped/grouped_aggregate
 *   gpu_executor_order.inl     — execute_order (ORDER BY)
 *   gpu_executor_limit.inl     — execute_limit (LIMIT)
 *   gpu_executor_join.inl      — execute_join (INNER JOIN)
 *   gpu_executor_result.inl    — to_query_result (GPU download → DuckDB QueryResult)
 */

#include "gpu/gpu_executor_internal.hpp"
#include "gpu/gpu_buffer_manager.hpp"

#include <rasterdf/gfx_groupby_engine.hpp>

#include <duckdb/execution/execution_context.hpp>
#include <duckdb/parallel/thread_context.hpp>
#include <algorithm>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <set>

namespace rasterdb {
namespace gpu {

// ============================================================================
// Constructor
// ============================================================================

gpu_executor::gpu_executor(gpu_context& ctx, duckdb::ClientContext& client_ctx)
  : _ctx(ctx), _client_ctx(client_ctx) {}

static void debug_print_plan(duckdb::LogicalOperator& op, int depth = 0);

// ============================================================================
// Top-level execute
// ============================================================================

std::unique_ptr<gpu_table> gpu_executor::execute(duckdb::LogicalOperator& plan)
{
  stage_timer t("TOTAL gpu_execute");

  // Reset the temporary workspace pool strictly for this query's execution
  _ctx.memory().reset_workspace();

  // Reset staging/processing bump pointers once per query (NOT per-scan)
  // so multiple table scans get non-overlapping staging regions.
  if (GPUBufferManager::is_initialized()) {
    auto& bufMgr = GPUBufferManager::GetInstance();
    bufMgr.cpuProcessingPointer.store(0, std::memory_order_relaxed);
    bufMgr.gpuProcessingPointer.store(0, std::memory_order_relaxed);
    bufMgr.gpuCachingPointer.store(0, std::memory_order_relaxed);
  }

  debug_print_plan(plan);
  analyze_plan_hints(plan);
  auto result = execute_operator(plan);
  return result;
}

// ============================================================================
// Pre-analyze the plan tree to set scan optimization hints.
//  - LIMIT pushdown: LIMIT → [PROJECTION →] GET  (no filter/order)
//  - count(*) only: AGGREGATE(count_star only) → [PROJECTION →] GET
// ============================================================================

void gpu_executor::analyze_plan_hints(duckdb::LogicalOperator& plan)
{
  _scan_limit = -1;
  _scan_count_star_only = false;

  // Walk down the plan to find patterns
  auto* cur = &plan;

  // Pattern 1: LIMIT pushdown
  // LIMIT → GET  or  LIMIT → PROJECTION → GET  (no filter, no order)
  if (cur->type == duckdb::LogicalOperatorType::LOGICAL_LIMIT) {
    auto& limit_op = cur->Cast<duckdb::LogicalLimit>();
    int64_t limit_val = -1;
    int64_t offset_val = 0;
    if (limit_op.limit_val.Type() == duckdb::LimitNodeType::CONSTANT_VALUE) {
      limit_val = static_cast<int64_t>(limit_op.limit_val.GetConstantValue());
    }
    if (limit_op.offset_val.Type() == duckdb::LimitNodeType::CONSTANT_VALUE) {
      offset_val = static_cast<int64_t>(limit_op.offset_val.GetConstantValue());
    }
    if (limit_val >= 0 && !limit_op.children.empty()) {
      auto* child = limit_op.children[0].get();
      // Skip PROJECTION node
      if (child->type == duckdb::LogicalOperatorType::LOGICAL_PROJECTION &&
          !child->children.empty()) {
        child = child->children[0].get();
      }
      // If child is a direct GET (no filter/order between), push LIMIT to scan
      if (child->type == duckdb::LogicalOperatorType::LOGICAL_GET) {
        _scan_limit = limit_val + offset_val;
        fprintf(stderr, "[TIMER]   hint: LIMIT pushdown = %ld rows\n", (long)_scan_limit);
      }
    }
  }

  // Pattern 2: count(*) only — all aggregates are count_star, no group columns needed
  // AGGREGATE → [PROJECTION →] GET  (NOT through JOINs/filters — they need all cols)
  if (cur->type == duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
    auto& agg_op = cur->Cast<duckdb::LogicalAggregate>();
    if (agg_op.groups.empty() && !agg_op.expressions.empty()) {
      // Only apply if the child chain leads directly to GET (no JOIN/FILTER)
      auto* child = agg_op.children.empty() ? nullptr : agg_op.children[0].get();
      if (child && child->type == duckdb::LogicalOperatorType::LOGICAL_PROJECTION &&
          !child->children.empty()) {
        child = child->children[0].get();
      }
      bool child_is_get = child &&
        child->type == duckdb::LogicalOperatorType::LOGICAL_GET;

      if (child_is_get) {
        bool all_count_star = true;
        for (auto& expr : agg_op.expressions) {
          if (expr->expression_class == duckdb::ExpressionClass::BOUND_AGGREGATE) {
            auto& agg_expr = expr->Cast<duckdb::BoundAggregateExpression>();
            bool is_cs = agg_expr.children.empty() &&
                         (agg_expr.function.name == "count" || agg_expr.function.name == "count_star");
            if (!is_cs) { all_count_star = false; break; }
          } else {
            all_count_star = false; break;
          }
        }
        if (all_count_star) {
          _scan_count_star_only = true;
          fprintf(stderr, "[TIMER]   hint: count(*) only — scan 1 col\n");
        }
      }
    }
  }

  // Pattern 2b: count(*) behind a PROJECTION
  // PROJECTION → AGGREGATE(count_star) → [PROJECTION →] GET
  if (cur->type == duckdb::LogicalOperatorType::LOGICAL_PROJECTION &&
      !cur->children.empty() &&
      cur->children[0]->type == duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
    auto& agg_op = cur->children[0]->Cast<duckdb::LogicalAggregate>();
    if (agg_op.groups.empty() && !agg_op.expressions.empty()) {
      // Only apply if aggregate's child leads directly to GET
      auto* child2 = agg_op.children.empty() ? nullptr : agg_op.children[0].get();
      if (child2 && child2->type == duckdb::LogicalOperatorType::LOGICAL_PROJECTION &&
          !child2->children.empty()) {
        child2 = child2->children[0].get();
      }
      bool child2_is_get = child2 &&
        child2->type == duckdb::LogicalOperatorType::LOGICAL_GET;

      if (child2_is_get) {
        bool all_count_star = true;
        for (auto& expr : agg_op.expressions) {
          if (expr->expression_class == duckdb::ExpressionClass::BOUND_AGGREGATE) {
            auto& agg_expr = expr->Cast<duckdb::BoundAggregateExpression>();
            bool is_cs = agg_expr.children.empty() &&
                         (agg_expr.function.name == "count" || agg_expr.function.name == "count_star");
            if (!is_cs) { all_count_star = false; break; }
          } else {
            all_count_star = false; break;
          }
        }
        if (all_count_star) {
          _scan_count_star_only = true;
          fprintf(stderr, "[TIMER]   hint: count(*) only — scan 1 col\n");
        }
      }
    }
  }
}

// ============================================================================
// Recursive operator dispatch
// ============================================================================

static void debug_print_plan(duckdb::LogicalOperator& op, int depth) {
  std::string indent(depth * 2, ' ');
  fprintf(stderr, "[RDB_PLAN] %s%s (types=%zu, children=%zu)\n",
          indent.c_str(),
          duckdb::LogicalOperatorToString(op.type).c_str(),
          op.types.size(), op.children.size());
  for (auto& child : op.children) {
    debug_print_plan(*child, depth + 1);
  }
}

std::unique_ptr<gpu_table> gpu_executor::execute_operator(duckdb::LogicalOperator& op)
{
  switch (op.type) {
    case duckdb::LogicalOperatorType::LOGICAL_GET:
      return execute_get(op.Cast<duckdb::LogicalGet>());
    case duckdb::LogicalOperatorType::LOGICAL_FILTER:
      return execute_filter(op.Cast<duckdb::LogicalFilter>());
    case duckdb::LogicalOperatorType::LOGICAL_PROJECTION:
      return execute_projection(op.Cast<duckdb::LogicalProjection>());
    case duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
      return execute_aggregate(op.Cast<duckdb::LogicalAggregate>());
    case duckdb::LogicalOperatorType::LOGICAL_ORDER_BY:
      return execute_order(op.Cast<duckdb::LogicalOrder>());
    case duckdb::LogicalOperatorType::LOGICAL_LIMIT:
      return execute_limit(op.Cast<duckdb::LogicalLimit>());
    case duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
      return execute_join(op.Cast<duckdb::LogicalComparisonJoin>());
    default:
      throw duckdb::NotImplementedException(
        "RasterDB GPU: unsupported operator %s",
        duckdb::LogicalOperatorToString(op.type).c_str());
  }
}

// ============================================================================
// SCAN (LOGICAL_GET) — use DuckDB to scan, then upload to GPU
// ============================================================================

std::unique_ptr<gpu_table> gpu_executor::execute_get(duckdb::LogicalGet& op)
{
  RASTERDB_LOG_DEBUG("GPU execute_get: {}", op.function.name);
  // Debug: check table_filters
  {
    fprintf(stderr, "[RDB_DEBUG] GET '%s': table_filters=%zu\n",
            op.function.name.c_str(), op.table_filters.filters.size());
    auto& cids = op.GetColumnIds();
    fprintf(stderr, "[RDB_DEBUG]   column_ids(%zu):", cids.size());
    for (auto& c : cids) fprintf(stderr, " %zu", (size_t)c.GetPrimaryIndex());
    fprintf(stderr, "\n[RDB_DEBUG]   projection_ids(%zu):", op.projection_ids.size());
    for (auto& p : op.projection_ids) fprintf(stderr, " %zu", (size_t)p);
    fprintf(stderr, "\n[RDB_DEBUG]   names(%zu):", op.names.size());
    for (size_t i = 0; i < op.names.size() && i < 20; i++) fprintf(stderr, " %s", op.names[i].c_str());
    fprintf(stderr, "\n");
    auto bindings = op.GetColumnBindings();
    fprintf(stderr, "[RDB_DEBUG]   bindings(%zu):", bindings.size());
    for (auto& b : bindings) fprintf(stderr, " (%zu,%zu)", (size_t)b.table_index, (size_t)b.column_index);
    fprintf(stderr, "\n");
    fflush(stderr);
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
      fprintf(stderr, "[TIMER]   cpu_scan                         0.00 ms (all cached)\n");
      fprintf(stderr, "[TIMER]   scan: %s %d rows x %zu cols\n",
              table_name.c_str(), (int)total_scanned, types.size());
      fprintf(stderr, "[TIMER]   gpu_upload                        0.00 ms (cached)\n");
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
      fprintf(stderr, "[TIMER]   scan: %s %d rows x %zu cols\n",
              table_name.c_str(), (int)total_scanned, types.size());

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
      fprintf(stderr, "[RDB_DEBUG]     gpu_upload_detail: %zu cols, %zu bytes (zero-copy reBAR)\n",
              gpu_tbl->num_columns(), total_bytes);
      fprintf(stderr, "[TIMER]   gpu_upload                        0.00 ms (zero-copy)\n");
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
    fprintf(stderr, "[TIMER]   scan: %s %d rows x %zu cols\n",
            table_name.c_str(), (int)total_scanned, types.size());

    // Debug: dump first 20 values from each column in CPU staging
    if (total_scanned > 0 && !chunks.empty()) {
      size_t sample = std::min((size_t)20, (size_t)total_scanned);
      fprintf(stderr, "[RDB_DEBUG] SCAN '%s' CPU staging first %zu:\n", table_name.c_str(), sample);
      for (size_t col_idx = 0; col_idx < types.size(); col_idx++) {
        fprintf(stderr, "[RDB_DEBUG]   col[%zu] type=%s:", col_idx, types[col_idx].ToString().c_str());
        size_t row_idx = 0;
        for (auto& chunk : chunks) {
          if (row_idx >= sample) break;
          auto& vec = chunk->data[col_idx];
          auto chunk_rows = std::min((size_t)chunk->size(), sample - row_idx);
          if (types[col_idx] == duckdb::LogicalType::FLOAT) {
            auto floats = duckdb::FlatVector::GetData<float>(vec);
            for (size_t i = 0; i < chunk_rows; i++) {
              fprintf(stderr, " %.0f", floats[i]);
            }
          } else if (types[col_idx] == duckdb::LogicalType::INTEGER) {
            auto ints = duckdb::FlatVector::GetData<int32_t>(vec);
            for (size_t i = 0; i < chunk_rows; i++) {
              fprintf(stderr, " %d", ints[i]);
            }
          } else {
            fprintf(stderr, " ?");
          }
          row_idx += chunk_rows;
          if (row_idx >= sample) break;
        }
        fprintf(stderr, "\n");
      }
      fflush(stderr);
    }

    {
      stage_timer t_upload("  gpu_upload");
      gpu_tbl = gpu_table::from_data_chunks(_ctx, types, chunks);
      size_t total_bytes = 0;
      for (size_t c = 0; c < gpu_tbl->num_columns(); c++) {
        total_bytes += gpu_tbl->col(c).byte_size();
      }
      fprintf(stderr, "[RDB_DEBUG]     gpu_upload_detail: %zu cols, %zu bytes (device copy)\n",
              gpu_tbl->num_columns(), total_bytes);
    }
  }

  return gpu_tbl;
}

// ============================================================================
// FILTER — evaluate WHERE clause predicates on GPU
// ============================================================================
std::unique_ptr<gpu_table> gpu_executor::execute_filter(duckdb::LogicalFilter& op)
{
  RASTERDB_LOG_DEBUG("GPU execute_filter");
  D_ASSERT(op.children.size() == 1);
  auto input = execute_operator(*op.children[0]);

  RASTERDB_LOG_DEBUG("Filter input: {} rows x {} cols", input->num_rows(), input->num_columns());
  for (size_t c = 0; c < input->num_columns(); c++) {
    RASTERDB_LOG_DEBUG("  input col {}: {} rows, type={}, addr={}", c, input->col(c).num_rows,
                       static_cast<int>(input->col(c).type.id), input->col(c).address());
  }

  fprintf(stderr, "[RDB_DEBUG] FILTER: %zu expressions, %d rows x %zu cols\n",
          op.expressions.size(), (int)input->num_rows(), input->num_columns());
  stage_timer t("  filter (total)");

  if (input->num_rows() == 0) return input;

  // Evaluate each filter expression to produce a boolean (int32 0/1) mask
  gpu_column mask;
  bool first = true;

  auto t_compare_start = std::chrono::high_resolution_clock::now();
  for (auto& expr : op.expressions) {
    gpu_column expr_mask = evaluate_comparison(*input, *expr);

    if (first) {
      mask = std::move(expr_mask);
      first = false;
    } else {
      // AND masks: multiply element-wise (0/1 * 0/1 = AND)
      auto result = allocate_column(_ctx, {rasterdf::type_id::INT32}, mask.num_rows);
      binary_op_push_constants pc{};
      pc.input_a = mask.address();
      pc.input_b = expr_mask.address();
      pc.output_addr = result.address();
      pc.size = static_cast<uint32_t>(mask.num_rows);
      pc.op = 2;       // MUL
      pc.scalar_val = 0;
      pc.mode = 0;     // COL_COL
      pc.debug_mode = 0;
      pc.type_id = 0;  // int32
      _ctx.dispatcher().dispatch_binary_op(pc);
      mask = std::move(result);
    }
  }

  {
    auto t_compare_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_compare_end - t_compare_start).count();
    fprintf(stderr, "[TIMER]     filter_compare             %8.2f ms\n", ms);
  }

  // Apply the mask to compact the table
  auto t_compact_start = std::chrono::high_resolution_clock::now();
  auto result = apply_filter_mask(*input, mask);
  {
    auto t_compact_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_compact_end - t_compact_start).count();
    fprintf(stderr, "[TIMER]     filter_compact             %8.2f ms\n", ms);
  }
  fprintf(stderr, "[RDB_DEBUG] filter: %d rows => %d rows\n",
          (int)input->num_rows(), (int)result->num_rows());
  fflush(stderr);
  return result;
}

// ============================================================================
// Apply boolean mask — GPU-side order-preserving stream compaction.
// Uses prefix scan on mask, then scatter_if per column. No CPU transfer.
// ============================================================================

std::unique_ptr<gpu_table> gpu_executor::apply_filter_mask(const gpu_table& input, gpu_column& mask)
{
  uint32_t n = static_cast<uint32_t>(input.num_rows());
  auto& disp = _ctx.dispatcher();
  auto* mr = _ctx.workspace_mr();

  // --- Step 1: GPU-side copy mask → prefix buffer (no host round-trip) ----
  // prefix_buf[0..N-1] = mask[0..N-1], prefix_buf[N] = 0 (sentinel)
  uint32_t prefix_count = n + 1;

  rasterdf::device_buffer prefix_buf(
    mr, prefix_count * sizeof(uint32_t),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

  // GPU-to-GPU copy: mask[0..N-1] → prefix_buf[0..N-1]
  disp.copy_buffer(mask.data.buffer(), prefix_buf.buffer(), n * sizeof(uint32_t),
                   mask.data.offset(), prefix_buf.offset());
  // Zero the sentinel element prefix_buf[N]
  disp.fill_buffer(prefix_buf.buffer(), 0u, sizeof(uint32_t),
                   prefix_buf.offset() + n * sizeof(uint32_t));

  // --- Step 2: 3-pass prefix scan on prefix_buf ---------------------------
  uint32_t scan_wg = 256;
  uint32_t scan_groups = (prefix_count + scan_wg - 1) / scan_wg;

  rasterdf::device_buffer block_sums(
    mr, scan_groups * sizeof(uint32_t),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

  // Device buffer for total_sum (written by scan_global shader)
  rasterdf::device_buffer total_sum_buf(
    mr, sizeof(uint32_t),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

  prefix_scan_pc scan_pc{};
  scan_pc.data_ptr = prefix_buf.data();
  scan_pc.block_sums_ptr = block_sums.data();
  scan_pc.total_sum_ptr = total_sum_buf.data();
  scan_pc.numElements = prefix_count;
  scan_pc.blockCount = scan_groups;

  disp.dispatch_prefix_scan_local(scan_pc, scan_groups);
  disp.dispatch_prefix_scan_global(scan_pc);
  disp.dispatch_prefix_scan_add(scan_pc, scan_groups);

  // --- Step 3: Read total count via copy_to_host --------------------------
  uint32_t output_count = 0;
  total_sum_buf.copy_to_host(&output_count, sizeof(uint32_t),
                             _ctx.device(), _ctx.queue(), _ctx.command_pool());

  RASTERDB_LOG_DEBUG("Filter (GPU compaction): {} -> {} rows", n, output_count);

  auto result = std::make_unique<gpu_table>();
  result->duckdb_types = input.duckdb_types;
  result->columns.resize(input.num_columns());

  if (output_count == 0) {
    for (size_t c = 0; c < input.num_columns(); c++) {
      result->columns[c].type = input.col(c).type;
      result->columns[c].num_rows = 0;
    }
    RASTERDB_LOG_DEBUG("Filter result: 0 rows (early return)");
    return result;
  }

  // --- Step 4: Scatter each column using prefix sums ----------------------
  uint32_t scatter_groups = (n + 255) / 256;
  for (size_t c = 0; c < input.num_columns(); c++) {
    auto& in_col = input.col(c);
    result->columns[c] = allocate_column(_ctx, in_col.type,
                                          static_cast<rasterdf::size_type>(output_count));

    scatter_if_pc spc{};
    spc.input_ptr = in_col.address();
    spc.prefix_ptr = prefix_buf.data();
    spc.output_ptr = result->columns[c].address();
    spc.size = n;
    if (rdf_type_size(in_col.type.id) == 8) {
      disp.dispatch_scatter_if_64(spc, scatter_groups);
    } else {
      disp.dispatch_scatter_if(spc, scatter_groups);
    }
  }

  result->set_num_rows(static_cast<rasterdf::size_type>(output_count));
  RASTERDB_LOG_DEBUG("Filter result: {} rows x {} cols", result->num_rows(), result->num_columns());
  return result;
}

// unwrap_cast is defined in gpu_executor_internal.hpp

// ============================================================================
// Evaluate comparison expression -> int32 mask (0/1 per element)
// ============================================================================

gpu_column gpu_executor::evaluate_comparison(const gpu_table& input, duckdb::Expression& expr)
{
  auto& disp = _ctx.dispatcher();
  uint32_t n = static_cast<uint32_t>(input.num_rows());

  // Comparison: column <op> constant or column <op> column
  if (expr.type == duckdb::ExpressionType::COMPARE_LESSTHAN ||
      expr.type == duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO ||
      expr.type == duckdb::ExpressionType::COMPARE_GREATERTHAN ||
      expr.type == duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO ||
      expr.type == duckdb::ExpressionType::COMPARE_EQUAL ||
      expr.type == duckdb::ExpressionType::COMPARE_NOTEQUAL) {

    auto& cmp = expr.Cast<duckdb::BoundComparisonExpression>();

    // Map to shader op code: 0=gt, 1=lt, 2=ge, 3=le, 4=eq, 5=ne
    int32_t cmp_op = 0;
    switch (expr.type) {
      case duckdb::ExpressionType::COMPARE_GREATERTHAN:         cmp_op = 0; break;
      case duckdb::ExpressionType::COMPARE_LESSTHAN:            cmp_op = 1; break;
      case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO: cmp_op = 2; break;
      case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:   cmp_op = 3; break;
      case duckdb::ExpressionType::COMPARE_EQUAL:               cmp_op = 4; break;
      case duckdb::ExpressionType::COMPARE_NOTEQUAL:            cmp_op = 5; break;
      default: break;
    }

    // Unwrap casts inserted by the optimizer
    auto& left = unwrap_cast(*cmp.left);
    auto& right = unwrap_cast(*cmp.right);

    // Column vs constant
    if (left.type == duckdb::ExpressionType::BOUND_REF &&
        right.type == duckdb::ExpressionType::VALUE_CONSTANT) {

      auto& col_ref = left.Cast<duckdb::BoundReferenceExpression>();
      auto& constant = right.Cast<duckdb::BoundConstantExpression>();
      auto& col = input.col(col_ref.index);
      int32_t type_id = rdf_shader_type_id(col.type.id);

      auto result = allocate_column(_ctx, {rasterdf::type_id::INT32}, input.num_rows());

      // Cast the constant value to match the column's native type
      int32_t threshold = 0;
      if (type_id == 0) { // int32
        threshold = constant.value.DefaultCastAs(duckdb::LogicalType::INTEGER).GetValue<int32_t>();
      } else { // float32
        float fval = constant.value.DefaultCastAs(duckdb::LogicalType::FLOAT).GetValue<float>();
        std::memcpy(&threshold, &fval, sizeof(float));
      }

      compare_push_constants pc{};
      pc.input_addr = col.address();
      pc.output_addr = result.address();
      pc.size = n;
      pc.threshold = threshold;
      pc.op = cmp_op;
      pc.type_id = type_id;

      disp.dispatch_compare(pc);
      return result;

    } else if (left.type == duckdb::ExpressionType::BOUND_REF &&
               right.type == duckdb::ExpressionType::BOUND_REF) {
      auto& left_ref = left.Cast<duckdb::BoundReferenceExpression>();
      auto& right_ref = right.Cast<duckdb::BoundReferenceExpression>();
      auto& left_col = input.col(left_ref.index);
      auto& right_col = input.col(right_ref.index);
      int32_t type_id = rdf_shader_type_id(left_col.type.id);

      auto result = allocate_column(_ctx, {rasterdf::type_id::INT32}, input.num_rows());

      compare_columns_push_constants pc{};
      pc.input_a = left_col.address();
      pc.input_b = right_col.address();
      pc.output_addr = result.address();
      pc.size = n;
      pc.op = cmp_op;
      pc.type_id = type_id;
      disp.dispatch_compare_columns(pc);
      return result;

    } else {
      throw duckdb::NotImplementedException(
        "RasterDB GPU: unsupported comparison operand types (left=%s, right=%s)",
        duckdb::ExpressionTypeToString(left.type).c_str(),
        duckdb::ExpressionTypeToString(right.type).c_str());
    }
  }

  // Conjunction (AND)
  if (expr.type == duckdb::ExpressionType::CONJUNCTION_AND) {
    auto& conj = expr.Cast<duckdb::BoundConjunctionExpression>();
    gpu_column result = evaluate_comparison(input, *conj.children[0]);
    for (size_t i = 1; i < conj.children.size(); i++) {
      gpu_column child_mask = evaluate_comparison(input, *conj.children[i]);
      auto combined = allocate_column(_ctx, {rasterdf::type_id::INT32}, result.num_rows);
      binary_op_push_constants pc{};
      pc.input_a = result.address();
      pc.input_b = child_mask.address();
      pc.output_addr = combined.address();
      pc.size = static_cast<uint32_t>(result.num_rows);
      pc.op = 2; pc.mode = 0; pc.type_id = 0; pc.debug_mode = 0; pc.scalar_val = 0;
      disp.dispatch_binary_op(pc);
      result = std::move(combined);
    }
    return result;
  }

  throw duckdb::NotImplementedException(
    "RasterDB GPU: unsupported filter expression type %s",
    duckdb::ExpressionTypeToString(expr.type).c_str());
}

// ============================================================================
// PROJECTION — select columns + evaluate arithmetic expressions
// ============================================================================

std::unique_ptr<gpu_table> gpu_executor::execute_projection(duckdb::LogicalProjection& op)
{
  RASTERDB_LOG_DEBUG("GPU execute_projection");
  D_ASSERT(op.children.size() == 1);
  auto input = execute_operator(*op.children[0]);

  stage_timer t("  projection");  // Timer starts AFTER child execution

  auto result = std::make_unique<gpu_table>();
  result->duckdb_types = op.types;
  result->columns.resize(op.expressions.size());

  for (size_t i = 0; i < op.expressions.size(); i++) {
    auto& expr = *op.expressions[i];

    if (expr.type == duckdb::ExpressionType::BOUND_REF) {
      auto& ref = expr.Cast<duckdb::BoundReferenceExpression>();
      auto& src = input->col(ref.index);

      // If source is a host-only column (e.g. scalar aggregate), pass through directly
      if (src.is_host_only) {
        result->columns[i].type = src.type;
        result->columns[i].num_rows = src.num_rows;
        result->columns[i].is_host_only = true;
        result->columns[i].host_data = src.host_data;
        continue;
      }

      // Copy column to output — use shader for INT32/FLOAT32, buffer copy otherwise
      result->columns[i] = allocate_column(_ctx, src.type, src.num_rows);
      bool has_shader = (src.type.id == rasterdf::type_id::INT32 ||
                         src.type.id == rasterdf::type_id::FLOAT32 ||
                         src.type.id == rasterdf::type_id::TIMESTAMP_DAYS);
      if (has_shader) {
        binary_op_push_constants pc{};
        pc.input_a = src.address();
        pc.input_b = 0;
        pc.output_addr = result->columns[i].address();
        pc.size = static_cast<uint32_t>(src.num_rows);
        pc.op = 0; pc.scalar_val = 0; pc.mode = 1; pc.debug_mode = 0;
        pc.type_id = rdf_shader_type_id(src.type.id);
        _ctx.dispatcher().dispatch_binary_op(pc);
      } else {
        // Fallback: host round-trip copy for types without shader support (e.g. INT64/FLOAT64)
        // Only used for small aggregate results so overhead is negligible.
        size_t byte_count = static_cast<size_t>(src.num_rows) * rdf_type_size(src.type.id);
        std::vector<uint8_t> tmp(byte_count);
        src.data.copy_to_host(tmp.data(), byte_count,
                              _ctx.device(), _ctx.queue(), _ctx.command_pool());
        result->columns[i].data.copy_from_host(tmp.data(), byte_count,
                                                _ctx.device(), _ctx.queue(), _ctx.command_pool());
      }
    } else if (expr.type == duckdb::ExpressionType::BOUND_FUNCTION) {
      result->columns[i] = evaluate_binary_op(*input, expr);
    } else {
      throw duckdb::NotImplementedException(
        "RasterDB GPU: unsupported projection expression %s",
        duckdb::ExpressionTypeToString(expr.type).c_str());
    }
  }

  return result;
}

// ============================================================================
// Evaluate binary function expression -> gpu_column
// ============================================================================

// ============================================================================
// Evaluate any expression against a gpu_table → returns a gpu_column.
// Handles: BOUND_REF, VALUE_CONSTANT, BOUND_FUNCTION, BOUND_CAST
// ============================================================================

gpu_column gpu_executor::evaluate_expression(const gpu_table& input, duckdb::Expression& raw_expr)
{
  // Strip casts
  auto& expr = unwrap_cast(raw_expr);

  switch (expr.type) {
  case duckdb::ExpressionType::BOUND_REF: {
    auto& ref = expr.Cast<duckdb::BoundReferenceExpression>();
    auto& src = input.col(ref.index);
    // Return a lightweight alias that references the same device memory
    gpu_column col;
    col.type = src.type;
    col.num_rows = src.num_rows;
    // Don't copy device_buffer (deleted copy). Use cached address to alias.
    col.cached_address = src.address();
    col.cached_buffer = src.cached_buffer;
    col.is_host_only = src.is_host_only;
    col.host_data = src.host_data;
    return col;
  }
  case duckdb::ExpressionType::VALUE_CONSTANT: {
    // Broadcast scalar to a full column
    auto& c = expr.Cast<duckdb::BoundConstantExpression>();
    rasterdf::data_type rdf_type = to_rdf_type(c.return_type);
    // Downcast FLOAT64 constants to FLOAT32 (shader only supports INT32/FLOAT32)
    if (rdf_type.id == rasterdf::type_id::FLOAT64) {
      rdf_type = {rasterdf::type_id::FLOAT32};
    }
    auto col = allocate_column(_ctx, rdf_type, input.num_rows());
    // Fill via binary_op: col = 0 + scalar (broadcast)
    binary_op_push_constants pc{};
    pc.input_a = col.address();  // will be overwritten
    pc.output_addr = col.address();
    pc.size = static_cast<uint32_t>(input.num_rows());
    pc.op = 0; // ADD (0 + scalar = scalar)
    pc.mode = 1; // COL_SCALAR — but we need a "fill" op. Use MUL 0 + scalar via ADD trick:
    // Actually, just copy scalar: col = input_a * 0 + scalar? No. Use: output = 0_col + scalar.
    // Simplest: use the identity: set input_a to the same address, op=MUL, scalar=0, then ADD scalar.
    // Actually let's just memset the staging and fill:
    // Easier: binary_op with mode=1 (COL_SCALAR), op=2 (MUL), scalar=1 then scalar=val via ADD
    // Let me just use: output = 0 + scalar via COL_SCALAR ADD with the output as input
    // Hmm, we need a cleaner approach. For now, just allocate and fill on host for small scalar cols.
    // This is used for constant expressions in aggregates, typically rare.
    {
      int32_t type_id_s = rdf_shader_type_id(rdf_type.id);
      // First zero the column, then add scalar to fill
      // Use ADD with scalar, mode=1 means input_a + scalar_val
      // To broadcast, we need input_a to be 0. Zero it first.
      // Actually: simplest approach — allocate, zero with MUL 0, then ADD scalar.
      binary_op_push_constants pz{};
      pz.input_a = col.address();
      pz.output_addr = col.address();
      pz.size = static_cast<uint32_t>(input.num_rows());
      pz.op = 2; // MUL
      pz.mode = 1; // COL_SCALAR
      pz.scalar_val = 0;
      pz.type_id = type_id_s;
      pz.debug_mode = 0;
      _ctx.dispatcher().dispatch_binary_op(pz);

      binary_op_push_constants pa{};
      pa.input_a = col.address();
      pa.output_addr = col.address();
      pa.size = static_cast<uint32_t>(input.num_rows());
      pa.op = 0; // ADD
      pa.mode = 1; // COL_SCALAR
      pa.type_id = type_id_s;
      pa.debug_mode = 0;
      if (type_id_s == 0) {
        pa.scalar_val = c.value.GetValue<int32_t>();
      } else {
        float fval = c.value.GetValue<float>();
        std::memcpy(&pa.scalar_val, &fval, sizeof(float));
      }
      _ctx.dispatcher().dispatch_binary_op(pa);
    }
    return col;
  }
  case duckdb::ExpressionType::BOUND_FUNCTION:
    return evaluate_binary_op(input, expr);
  default:
    throw duckdb::NotImplementedException(
      "RasterDB GPU: unsupported expression type %s in evaluate_expression",
      duckdb::ExpressionTypeToString(expr.type).c_str());
  }
}

// Helper: cast an INT32 gpu_column to FLOAT32 via CPU round-trip.
// Used to align mixed-type operands before dispatching the (single-type) binary_op shader.
static gpu_column cast_int32_to_float32(gpu_context& ctx, const gpu_column& src)
{
  size_t n = static_cast<size_t>(src.num_rows);
  const int32_t* src_int = nullptr;
  std::vector<int32_t> h_int;

  auto& bufMgr = GPUBufferManager::GetInstance();
  if (src.cached_address != 0 && src.cached_buffer == bufMgr.cpuStagingBuffer()) {
    // Zero-copy reBAR: data is directly accessible via mapped CPU staging.
    size_t staging_off = static_cast<size_t>(
        src.cached_address - bufMgr.cpuStagingAddress());
    src_int = reinterpret_cast<const int32_t*>(bufMgr.cpuProcessing + staging_off);
  } else {
    // gpuCache-backed or device_buffer-owned: fall back to generic download.
    h_int.resize(n);
    download_column(ctx, src, h_int.data(), n * sizeof(int32_t));
    src_int = h_int.data();
  }

  std::vector<float> h_flt(n);
  for (size_t i = 0; i < n; i++) h_flt[i] = static_cast<float>(src_int[i]);
  auto out = allocate_column(ctx, {rasterdf::type_id::FLOAT32}, static_cast<rasterdf::size_type>(n));
  out.data.copy_from_host(h_flt.data(), n * sizeof(float),
                          ctx.device(), ctx.queue(), ctx.command_pool());
  return out;
}

gpu_column gpu_executor::evaluate_binary_op(const gpu_table& input, duckdb::Expression& expr)
{
  auto& func = expr.Cast<duckdb::BoundFunctionExpression>();
  auto& fname = func.function.name;

  // Map function names to binary op codes: 0=ADD, 1=SUB, 2=MUL, 3=DIV
  int32_t op_code = -1;
  if (fname == "+" || fname == "add") op_code = 0;
  else if (fname == "-" || fname == "subtract") op_code = 1;
  else if (fname == "*" || fname == "multiply") op_code = 2;
  else if (fname == "/" || fname == "divide") op_code = 3;
  else if (fname == "%" || fname == "modulo") op_code = 4;
  else {
    throw duckdb::NotImplementedException(
      "RasterDB GPU: unsupported function '%s'", fname.c_str());
  }

  D_ASSERT(func.children.size() == 2);

  // Recursively evaluate both operands
  auto& left_expr = unwrap_cast(*func.children[0]);
  auto& right_expr = unwrap_cast(*func.children[1]);

  rasterdf::data_type out_type = to_rdf_type(func.return_type);
  // Downcast FLOAT64 to FLOAT32 for binary op shader (inputs are FLOAT32 from the integer dataset;
  // the aggregation shader accumulates FLOAT32 values in double precision internally)
  if (out_type.id == rasterdf::type_id::FLOAT64) {
    out_type = {rasterdf::type_id::FLOAT32};
  }
  int32_t type_id = rdf_shader_type_id(out_type.id);
  auto result = allocate_column(_ctx, out_type, input.num_rows());

  binary_op_push_constants pc{};
  pc.output_addr = result.address();
  pc.size = static_cast<uint32_t>(input.num_rows());
  pc.op = op_code;
  pc.debug_mode = 0;
  pc.type_id = type_id;

  // Check for simple col-col or col-scalar cases first (avoid temp allocation)
  bool left_is_ref = left_expr.type == duckdb::ExpressionType::BOUND_REF;
  bool right_is_ref = right_expr.type == duckdb::ExpressionType::BOUND_REF;
  bool right_is_const = right_expr.type == duckdb::ExpressionType::VALUE_CONSTANT;
  bool left_is_const = left_expr.type == duckdb::ExpressionType::VALUE_CONSTANT;

  // Resolve addresses — evaluate complex sub-expressions to temp columns
  gpu_column left_temp, right_temp;    // keep alive for address validity
  gpu_column left_cast, right_cast;    // keep cast-to-float temp alive if needed

  // Helper lambda: resolve an operand to (addr, source_type_id). For a BOUND_REF,
  // reads directly from input.col(idx); otherwise evaluates into `temp_out`.
  auto resolve_operand =
      [&](duckdb::Expression& e, bool is_ref, gpu_column& temp_out)
      -> std::pair<VkDeviceAddress, rasterdf::type_id> {
    if (is_ref) {
      auto& ref = e.Cast<duckdb::BoundReferenceExpression>();
      const auto& src = input.col(ref.index);
      return {src.address(), src.type.id};
    }
    temp_out = evaluate_expression(input, e);
    return {temp_out.address(), temp_out.type.id};
  };

  VkDeviceAddress left_addr = 0;
  rasterdf::type_id left_src_type = rasterdf::type_id::INT32;
  if (!left_is_const) {
    auto r = resolve_operand(left_expr, left_is_ref, left_temp);
    left_addr = r.first;
    left_src_type = r.second;
  }

  // If shader will run the float path but this operand is INT32, cast it.
  auto align_if_int_to_float = [&](VkDeviceAddress& addr,
                                   rasterdf::type_id& src_type,
                                   gpu_column& cast_out,
                                   const gpu_column* ref_src) {
    if (out_type.id != rasterdf::type_id::FLOAT32) return;
    if (src_type == rasterdf::type_id::INT32) {
      // Build a gpu_column view of the int32 source so we can cast it.
      gpu_column tmp_view;
      tmp_view.type = {rasterdf::type_id::INT32};
      tmp_view.num_rows = static_cast<rasterdf::size_type>(input.num_rows());
      if (ref_src) {
        tmp_view.cached_address = ref_src->address();
        tmp_view.cached_buffer = ref_src->cached_buffer;
        tmp_view.is_host_only = ref_src->is_host_only;
        tmp_view.host_data = ref_src->host_data;
        cast_out = cast_int32_to_float32(_ctx, *ref_src);
      } else {
        // Can't cheaply view without the gpu_column; fall back to a re-read.
        // Not expected in current flows (non-ref temps are already float).
        throw duckdb::NotImplementedException(
          "RasterDB GPU: unexpected INT32 temp operand requiring float cast");
      }
      addr = cast_out.address();
      src_type = rasterdf::type_id::FLOAT32;
    }
  };

  // Apply cast to left operand if needed (only for BOUND_REF, where we can access the source column)
  if (!left_is_const && left_is_ref) {
    auto& ref = left_expr.Cast<duckdb::BoundReferenceExpression>();
    align_if_int_to_float(left_addr, left_src_type, left_cast, &input.col(ref.index));
  }

  if (left_is_ref && right_is_ref) {
    auto& rref = right_expr.Cast<duckdb::BoundReferenceExpression>();
    VkDeviceAddress right_addr = input.col(rref.index).address();
    rasterdf::type_id right_src_type = input.col(rref.index).type.id;
    align_if_int_to_float(right_addr, right_src_type, right_cast, &input.col(rref.index));
    pc.input_a = left_addr;
    pc.input_b = right_addr;
    pc.mode = 0; // COL_COL
    pc.scalar_val = 0;
  } else if (left_addr != 0 && right_is_const) {
    auto& c = right_expr.Cast<duckdb::BoundConstantExpression>();
    pc.input_a = left_addr;
    pc.input_b = 0;
    pc.mode = 1; // COL_SCALAR
    if (type_id == 0) {
      pc.scalar_val = c.value.GetValue<int32_t>();
    } else {
      float fval = c.value.GetValue<float>();
      std::memcpy(&pc.scalar_val, &fval, sizeof(float));
    }
  } else if (left_is_const && right_is_ref) {
    // SCALAR op COL — swap to COL op SCALAR with adjusted op (only for commutative, else temp)
    auto& c = left_expr.Cast<duckdb::BoundConstantExpression>();
    VkDeviceAddress right_addr = input.col(right_expr.Cast<duckdb::BoundReferenceExpression>().index).address();
    // For subtraction (scalar - col), evaluate scalar as column
    if (op_code == 1 || op_code == 3 || op_code == 4) {
      // Non-commutative: evaluate left as column
      left_temp = evaluate_expression(input, left_expr);
      pc.input_a = left_temp.address();
      pc.input_b = right_addr;
      pc.mode = 0; // COL_COL
    } else {
      // Commutative (ADD, MUL): swap
      pc.input_a = right_addr;
      pc.input_b = 0;
      pc.mode = 1; // COL_SCALAR
      if (type_id == 0) {
        pc.scalar_val = c.value.GetValue<int32_t>();
      } else {
        float fval = c.value.GetValue<float>();
        std::memcpy(&pc.scalar_val, &fval, sizeof(float));
      }
    }
  } else {
    // General case: evaluate both operands to temp columns
    if (!left_addr) {
      left_temp = evaluate_expression(input, left_expr);
      left_addr = left_temp.address();
    }
    right_temp = evaluate_expression(input, right_expr);
    pc.input_a = left_addr;
    pc.input_b = right_temp.address();
    pc.mode = 0; // COL_COL
    pc.scalar_val = 0;
  }

  _ctx.dispatcher().dispatch_binary_op(pc);
  return result;
}

// ============================================================================
// AGGREGATE (ungrouped only for now)
// ============================================================================

std::unique_ptr<gpu_table> gpu_executor::execute_aggregate(duckdb::LogicalAggregate& op)
{
  RASTERDB_LOG_DEBUG("GPU execute_aggregate");
  D_ASSERT(op.children.size() == 1);
  auto input = execute_operator(*op.children[0]);

  stage_timer t("  aggregate");  // Timer starts AFTER child execution

  if (!op.groups.empty()) {
    auto result = std::make_unique<gpu_table>();
    // op.types may be empty in unoptimized plans — compute output count manually
    size_t out_cols = op.groups.size() + op.expressions.size();
    result->columns.resize(out_cols);
    // Build duckdb_types: group key types + aggregate result types
    // For now, leave duckdb_types empty — to_query_result will infer from gpu_columns
    execute_grouped_aggregate(*input, op.groups, op.expressions, op.types, *result);
    return result;
  }

  auto result = std::make_unique<gpu_table>();
  result->duckdb_types = op.types;
  result->columns.resize(op.expressions.size());

  execute_ungrouped_aggregate(*input, op.expressions, *result);
  return result;
}

void gpu_executor::execute_ungrouped_aggregate(
  const gpu_table& input,
  const duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& aggregates,
  gpu_table& output)
{
  uint32_t n = static_cast<uint32_t>(input.num_rows());

  for (duckdb::idx_t i = 0; i < aggregates.size(); i++) {
    auto& expr = aggregates[i]->Cast<duckdb::BoundAggregateExpression>();
    auto& fname = expr.function.name;

    bool is_count_star = false;

    if (expr.children.empty()) {
      is_count_star = (fname == "count" || fname == "count_star");
      if (!is_count_star) {
        throw duckdb::NotImplementedException(
          "RasterDB GPU: aggregate '%s' with no children", fname.c_str());
      }
    }

    if (is_count_star) {
      // count(*) — store directly as host-side scalar (no GPU needed)
      output.columns[i].type = {rasterdf::type_id::INT64};
      output.columns[i].num_rows = 1;
      output.columns[i].is_host_only = true;
      output.columns[i].host_data.resize(sizeof(int64_t));
      int64_t count = static_cast<int64_t>(n);
      std::memcpy(output.columns[i].host_data.data(), &count, sizeof(int64_t));
      fprintf(stderr, "[RDB_DEBUG]     count_star = %ld (host-only)\n", (long)count);
      continue;
    }

    // Evaluate the aggregate's value expression (may be a column ref or complex expr)
    gpu_column val_col = evaluate_expression(input, *expr.children[0]);
    auto col_view = val_col.view();

    rasterdf::aggregation_kind kind;
    if (fname == "sum" || fname == "sum_no_overflow") {
      kind = rasterdf::aggregation_kind::SUM;
    } else if (fname == "min") {
      kind = rasterdf::aggregation_kind::MIN;
    } else if (fname == "max") {
      kind = rasterdf::aggregation_kind::MAX;
    } else if (fname == "count") {
      kind = rasterdf::aggregation_kind::COUNT_VALID;
    } else if (fname == "avg" || fname == "mean") {
      kind = rasterdf::aggregation_kind::MEAN;
    } else {
      throw duckdb::NotImplementedException(
        "RasterDB GPU: unsupported aggregate function '%s'", fname.c_str());
    }

    rasterdf::reduce_aggregation agg(kind);

    auto t_reduce = std::chrono::high_resolution_clock::now();
    auto scalar = rasterdf::reduce(col_view, agg, val_col.type,
                                   _ctx.vk_context(), _ctx.dispatcher(), _ctx.workspace_mr());
    auto t_reduce_end = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[RDB_DEBUG]     reduce(%s)              %8.2f ms\n",
            fname.c_str(),
            std::chrono::duration<double, std::milli>(t_reduce_end - t_reduce).count());

    // PERF FIX: Store scalar in host_data — skip GPU alloc + copy round-trip
    auto out_type = scalar->type;
    output.columns[i].type = out_type;
    output.columns[i].num_rows = 1;
    output.columns[i].is_host_only = true;

    switch (out_type.id) {
    case rasterdf::type_id::INT64: {
      output.columns[i].host_data.resize(sizeof(int64_t));
      int64_t val = scalar->as<int64_t>();
      std::memcpy(output.columns[i].host_data.data(), &val, sizeof(int64_t));
      break;
    }
    case rasterdf::type_id::FLOAT64: {
      output.columns[i].host_data.resize(sizeof(double));
      double val = scalar->as<double>();
      std::memcpy(output.columns[i].host_data.data(), &val, sizeof(double));
      break;
    }
    case rasterdf::type_id::FLOAT32: {
      output.columns[i].host_data.resize(sizeof(float));
      float val = scalar->as<float>();
      std::memcpy(output.columns[i].host_data.data(), &val, sizeof(float));
      break;
    }
    case rasterdf::type_id::INT32: {
      output.columns[i].host_data.resize(sizeof(int32_t));
      int32_t val = scalar->as<int32_t>();
      std::memcpy(output.columns[i].host_data.data(), &val, sizeof(int32_t));
      break;
    }
    default:
      throw duckdb::NotImplementedException(
        "RasterDB GPU: unsupported reduce output type_id %d",
        static_cast<int>(out_type.id));
    }
    fprintf(stderr, "[RDB_DEBUG]     agg(%s) stored host-only (no GPU alloc)\n", fname.c_str());
  }
}

// ============================================================================
// GROUP BY aggregate — hash-based groupby via rasterdf
// Supports 1-, 2-, and 3-column GROUP BY keys.
// Multi-column keys use a composite INT32 key: col0*M+col1 (2-col) or
// (col0*M+col1)*M+col2 (3-col), then decompose after groupby.
// ============================================================================

// Composite key multipliers.
// INT32 path (GPU): fast, but limited to small-range group columns.
static constexpr int32_t GROUPBY_COMPOSITE_M_I32 = 10007;
// INT64 path (CPU): handles any value range, used when INT32 would overflow.
static constexpr int64_t GROUPBY_COMPOSITE_M_I64 = 100000007LL;

// Toggle between compute-shader groupby and mesh-shader gfxm groupby
static constexpr bool USE_SIMPLE_GFX_AGGR = false;

void gpu_executor::execute_grouped_aggregate(
  const gpu_table& input,
  const duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& groups,
  const duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& aggregates,
  const duckdb::vector<duckdb::LogicalType>& result_types,
  gpu_table& output)
{
  RASTERDB_LOG_DEBUG("GPU execute_grouped_aggregate: {} groups, {} aggs",
                     groups.size(), aggregates.size());

  size_t num_group_cols = groups.size();
  if (num_group_cols < 1 || num_group_cols > 3) {
    throw duckdb::NotImplementedException(
      "RasterDB GPU: GROUP BY supports 1-3 columns, got %zu", num_group_cols);
  }

  // Extract group column indices and validate
  std::vector<duckdb::idx_t> group_col_indices;
  for (size_t g = 0; g < num_group_cols; g++) {
    auto& group_expr = *groups[g];
    if (group_expr.type != duckdb::ExpressionType::BOUND_REF) {
      throw duckdb::NotImplementedException(
        "RasterDB GPU: GROUP BY expression must be a column reference");
    }
    group_col_indices.push_back(
      group_expr.Cast<duckdb::BoundReferenceExpression>().index);
  }

  RASTERDB_LOG_DEBUG("GROUP BY {} cols, {} rows", num_group_cols, input.num_rows());

  // Build the effective group key column (single or composite)
  auto n_rows = input.num_rows();
  gpu_column composite_key_storage;   // owns memory for multi-col case
  const gpu_column* key_col_ptr;
  bool composite_is_int64 = false;   // tracks which decomposition M to use
  int64_t decompose_base1 = 0;      // base for 2-col, or middle base for 3-col
  int64_t decompose_base2 = 0;      // last base for 3-col (unused for 2-col)
  std::vector<int64_t> surrogate_id_to_composite;  // INT64 surrogate mapping

  if (num_group_cols == 1) {
    key_col_ptr = &input.col(group_col_indices[0]);
  } else {
    stage_timer tc("    groupby_composite_key");
    auto& disp = _ctx.dispatcher();
    uint32_t sz = static_cast<uint32_t>(n_rows);

    // Quick max-reduction on each group column to decide INT32 vs INT64
    int64_t max_composite_estimate = 1;
    std::vector<int64_t> max_vals(num_group_cols, 0);
    {
      for (size_t g = 0; g < num_group_cols; g++) {
        auto col_view = input.col(group_col_indices[g]).view();
        rasterdf::reduce_aggregation max_agg(rasterdf::aggregation_kind::MAX);
        auto max_s = rasterdf::reduce(col_view, max_agg,
            rasterdf::data_type{rasterdf::type_id::INT32},
            _ctx.vk_context(), _ctx.dispatcher(), _ctx.workspace_mr());
        int64_t mv = static_cast<int64_t>(max_s->as<int32_t>());
        max_vals[g] = mv;
        if (g == 0) {
          max_composite_estimate = mv;
        } else {
          max_composite_estimate = max_composite_estimate * GROUPBY_COMPOSITE_M_I32 + mv;
        }
      }
    }
    composite_is_int64 = (max_composite_estimate > INT32_MAX / 2);

    if (!composite_is_int64) {
      // ---- INT32 GPU path (fast) ----
      decompose_base1 = GROUPBY_COMPOSITE_M_I32;
      decompose_base2 = GROUPBY_COMPOSITE_M_I32;

      // Step 1: temp = col0 * M  (COL_SCALAR multiply)
      auto temp = allocate_column(_ctx, {rasterdf::type_id::INT32}, n_rows);
      {
        binary_op_push_constants pc{};
        pc.input_a = input.col(group_col_indices[0]).address();
        pc.input_b = 0;
        pc.output_addr = temp.address();
        pc.size = sz;
        pc.op = 2;  // MUL
        pc.scalar_val = GROUPBY_COMPOSITE_M_I32;
        pc.mode = 1; // COL_SCALAR
        pc.type_id = 0; // INT32
        pc.debug_mode = 0;
        disp.dispatch_binary_op(pc);
      }

      // Step 2: composite = temp + col1  (COL_COL add)
      composite_key_storage = allocate_column(_ctx, {rasterdf::type_id::INT32}, n_rows);
      {
        binary_op_push_constants pc{};
        pc.input_a = temp.address();
        pc.input_b = input.col(group_col_indices[1]).address();
        pc.output_addr = composite_key_storage.address();
        pc.size = sz;
        pc.op = 0;  // ADD
        pc.mode = 0; // COL_COL
        pc.type_id = 0; // INT32
        pc.debug_mode = 0;
        disp.dispatch_binary_op(pc);
      }

      if (num_group_cols == 3) {
        // Step 3: temp2 = composite * M
        auto temp2 = allocate_column(_ctx, {rasterdf::type_id::INT32}, n_rows);
        {
          binary_op_push_constants pc{};
          pc.input_a = composite_key_storage.address();
          pc.input_b = 0;
          pc.output_addr = temp2.address();
          pc.size = sz;
          pc.op = 2; // MUL
          pc.scalar_val = GROUPBY_COMPOSITE_M_I32;
          pc.mode = 1; // COL_SCALAR
          pc.type_id = 0; // INT32
          pc.debug_mode = 0;
          disp.dispatch_binary_op(pc);
        }
        // Step 4: composite = temp2 + col2
        {
          binary_op_push_constants pc{};
          pc.input_a = temp2.address();
          pc.input_b = input.col(group_col_indices[2]).address();
          pc.output_addr = composite_key_storage.address();
          pc.size = sz;
          pc.op = 0; // ADD
          pc.mode = 0; // COL_COL
          pc.type_id = 0; // INT32
          pc.debug_mode = 0;
          disp.dispatch_binary_op(pc);
        }
      }
      key_col_ptr = &composite_key_storage;
    } else {
      // ---- INT64 CPU path (safe for large keys) ----
      // Build reversible mixed-radix bases from per-column maxima.
      // 2-col: key = col0 * base1 + col1, where base1 = max(col1) + 1
      // 3-col: key = (col0 * base1 + col1) * base2 + col2,
      //        where base1 = max(col1) + 1, base2 = max(col2) + 1
      decompose_base1 = (num_group_cols >= 2) ? (max_vals[1] + 1) : 1;
      decompose_base2 = (num_group_cols >= 3) ? (max_vals[2] + 1) : 1;
      if (decompose_base1 <= 0 || decompose_base2 <= 0) {
        throw duckdb::NotImplementedException(
          "RasterDB GPU: non-positive GROUP BY base in INT64 composite path");
      }

      std::vector<std::vector<int32_t>> h_group_cols(num_group_cols);
      for (size_t g = 0; g < num_group_cols; g++) {
        h_group_cols[g].resize(n_rows);
        const_cast<rasterdf::device_buffer&>(input.col(group_col_indices[g]).data).copy_to_host(
            h_group_cols[g].data(), n_rows * sizeof(int32_t),
            _ctx.device(), _ctx.queue(), _ctx.command_pool());
      }

      std::vector<int64_t> h_composite(n_rows);
      for (rasterdf::size_type r = 0; r < n_rows; r++) {
        __int128 key = static_cast<int64_t>(h_group_cols[0][r]);
        if (num_group_cols == 2) {
          key = key * static_cast<__int128>(decompose_base1) +
                static_cast<int64_t>(h_group_cols[1][r]);
        } else {
          key = (key * static_cast<__int128>(decompose_base1) +
                 static_cast<int64_t>(h_group_cols[1][r])) *
                    static_cast<__int128>(decompose_base2) +
                static_cast<int64_t>(h_group_cols[2][r]);
        }
        if (key > static_cast<__int128>(std::numeric_limits<int64_t>::max()) ||
            key < static_cast<__int128>(std::numeric_limits<int64_t>::min())) {
          throw duckdb::NotImplementedException(
            "RasterDB GPU: INT64 composite key overflow in GROUP BY");
        }
        h_composite[r] = static_cast<int64_t>(key);
      }

      // INT64-key atomics are broken on this driver; remap to dense INT32 surrogates
      surrogate_id_to_composite.reserve(n_rows);
      std::unordered_map<int64_t, int32_t> composite_to_id;
      composite_to_id.reserve(static_cast<size_t>(n_rows) * 2);

      std::vector<int32_t> h_key_ids(n_rows);
      for (rasterdf::size_type r = 0; r < n_rows; r++) {
        int64_t key = h_composite[r];
        auto it = composite_to_id.find(key);
        if (it == composite_to_id.end()) {
          int32_t id = static_cast<int32_t>(surrogate_id_to_composite.size());
          composite_to_id.emplace(key, id);
          surrogate_id_to_composite.push_back(key);
          h_key_ids[r] = id;
        } else {
          h_key_ids[r] = it->second;
        }
      }

      fprintf(stderr, "[RDB_DEBUG] surrogate: %zu unique composite keys from %d rows\n",
              surrogate_id_to_composite.size(), (int)n_rows);
      fflush(stderr);

      composite_key_storage = allocate_column(_ctx, {rasterdf::type_id::INT32}, n_rows);
      composite_key_storage.data.copy_from_host(
          h_key_ids.data(), n_rows * sizeof(int32_t),
          _ctx.device(), _ctx.queue(), _ctx.command_pool());
      key_col_ptr = &composite_key_storage;
    }
  }

  // Build table_view for the effective group key
  auto group_key_view = key_col_ptr->view();
  std::vector<rasterdf::column_view> key_views = {group_key_view};
  rasterdf::table_view keys_tv(key_views);

  // Process each aggregate expression using GFXM mesh shader or compute shader
  // Result layout: [group_key_cols..., agg_cols...]

  bool keys_set = false;
  rasterdf::size_type num_groups_result = 0;

  if constexpr (USE_SIMPLE_GFX_AGGR) {
    // ── GFXM Mesh Shader Groupby (graphics-pipeline, mesh shader hash aggregation) ──
    // Initialize gfx engine
    rasterdf::gfx_groupby_engine_init(_ctx.vk_context());
    RASTERDB_LOG_DEBUG("     [GFXM] Engine initialized");

    for (duckdb::idx_t i = 0; i < aggregates.size(); i++) {
      auto& expr = aggregates[i]->Cast<duckdb::BoundAggregateExpression>();
      auto& fname = expr.function.name;

      bool is_count_star = false;

      if (expr.children.empty()) {
        is_count_star = (fname == "count" || fname == "count_star");
        if (!is_count_star) {
          throw duckdb::NotImplementedException(
            "RasterDB GPU: grouped aggregate '%s' with no children", fname.c_str());
        }
      }

      // Map aggregate name to gfxm agg_type (0=sum, 1=count, 2=min, 3=max, 4=mean)
      int gfxm_agg_type;
      if (fname == "sum" || fname == "sum_no_overflow") {
        gfxm_agg_type = 0;
      } else if (fname == "min") {
        gfxm_agg_type = 2;
      } else if (fname == "max") {
        gfxm_agg_type = 3;
      } else if (fname == "count" || fname == "count_star") {
        gfxm_agg_type = 1;
      } else if (fname == "avg" || fname == "mean") {
        gfxm_agg_type = 4;
      } else {
        throw duckdb::NotImplementedException(
          "RasterDB GPU: unsupported grouped aggregate '%s'", fname.c_str());
      }

      RASTERDB_LOG_DEBUG("     [GFXM] Aggregate %zu/%zu: %s (type=%d, count_star=%s)",
                         i+1, aggregates.size(), fname.c_str(), gfxm_agg_type,
                         is_count_star ? "true" : "false");

      // Get key and value device addresses
      VkDeviceAddress keys_addr = key_col_ptr->address();

      gpu_column val_temp;
      VkDeviceAddress values_addr = 0;
      if (!is_count_star) {
        val_temp = evaluate_expression(input, *expr.children[0]);
        values_addr = val_temp.address();
        RASTERDB_LOG_DEBUG("     [GFXM] Value column evaluated, addr=0x%lx", values_addr);
      }

      uint32_t n = static_cast<uint32_t>(input.num_rows());
      RASTERDB_LOG_DEBUG("     [GFXM] Input rows: %u", n);

      // Call gfxm groupby - use INT64 version for multi-column composite keys
      rasterdf::device_buffer out_keys(_ctx.workspace_mr(), 0,
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
      rasterdf::device_buffer out_values(_ctx.workspace_mr(), 0,
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
      uint32_t out_num_groups = 0;

      auto t_gfxm_start = std::chrono::high_resolution_clock::now();
      
      // Use INT64 mesh shaders for multi-column GROUP BY (composite keys)
      if (num_group_cols > 1) {
        rasterdf::gfxm_groupby_aggregate_int64key(
            gfxm_agg_type,
            keys_addr, values_addr,
            n,
            _ctx.dispatcher(),
            _ctx.workspace_mr(),
            out_keys, out_values,
            out_num_groups);
      } else {
        rasterdf::gfxm_groupby_aggregate(
            gfxm_agg_type,
            keys_addr, values_addr,
            n,
            _ctx.dispatcher(),
            _ctx.workspace_mr(),
            out_keys, out_values,
            out_num_groups);
      }
      
      auto t_gfxm_end = std::chrono::high_resolution_clock::now();
      RASTERDB_LOG_DEBUG("     [GFXM] gfxm_groupby_aggregate time: %.2f ms",
                         std::chrono::duration<double, std::milli>(t_gfxm_end - t_gfxm_start).count());

      if (out_num_groups == 0) {
        throw duckdb::NotImplementedException(
          "RasterDB GPU: gfxm groupby produced zero groups");
      }

      RASTERDB_LOG_DEBUG("     [GFXM] Output groups: %u", out_num_groups);

      // Convert device buffers to rasterdf columns
      rasterdf::data_type key_type = (num_group_cols > 1) 
          ? rasterdf::data_type{rasterdf::type_id::INT64}  // Composite keys are INT64
          : key_col_ptr->type;
      rasterdf::column key_col_rdf(key_type, out_num_groups, std::move(out_keys));

      rasterdf::data_type val_type;
      if (gfxm_agg_type == 0) {
        val_type = rasterdf::data_type{rasterdf::type_id::INT64};
      } else if (gfxm_agg_type == 4) {
        val_type = rasterdf::data_type{rasterdf::type_id::FLOAT32};
      } else {
        val_type = rasterdf::data_type{rasterdf::type_id::INT32};
      }
      rasterdf::column val_col_rdf(val_type, out_num_groups, std::move(out_values));

      // Download keys for sorting
      size_t key_elem_size = rasterdf::size_of(key_col_rdf.type());
      bool keys_are_int64 = (key_col_rdf.type().id == rasterdf::type_id::INT64);
      RASTERDB_LOG_DEBUG("     [GFXM] Key type: %s, elem_size=%zu",
                         keys_are_int64 ? "INT64" : "INT32", key_elem_size);

      auto t_download_keys_start = std::chrono::high_resolution_clock::now();
      std::vector<uint8_t> h_keys_raw(out_num_groups * key_elem_size);
      key_col_rdf.device_data().copy_to_host(
          h_keys_raw.data(), out_num_groups * key_elem_size, 0,
          _ctx.device(), _ctx.queue(), _ctx.command_pool());
      auto t_download_keys_end = std::chrono::high_resolution_clock::now();
      RASTERDB_LOG_DEBUG("     [GFXM] Download keys time: %.2f ms",
                         std::chrono::duration<double, std::milli>(t_download_keys_end - t_download_keys_start).count());

      // Build a sort permutation (ascending by key)
      auto t_sort_start = std::chrono::high_resolution_clock::now();
      std::vector<size_t> perm(out_num_groups);
      std::iota(perm.begin(), perm.end(), 0);
      if (keys_are_int64) {
        auto* kp = reinterpret_cast<const int64_t*>(h_keys_raw.data());
        std::sort(perm.begin(), perm.end(),
                  [kp](size_t a, size_t b) { return kp[a] < kp[b]; });
      } else {
        auto* kp = reinterpret_cast<const int32_t*>(h_keys_raw.data());
        std::sort(perm.begin(), perm.end(),
                  [kp](size_t a, size_t b) { return kp[a] < kp[b]; });
      }
      auto t_sort_end = std::chrono::high_resolution_clock::now();
      RASTERDB_LOG_DEBUG("     [GFXM] Sort permutation time: %.2f ms",
                         std::chrono::duration<double, std::milli>(t_sort_end - t_sort_start).count());

      // Download value column to CPU, apply permutation, re-upload
      size_t val_elem_size = rasterdf::size_of(val_col_rdf.type());
      RASTERDB_LOG_DEBUG("     [GFXM] Value elem_size=%zu", val_elem_size);

      auto t_download_vals_start = std::chrono::high_resolution_clock::now();
      std::vector<uint8_t> h_vals(out_num_groups * val_elem_size);
      val_col_rdf.device_data().copy_to_host(
          h_vals.data(), out_num_groups * val_elem_size, 0,
          _ctx.device(), _ctx.queue(), _ctx.command_pool());
      auto t_download_vals_end = std::chrono::high_resolution_clock::now();
      RASTERDB_LOG_DEBUG("     [GFXM] Download values time: %.2f ms",
                         std::chrono::duration<double, std::milli>(t_download_vals_end - t_download_vals_start).count());

      // Apply permutation to keys and values
      auto t_permute_start = std::chrono::high_resolution_clock::now();
      std::vector<uint8_t> sorted_keys_raw(out_num_groups * key_elem_size);
      std::vector<uint8_t> sorted_vals(out_num_groups * val_elem_size);
      for (size_t j = 0; j < out_num_groups; j++) {
        std::memcpy(sorted_keys_raw.data() + j * key_elem_size,
                    h_keys_raw.data() + perm[j] * key_elem_size, key_elem_size);
        std::memcpy(sorted_vals.data() + j * val_elem_size,
                    h_vals.data() + perm[j] * val_elem_size, val_elem_size);
      }
      auto t_permute_end = std::chrono::high_resolution_clock::now();
      RASTERDB_LOG_DEBUG("     [GFXM] Apply permutation time: %.2f ms",
                         std::chrono::duration<double, std::milli>(t_permute_end - t_permute_start).count());

      // On first aggregate, store the sorted keys
      if (!keys_set) {
        num_groups_result = out_num_groups;
        if (num_group_cols == 1) {
          auto t_upload_keys_start = std::chrono::high_resolution_clock::now();
          auto sorted_key_col = allocate_column(_ctx, key_col_rdf.type(), out_num_groups);
          sorted_key_col.data.copy_from_host(
              sorted_keys_raw.data(), out_num_groups * key_elem_size,
              _ctx.device(), _ctx.queue(), _ctx.command_pool());
          auto t_upload_keys_end = std::chrono::high_resolution_clock::now();
          RASTERDB_LOG_DEBUG("     [GFXM] Upload sorted keys time: %.2f ms",
                             std::chrono::duration<double, std::milli>(t_upload_keys_end - t_upload_keys_start).count());
          output.columns[0] = std::move(sorted_key_col);
        } else {
          // For multi-col: decompose composite keys to individual INT32 cols on CPU
          RASTERDB_LOG_DEBUG("     [GFXM] Multi-column GROUP BY, decomposing composite keys");
          std::vector<int64_t> sorted_composite_i64(out_num_groups);
          if (keys_are_int64) {
            auto* p = reinterpret_cast<const int64_t*>(sorted_keys_raw.data());
            for (uint32_t j = 0; j < out_num_groups; j++) sorted_composite_i64[j] = p[j];
          } else if (!surrogate_id_to_composite.empty()) {
            // Surrogate INT32 keys → map back to INT64 composite
            auto* p = reinterpret_cast<const int32_t*>(sorted_keys_raw.data());
            for (uint32_t j = 0; j < out_num_groups; j++) {
              int32_t id = p[j];
              if (id < 0 || static_cast<size_t>(id) >= surrogate_id_to_composite.size()) {
                throw duckdb::NotImplementedException(
                  "RasterDB GPU: invalid surrogate GROUP BY key id %d", id);
              }
              sorted_composite_i64[j] = surrogate_id_to_composite[static_cast<size_t>(id)];
            }
          } else {
            auto* p = reinterpret_cast<const int32_t*>(sorted_keys_raw.data());
            for (uint32_t j = 0; j < out_num_groups; j++) sorted_composite_i64[j] = static_cast<int64_t>(p[j]);
          }

          if (num_group_cols == 2) {
            std::vector<int32_t> col0(out_num_groups), col1(out_num_groups);
            for (uint32_t j = 0; j < out_num_groups; j++) {
              col1[j] = static_cast<int32_t>(sorted_composite_i64[j] % decompose_base1);
              col0[j] = static_cast<int32_t>(sorted_composite_i64[j] / decompose_base1);
            }
            auto t_upload_keys_start = std::chrono::high_resolution_clock::now();
            output.columns[0] = allocate_column(_ctx, {rasterdf::type_id::INT32}, out_num_groups);
            output.columns[0].data.copy_from_host(col0.data(), out_num_groups * sizeof(int32_t),
                _ctx.device(), _ctx.queue(), _ctx.command_pool());
            output.columns[1] = allocate_column(_ctx, {rasterdf::type_id::INT32}, out_num_groups);
            output.columns[1].data.copy_from_host(col1.data(), out_num_groups * sizeof(int32_t),
                _ctx.device(), _ctx.queue(), _ctx.command_pool());
            auto t_upload_keys_end = std::chrono::high_resolution_clock::now();
            RASTERDB_LOG_DEBUG("     [GFXM] Upload decomposed keys time: %.2f ms",
                               std::chrono::duration<double, std::milli>(t_upload_keys_end - t_upload_keys_start).count());
          } else {
            // 3-col: composite = (col0 * M + col1) * M + col2
            std::vector<int32_t> col0(out_num_groups), col1(out_num_groups), col2(out_num_groups);
            for (uint32_t j = 0; j < out_num_groups; j++) {
              int64_t c = sorted_composite_i64[j];
              col2[j] = static_cast<int32_t>(c % decompose_base2);
              c /= decompose_base2;
              col1[j] = static_cast<int32_t>(c % decompose_base1);
              col0[j] = static_cast<int32_t>(c / decompose_base1);
            }
            auto t_upload_keys_start = std::chrono::high_resolution_clock::now();
            output.columns[0] = allocate_column(_ctx, {rasterdf::type_id::INT32}, out_num_groups);
            output.columns[0].data.copy_from_host(col0.data(), out_num_groups * sizeof(int32_t),
                _ctx.device(), _ctx.queue(), _ctx.command_pool());
            output.columns[1] = allocate_column(_ctx, {rasterdf::type_id::INT32}, out_num_groups);
            output.columns[1].data.copy_from_host(col1.data(), out_num_groups * sizeof(int32_t),
                _ctx.device(), _ctx.queue(), _ctx.command_pool());
            output.columns[2] = allocate_column(_ctx, {rasterdf::type_id::INT32}, out_num_groups);
            output.columns[2].data.copy_from_host(col2.data(), out_num_groups * sizeof(int32_t),
                _ctx.device(), _ctx.queue(), _ctx.command_pool());
            auto t_upload_keys_end = std::chrono::high_resolution_clock::now();
            RASTERDB_LOG_DEBUG("     [GFXM] Upload decomposed keys time: %.2f ms",
                               std::chrono::duration<double, std::milli>(t_upload_keys_end - t_upload_keys_start).count());
          }
        }
        keys_set = true;
      }

      // Create gpu_column for sorted values and upload
      size_t out_col_idx = num_group_cols + i;
      auto t_upload_vals_start = std::chrono::high_resolution_clock::now();
      auto sorted_val_col = allocate_column(_ctx, val_col_rdf.type(), out_num_groups);
      sorted_val_col.data.copy_from_host(
          sorted_vals.data(), out_num_groups * val_elem_size,
          _ctx.device(), _ctx.queue(), _ctx.command_pool());
      auto t_upload_vals_end = std::chrono::high_resolution_clock::now();
      RASTERDB_LOG_DEBUG("     [GFXM] Upload sorted values time: %.2f ms",
                         std::chrono::duration<double, std::milli>(t_upload_vals_end - t_upload_vals_start).count());
      output.columns[out_col_idx] = std::move(sorted_val_col);
    }

  } // end if constexpr (USE_SIMPLE_GFX_AGGR)
  else {
    // ── Compute Shader Groupby (rasterdf::groupby) ──
    RASTERDB_LOG_DEBUG("     [COMPUTE] Using compute shader groupby");

    for (duckdb::idx_t i = 0; i < aggregates.size(); i++) {
      auto& expr = aggregates[i]->Cast<duckdb::BoundAggregateExpression>();
      auto& fname = expr.function.name;

      bool is_count_star = false;

      if (expr.children.empty()) {
        is_count_star = (fname == "count" || fname == "count_star");
        if (!is_count_star) {
          throw duckdb::NotImplementedException(
            "RasterDB GPU: grouped aggregate '%s' with no children", fname.c_str());
        }
      }

      // Map aggregate name to rasterdf kind
      rasterdf::aggregation_kind kind;
      if (fname == "sum" || fname == "sum_no_overflow") {
        kind = rasterdf::aggregation_kind::SUM;
      } else if (fname == "min") {
        kind = rasterdf::aggregation_kind::MIN;
      } else if (fname == "max") {
        kind = rasterdf::aggregation_kind::MAX;
      } else if (fname == "count" || fname == "count_star") {
        kind = rasterdf::aggregation_kind::COUNT_ALL;
      } else if (fname == "avg" || fname == "mean") {
        kind = rasterdf::aggregation_kind::MEAN;
      } else {
        throw duckdb::NotImplementedException(
          "RasterDB GPU: unsupported grouped aggregate '%s'", fname.c_str());
      }

      // Build aggregation request
      rasterdf::groupby gb(keys_tv, _ctx.vk_context(), _ctx.dispatcher(), _ctx.workspace_mr());

      std::vector<rasterdf::aggregation_request> requests;
      rasterdf::aggregation_request req;

      // Evaluate the value expression (may be column ref or complex expression)
      gpu_column val_temp;  // keep alive for view validity
      if (is_count_star) {
        req.values = key_col_ptr->view();
      } else {
        val_temp = evaluate_expression(input, *expr.children[0]);
        req.values = val_temp.view();
      }
      req.aggregations.push_back(std::make_unique<rasterdf::groupby_aggregation>(kind));
      requests.push_back(std::move(req));

      auto agg_result = gb.aggregate(std::move(requests));

      // Extract keys + value from this aggregate call
      auto this_key_cols = agg_result.keys->extract();
      if (this_key_cols.empty() || agg_result.results.empty() || !agg_result.results[0]) {
        throw duckdb::NotImplementedException(
          "RasterDB GPU: grouped aggregate '%s' produced empty keys/results",
          fname.c_str());
      }
      auto ng = this_key_cols[0]->size();

      // Download keys to CPU for sorting (INT32 or INT64 depending on key type)
      auto& key_col_rdf = *this_key_cols[0];
      auto& val_col_rdf = *agg_result.results[0];
      size_t key_elem_size = rasterdf::size_of(key_col_rdf.type());
      bool keys_are_int64 = (key_col_rdf.type().id == rasterdf::type_id::INT64);

      std::vector<uint8_t> h_keys_raw(ng * key_elem_size);
      key_col_rdf.device_data().copy_to_host(
          h_keys_raw.data(), ng * key_elem_size, 0,
          _ctx.device(), _ctx.queue(), _ctx.command_pool());

      // Build a sort permutation (ascending by key)
      std::vector<size_t> perm(ng);
      std::iota(perm.begin(), perm.end(), 0);
      if (keys_are_int64) {
        auto* kp = reinterpret_cast<const int64_t*>(h_keys_raw.data());
        std::sort(perm.begin(), perm.end(),
                  [kp](size_t a, size_t b) { return kp[a] < kp[b]; });
      } else {
        auto* kp = reinterpret_cast<const int32_t*>(h_keys_raw.data());
        std::sort(perm.begin(), perm.end(),
                  [kp](size_t a, size_t b) { return kp[a] < kp[b]; });
      }

      // Download value column to CPU, apply permutation, re-upload
      size_t val_elem_size = rasterdf::size_of(val_col_rdf.type());
      std::vector<uint8_t> h_vals(ng * val_elem_size);
      val_col_rdf.device_data().copy_to_host(
          h_vals.data(), ng * val_elem_size, 0,
          _ctx.device(), _ctx.queue(), _ctx.command_pool());

      // Apply permutation to keys and values
      std::vector<uint8_t> sorted_keys_raw(ng * key_elem_size);
      std::vector<uint8_t> sorted_vals(ng * val_elem_size);
      for (size_t j = 0; j < ng; j++) {
        std::memcpy(sorted_keys_raw.data() + j * key_elem_size,
                    h_keys_raw.data() + perm[j] * key_elem_size, key_elem_size);
        std::memcpy(sorted_vals.data() + j * val_elem_size,
                    h_vals.data() + perm[j] * val_elem_size, val_elem_size);
      }

      // On first aggregate, store the sorted keys
      if (!keys_set) {
        num_groups_result = ng;
        if (num_group_cols == 1) {
          auto sorted_key_col = allocate_column(_ctx, key_col_rdf.type(), ng);
          sorted_key_col.data.copy_from_host(
              sorted_keys_raw.data(), ng * key_elem_size,
              _ctx.device(), _ctx.queue(), _ctx.command_pool());
          output.columns[0] = std::move(sorted_key_col);
        } else {
          // For multi-col: decompose composite keys to individual INT32 cols on CPU
          // Works for both INT32 and INT64 composite keys via int64_t arithmetic
          std::vector<int64_t> sorted_composite_i64(ng);
          if (keys_are_int64) {
            auto* p = reinterpret_cast<const int64_t*>(sorted_keys_raw.data());
            for (rasterdf::size_type j = 0; j < ng; j++) sorted_composite_i64[j] = p[j];
          } else if (!surrogate_id_to_composite.empty()) {
            // Surrogate INT32 keys → map back to INT64 composite
            auto* p = reinterpret_cast<const int32_t*>(sorted_keys_raw.data());
            for (rasterdf::size_type j = 0; j < ng; j++) {
              int32_t id = p[j];
              if (id < 0 || static_cast<size_t>(id) >= surrogate_id_to_composite.size()) {
                throw duckdb::NotImplementedException(
                  "RasterDB GPU: invalid surrogate GROUP BY key id %d", id);
              }
              sorted_composite_i64[j] = surrogate_id_to_composite[static_cast<size_t>(id)];
            }
          } else {
            auto* p = reinterpret_cast<const int32_t*>(sorted_keys_raw.data());
            for (rasterdf::size_type j = 0; j < ng; j++) sorted_composite_i64[j] = static_cast<int64_t>(p[j]);
          }

          if (num_group_cols == 2) {
            std::vector<int32_t> col0(ng), col1(ng);
            for (rasterdf::size_type j = 0; j < ng; j++) {
              col1[j] = static_cast<int32_t>(sorted_composite_i64[j] % decompose_base1);
              col0[j] = static_cast<int32_t>(sorted_composite_i64[j] / decompose_base1);
            }
            output.columns[0] = allocate_column(_ctx, {rasterdf::type_id::INT32}, ng);
            output.columns[0].data.copy_from_host(col0.data(), ng * sizeof(int32_t),
                _ctx.device(), _ctx.queue(), _ctx.command_pool());
            output.columns[1] = allocate_column(_ctx, {rasterdf::type_id::INT32}, ng);
            output.columns[1].data.copy_from_host(col1.data(), ng * sizeof(int32_t),
                _ctx.device(), _ctx.queue(), _ctx.command_pool());
          } else {
            // 3-col: composite = (col0 * M + col1) * M + col2
            std::vector<int32_t> col0(ng), col1(ng), col2(ng);
            for (rasterdf::size_type j = 0; j < ng; j++) {
              int64_t c = sorted_composite_i64[j];
              col2[j] = static_cast<int32_t>(c % decompose_base2);
              c /= decompose_base2;
              col1[j] = static_cast<int32_t>(c % decompose_base1);
              col0[j] = static_cast<int32_t>(c / decompose_base1);
            }
            output.columns[0] = allocate_column(_ctx, {rasterdf::type_id::INT32}, ng);
            output.columns[0].data.copy_from_host(col0.data(), ng * sizeof(int32_t),
                _ctx.device(), _ctx.queue(), _ctx.command_pool());
            output.columns[1] = allocate_column(_ctx, {rasterdf::type_id::INT32}, ng);
            output.columns[1].data.copy_from_host(col1.data(), ng * sizeof(int32_t),
                _ctx.device(), _ctx.queue(), _ctx.command_pool());
            output.columns[2] = allocate_column(_ctx, {rasterdf::type_id::INT32}, ng);
            output.columns[2].data.copy_from_host(col2.data(), ng * sizeof(int32_t),
                _ctx.device(), _ctx.queue(), _ctx.command_pool());
          }
        }
        keys_set = true;
      }

      // Create gpu_column for sorted values and upload
      size_t out_col_idx = num_group_cols + i;
      auto sorted_val_col = allocate_column(_ctx, val_col_rdf.type(), ng);
      sorted_val_col.data.copy_from_host(
          sorted_vals.data(), ng * val_elem_size,
          _ctx.device(), _ctx.queue(), _ctx.command_pool());
      output.columns[out_col_idx] = std::move(sorted_val_col);
    }
  }
  RASTERDB_LOG_DEBUG("GROUP BY result: {} groups, {} output cols",
                     num_groups_result, output.columns.size());
}

// ============================================================================
// ORDER BY — radix sort via rasterdf
// ============================================================================

std::unique_ptr<gpu_table> gpu_executor::execute_order(duckdb::LogicalOrder& op)
{
  RASTERDB_LOG_DEBUG("GPU execute_order");
  D_ASSERT(op.children.size() == 1);
  auto input = execute_operator(*op.children[0]);

  stage_timer t("  order_by");  // Timer starts AFTER child execution

  if (input->num_rows() <= 1) return input;

  // Build key views and order vectors for all sort columns
  std::vector<rasterdf::column_view> key_views;
  std::vector<rasterdf::order> col_order;
  std::vector<gpu_column> expr_temps;  // keep alive for evaluated expressions

  for (auto& order : op.orders) {
    auto& expr = unwrap_cast(*order.expression);
    if (expr.type == duckdb::ExpressionType::BOUND_REF) {
      auto& ref = expr.Cast<duckdb::BoundReferenceExpression>();
      key_views.push_back(input->col(ref.index).view());
    } else {
      // Evaluate expression to a temporary column
      expr_temps.push_back(evaluate_expression(*input, expr));
      key_views.push_back(expr_temps.back().view());
    }
    col_order.push_back(
      (order.type == duckdb::OrderType::DESCENDING)
        ? rasterdf::order::DESCENDING
        : rasterdf::order::ASCENDING);
  }
  rasterdf::table_view keys_tv(key_views);

  // Use rasterdf high-level API: sorted_order → gather
  auto indices = rasterdf::sorted_order(keys_tv, col_order,
                                        _ctx.vk_context(), _ctx.dispatcher(), _ctx.workspace_mr());
  auto indices_view = indices->view();

  // Gather all columns using sorted indices
  auto sorted_table = rasterdf::gather(input->view(), indices_view,
                                       _ctx.vk_context(), _ctx.dispatcher(), _ctx.workspace_mr());

  return gpu_table_from_rdf(std::move(sorted_table), input->duckdb_types);
}

// ============================================================================
// LIMIT — take first N rows
// ============================================================================

std::unique_ptr<gpu_table> gpu_executor::execute_limit(duckdb::LogicalLimit& op)
{
  RASTERDB_LOG_DEBUG("GPU execute_limit");
  D_ASSERT(op.children.size() == 1);
  auto input = execute_operator(*op.children[0]);

  stage_timer t("  limit");  // Timer starts AFTER child execution

  int64_t limit_val = 0;
  int64_t offset_val = 0;

  if (op.limit_val.Type() == duckdb::LimitNodeType::CONSTANT_VALUE) {
    limit_val = static_cast<int64_t>(op.limit_val.GetConstantValue());
  } else if (op.limit_val.Type() == duckdb::LimitNodeType::UNSET) {
    limit_val = input->num_rows();
  } else {
    throw duckdb::NotImplementedException("RasterDB GPU: expression LIMIT not supported");
  }

  if (op.offset_val.Type() == duckdb::LimitNodeType::CONSTANT_VALUE) {
    offset_val = static_cast<int64_t>(op.offset_val.GetConstantValue());
  }

  rasterdf::size_type start = static_cast<rasterdf::size_type>(
    std::min(offset_val, static_cast<int64_t>(input->num_rows())));
  rasterdf::size_type count = static_cast<rasterdf::size_type>(
    std::min(limit_val, static_cast<int64_t>(input->num_rows() - start)));

  if (count == input->num_rows() && start == 0) return input;

  auto& disp = _ctx.dispatcher();

  // Create index column [start, start+1, ..., start+count-1]
  auto indices = allocate_column(_ctx, {rasterdf::type_id::INT32}, count);
  uint32_t ng = div_ceil(static_cast<uint32_t>(count), WG_SIZE);
  {
    radix_init_indices_pc pc{};
    pc.indices_ptr = indices.address();
    pc.numElements = static_cast<uint32_t>(count);
    disp.dispatch_radix_init_indices(pc, ng);

    if (start > 0) {
      binary_op_push_constants bpc{};
      bpc.input_a = indices.address();
      bpc.output_addr = indices.address();
      bpc.size = static_cast<uint32_t>(count);
      bpc.op = 0; bpc.scalar_val = static_cast<int32_t>(start);
      bpc.mode = 1; bpc.type_id = 0; bpc.input_b = 0; bpc.debug_mode = 0;
      disp.dispatch_binary_op(bpc);
    }
  }

  auto result = std::make_unique<gpu_table>();
  result->duckdb_types = input->duckdb_types;
  result->columns.resize(input->num_columns());

  for (size_t c = 0; c < input->num_columns(); c++) {
    auto& in_col = input->col(c);
    result->columns[c] = allocate_column(_ctx, in_col.type, count);

    gather_indices_pc gc{};
    gc.input_addr = in_col.address();
    gc.indices_addr = indices.address();
    gc.output_addr = result->columns[c].address();
    gc.size = static_cast<uint32_t>(count);
    if (rdf_type_size(in_col.type.id) == 8) {
      disp.dispatch_gather_indices_64(gc, ng);
    } else {
      disp.dispatch_gather_indices(gc, ng);
    }
  }

  return result;
}

// ============================================================================
// JOIN — hash join via rasterdf
// ============================================================================

// Toggle between compute-shader hash join and graphics-pipeline simple garuda join
static constexpr bool USE_SIMPLE_GARUDA_JOIN = false;

std::unique_ptr<gpu_table> gpu_executor::execute_join(duckdb::LogicalComparisonJoin& op)
{
  RASTERDB_LOG_DEBUG("GPU execute_join (simple_garuda=%s)", USE_SIMPLE_GARUDA_JOIN ? "true" : "false");
  D_ASSERT(op.children.size() == 2);

  if (op.join_type != duckdb::JoinType::INNER) {
    throw duckdb::NotImplementedException(
      "RasterDB GPU: only INNER JOIN supported, got %s",
      duckdb::JoinTypeToString(op.join_type).c_str());
  }

  // Validate all conditions are equi-joins on column references
  fprintf(stderr, "[RDB_DEBUG] JOIN: %zu conditions\n", op.conditions.size());
  for (size_t ci = 0; ci < op.conditions.size(); ci++) {
    auto& c = op.conditions[ci];
    fprintf(stderr, "[RDB_DEBUG]   cond[%zu]: cmp=%s left=%s right=%s\n",
            ci, duckdb::ExpressionTypeToString(c.comparison).c_str(),
            c.left->ToString().c_str(), c.right->ToString().c_str());
  }
  fflush(stderr);
  for (auto& cond : op.conditions) {
    if (cond.comparison != duckdb::ExpressionType::COMPARE_EQUAL) {
      throw duckdb::NotImplementedException("RasterDB GPU: only equi-join supported");
    }
    auto& le = unwrap_cast(*cond.left);
    auto& re = unwrap_cast(*cond.right);
    if (le.type != duckdb::ExpressionType::BOUND_REF ||
        re.type != duckdb::ExpressionType::BOUND_REF) {
      throw duckdb::NotImplementedException(
        "RasterDB GPU: join conditions must be column references");
    }
  }

  // Execute both children
  auto left_table = execute_operator(*op.children[0]);
  auto right_table = execute_operator(*op.children[1]);

  stage_timer t("  join");  // Timer starts AFTER child execution

  RASTERDB_LOG_DEBUG("JOIN: left {} rows x {} cols, right {} rows x {} cols",
                     left_table->num_rows(), left_table->num_columns(),
                     right_table->num_rows(), right_table->num_columns());

  // Join on FIRST condition
  auto& cond0 = op.conditions[0];
  auto left_key_idx = unwrap_cast(*cond0.left).Cast<duckdb::BoundReferenceExpression>().index;
  auto right_key_idx = unwrap_cast(*cond0.right).Cast<duckdb::BoundReferenceExpression>().index;

  auto left_key_view = left_table->col(left_key_idx).view();
  auto right_key_view = right_table->col(right_key_idx).view();
  fprintf(stderr, "[RDB_DEBUG] JOIN keys: L col[%zu] addr=0x%lx size=%d, R col[%zu] addr=0x%lx size=%d\n",
          (size_t)left_key_idx, (unsigned long)left_key_view.data(), (int)left_key_view.size(),
          (size_t)right_key_idx, (unsigned long)right_key_view.data(), (int)right_key_view.size());
  fflush(stderr);

  std::unique_ptr<rasterdf::column> left_indices;
  std::unique_ptr<rasterdf::column> right_indices;
  rasterdf::size_type match_count = 0;

  if constexpr (USE_SIMPLE_GARUDA_JOIN) {
    // ── Simple Garuda Join (graphics-pipeline, vertex shader hash join) ──
    uint32_t left_n  = static_cast<uint32_t>(left_key_view.size());
    uint32_t right_n = static_cast<uint32_t>(right_key_view.size());

    auto sg_result = rasterdf::simple_garuda_inner_join(
      left_key_view.data(), left_n,
      right_key_view.data(), right_n,
      _ctx.vk_context(), _ctx.dispatcher(), _ctx.workspace_mr());

    match_count = static_cast<rasterdf::size_type>(sg_result.num_matches);

    if (match_count > 0) {
      left_indices = std::make_unique<rasterdf::column>(
        rasterdf::data_type{rasterdf::type_id::INT32}, match_count,
        std::move(*sg_result.left_indices));
      right_indices = std::make_unique<rasterdf::column>(
        rasterdf::data_type{rasterdf::type_id::INT32}, match_count,
        std::move(*sg_result.right_indices));
    }
  } else {
    // ── Compute-shader hash join ──
    std::vector<rasterdf::column_view> lk = {left_key_view};
    std::vector<rasterdf::column_view> rk = {right_key_view};
    rasterdf::table_view left_keys_tv(lk);
    rasterdf::table_view right_keys_tv(rk);

    auto join_result = rasterdf::inner_join(
      left_keys_tv, right_keys_tv,
      _ctx.vk_context(), _ctx.dispatcher(), _ctx.workspace_mr());

    left_indices = std::move(join_result.first);
    right_indices = std::move(join_result.second);
    match_count = left_indices ? left_indices->size() : 0;
  }
  RASTERDB_LOG_DEBUG("JOIN: {} matches after first condition", match_count);

  if (match_count == 0) {
    auto result = std::make_unique<gpu_table>();
    result->duckdb_types = op.types;
    result->columns.resize(op.types.size());
    for (size_t i = 0; i < op.types.size(); i++) {
      result->columns[i].type = to_rdf_type(op.types[i]);
      result->columns[i].num_rows = 0;
    }
    return result;
  }

  auto left_idx_view = left_indices->view();
  auto right_idx_view = right_indices->view();

  // Gather from both tables using the matched indices
  auto left_gathered = rasterdf::gather(left_table->view(), left_idx_view,
    _ctx.vk_context(), _ctx.dispatcher(), _ctx.workspace_mr());
  auto right_gathered = rasterdf::gather(right_table->view(), right_idx_view,
    _ctx.vk_context(), _ctx.dispatcher(), _ctx.workspace_mr());

  auto left_cols = left_gathered->extract();
  auto right_cols = right_gathered->extract();

  // Build result: left columns then right columns
  auto result = std::make_unique<gpu_table>();
  result->duckdb_types = op.types;

  size_t total_cols = left_cols.size() + right_cols.size();
  result->columns.resize(total_cols);
  for (size_t i = 0; i < left_cols.size(); i++) {
    result->columns[i] = gpu_column_from_rdf(std::move(*left_cols[i]));
  }
  for (size_t i = 0; i < right_cols.size(); i++) {
    result->columns[left_cols.size() + i] = gpu_column_from_rdf(std::move(*right_cols[i]));
  }

  RASTERDB_LOG_DEBUG("JOIN result: {} rows x {} cols", match_count, total_cols);

  // Post-filter on remaining conditions (multi-condition join)
  size_t num_left_cols = left_cols.size();
  for (size_t ci = 1; ci < op.conditions.size(); ci++) {
    auto& cond = op.conditions[ci];
    auto lk_idx = unwrap_cast(*cond.left).Cast<duckdb::BoundReferenceExpression>().index;
    auto rk_idx = unwrap_cast(*cond.right).Cast<duckdb::BoundReferenceExpression>().index;

    // In the merged table: left cols at [0..num_left-1], right cols at [num_left..]
    auto& left_key_col = result->col(lk_idx);
    auto& right_key_col = result->col(num_left_cols + rk_idx);

    uint32_t n = static_cast<uint32_t>(result->num_rows());
    auto mask = allocate_column(_ctx, {rasterdf::type_id::INT32}, n);

    compare_columns_push_constants cpc{};
    cpc.input_a = left_key_col.address();
    cpc.input_b = right_key_col.address();
    cpc.output_addr = mask.address();
    cpc.size = n;
    cpc.op = 4; // EQ
    cpc.type_id = 0; // INT32
    _ctx.dispatcher().dispatch_compare_columns(cpc);

    result = apply_filter_mask(*result, mask);
    RASTERDB_LOG_DEBUG("JOIN: {} rows after condition {}", result->num_rows(), ci);
  }

  return result;
}

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
  fprintf(stderr, "[RDB_DEBUG]     to_query_result: %zu cols, %d rows\n",
          num_cols, (int)total_rows);
  for (size_t c = 0; c < num_cols; c++) {
    auto& col = table->col(c);
    fprintf(stderr, "[RDB_DEBUG]       col[%zu]: type_id=%d, num_rows=%d, is_host_only=%d, host_data_sz=%zu\n",
            c, static_cast<int>(col.type.id), (int)col.num_rows, (int)col.is_host_only,
            col.host_data.size());
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
      fprintf(stderr, "[RDB_DEBUG]     col[%zu] host-only, %zu bytes (no GPU download)\n",
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
  fprintf(stderr, "[RDB_DEBUG]     col_download: %zu cols, %zu bytes  %8.2f ms\n",
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
      size_t rdf_elem_size = rdf_type_size(table->col(c).type.id);
      size_t byte_offset = static_cast<size_t>(offset) * rdf_elem_size;
      const uint8_t* src = host_data[c].data() + byte_offset;
      auto dst = reinterpret_cast<uint8_t*>(chunk.data[c].GetData());

      auto rdf_tid = table->col(c).type.id;
      auto duckdb_tid = types[c].id();

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
