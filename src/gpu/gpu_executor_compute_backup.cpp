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
 */

#include "gpu/gpu_executor.hpp"
#include "gpu/gpu_buffer_manager.hpp"
#include "gpu/gpu_types.hpp"
#include "log/logging.hpp"

#include <rasterdf/execution/dispatcher.hpp>
#include <rasterdf/reduction.hpp>
#include <rasterdf/sorting.hpp>
#include <rasterdf/copying.hpp>
#include <rasterdf/stream_compaction.hpp>
#include <rasterdf/join.hpp>
#include <rasterdf/groupby.hpp>

#include <duckdb/common/exception.hpp>
#include <duckdb/common/types/data_chunk.hpp>
#include <duckdb/common/types/column/column_data_collection.hpp>
#include <duckdb/main/connection.hpp>
#include <duckdb/main/materialized_query_result.hpp>
#include <duckdb/planner/expression/bound_comparison_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_conjunction_expression.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <duckdb/planner/expression/bound_aggregate_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/expression/bound_cast_expression.hpp>
#include <duckdb/planner/bound_result_modifier.hpp>
#include <duckdb/catalog/catalog_entry/table_catalog_entry.hpp>

#include <duckdb/common/types/hugeint.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <vector>

namespace rasterdb {
namespace gpu {

using namespace rasterdf::execution;

static constexpr uint32_t WG_SIZE = 256;
static uint32_t div_ceil(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// Per-stage timing helper for debug-level timer logs.
struct stage_timer {
  const char* name;
  std::chrono::high_resolution_clock::time_point t0;
  stage_timer(const char* n) : name(n), t0(std::chrono::high_resolution_clock::now()) {}
  ~stage_timer() {
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    RASTERDB_LOG_DEBUG("[TIMER] {:<30s} {:8.2f} ms", name, ms);
  }
};

// ============================================================================
// Constructor
// ============================================================================

gpu_executor::gpu_executor(gpu_context& ctx, duckdb::ClientContext& client_ctx)
  : _ctx(ctx), _client_ctx(client_ctx) {}

// ============================================================================
// Top-level execute
// ============================================================================

std::unique_ptr<gpu_table> gpu_executor::execute(duckdb::LogicalOperator& plan)
{
  stage_timer t("TOTAL gpu_execute");

  // Reset the temporary workspace pool strictly for this query's execution
  _ctx.memory().reset_workspace();

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
        RASTERDB_LOG_DEBUG("[TIMER]   hint: LIMIT pushdown = {} rows", _scan_limit);
      }
    }
  }

  // Pattern 2: count(*) only — all aggregates are count_star, no group columns needed
  // AGGREGATE → [PROJECTION →] GET
  if (cur->type == duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
    auto& agg_op = cur->Cast<duckdb::LogicalAggregate>();
    if (agg_op.groups.empty() && !agg_op.expressions.empty()) {
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
        RASTERDB_LOG_DEBUG("[TIMER]   hint: count(*) only - scan 1 col");
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
        RASTERDB_LOG_DEBUG("[TIMER]   hint: count(*) only - scan 1 col");
      }
    }
  }
}

// ============================================================================
// Recursive operator dispatch
// ============================================================================

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

  // Use a DuckDB Connection to scan the table and get the data on CPU.
  // We build a simple SELECT from the table using its bind_data.
  // For table scans, the simplest approach is to use DuckDB's execution
  // engine to materialize the scan, then upload.

  // Get table name from the function name / bind data
  // For seq_scan, the table name is embedded in the bind data
  // For simplicity, we construct a SELECT * query for the table
  duckdb::unique_ptr<duckdb::Connection> conn;
  {
    auto t_conn = std::chrono::high_resolution_clock::now();
    conn = duckdb::make_uniq<duckdb::Connection>(*_client_ctx.db);
    auto t_conn_end = std::chrono::high_resolution_clock::now();
    RASTERDB_LOG_DEBUG("[RDB_DEBUG]     conn_alloc               {:8.2f} ms",
                       std::chrono::duration<double, std::milli>(t_conn_end - t_conn).count());
  }

  // Try to get table name from LogicalGet
  std::string table_name;
  auto table_entry = op.GetTable();
  if (table_entry) {
    table_name = table_entry->name;
  }

  if (table_name.empty()) {
    throw duckdb::NotImplementedException(
      "RasterDB GPU: cannot determine table name for scan '%s'",
      op.function.name.c_str());
  }

  // Build SELECT with projected columns (from column_ids) in order
  // Optimization: if count(*)-only, scan just 1 column to minimize I/O
  std::string col_list;
  if (_scan_count_star_only && !col_ids.empty()) {
    // Only scan the first column — we just need the row count
    auto col_idx = col_ids[0].GetPrimaryIndex();
    if (col_idx < op.names.size()) {
      col_list = "\"" + op.names[col_idx] + "\"";
    }
    // Override types to just the first column
    if (!types.empty()) {
      types = {types[0]};
    }
  } else {
    for (size_t i = 0; i < col_ids.size(); i++) {
      if (i > 0) col_list += ", ";
      auto col_idx = col_ids[i].GetPrimaryIndex();
      if (col_idx < op.names.size()) {
        col_list += "\"" + op.names[col_idx] + "\"";
      } else {
        col_list += "\"" + op.names[0] + "\"";
      }
    }
  }
  if (col_list.empty()) col_list = "*";

  std::string scan_query = "SELECT " + col_list + " FROM \"" + table_name + "\"";

  // Optimization: push down filters to Parquet scanner (requires optimizer)
  if (!op.table_filters.filters.empty()) {
    std::string where_clause = " WHERE ";
    bool first_filter = true;
    for (auto& entry : op.table_filters.filters) {
      if (!first_filter) where_clause += " AND ";
      
      idx_t absolute_col_idx = entry.first;
      std::string filter_col_name = "unknown";
      
      if (absolute_col_idx < op.names.size()) {
          filter_col_name = op.names[absolute_col_idx];
      } else {
          // If for some reason names is missing the absolute column,
          // try to match it through projected column_ids 
          for (size_t i = 0; i < col_ids.size(); i++) {
              if (col_ids[i].GetPrimaryIndex() == absolute_col_idx) {
                  if (i < op.names.size()) {
                      filter_col_name = op.names[i];
                  }
                  break;
              }
          }
      }
      
      where_clause += entry.second->ToString("\"" + filter_col_name + "\"");
      first_filter = false;
    }
    scan_query += where_clause;
  }

  // Optimization: LIMIT pushdown — add LIMIT to SQL scan if no filter/order
  if (_scan_limit > 0) {
    scan_query += " LIMIT " + std::to_string(_scan_limit);
  }

  RASTERDB_LOG_DEBUG("GPU scan: {}", scan_query);

  rasterdf::size_type total_scanned = 0;
  std::vector<std::unique_ptr<duckdb::DataChunk>> chunks;
  {
    stage_timer t_scan("  cpu_scan");
    auto t_qexec = std::chrono::high_resolution_clock::now();
    auto result = conn->Query(scan_query);
    if (result->HasError()) {
      throw std::runtime_error("GPU scan failed: " + result->GetError());
    }
    auto t_qexec_end = std::chrono::high_resolution_clock::now();
    RASTERDB_LOG_DEBUG("[RDB_DEBUG]     query_exec               {:8.2f} ms",
                       std::chrono::duration<double, std::milli>(t_qexec_end - t_qexec).count());

    auto t_fetch = std::chrono::high_resolution_clock::now();
    while (true) {
      auto chunk = result->Fetch();
      if (!chunk || chunk->size() == 0) break;
      chunks.push_back(std::move(chunk));
    }
    for (auto& c : chunks) total_scanned += static_cast<rasterdf::size_type>(c->size());
    auto t_fetch_end = std::chrono::high_resolution_clock::now();
    RASTERDB_LOG_DEBUG("[RDB_DEBUG]     chunk_fetch ({} chunks)   {:8.2f} ms",
                       chunks.size(),
                       std::chrono::duration<double, std::milli>(t_fetch_end - t_fetch).count());
  }
  RASTERDB_LOG_DEBUG("[TIMER]   scan: {} {} rows x {} cols",
                     table_name, total_scanned, types.size());

  std::unique_ptr<gpu_table> gpu_tbl;
  {
    stage_timer t_upload("  gpu_upload");
    auto t_up = std::chrono::high_resolution_clock::now();

    if (GPUBufferManager::is_initialized()) {
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
      // Reset staging bump pointer for this query
      GPUBufferManager::GetInstance().cpuProcessingPointer.store(0, std::memory_order_relaxed);
      gpu_tbl = gpu_table::from_buffer_manager(_ctx, table_name, col_names, types, chunks);
    } else {
      gpu_tbl = gpu_table::from_data_chunks(_ctx, types, chunks);
    }

    auto t_up_end = std::chrono::high_resolution_clock::now();
    size_t total_bytes = 0;
    for (size_t c = 0; c < gpu_tbl->num_columns(); c++) {
      total_bytes += gpu_tbl->col(c).byte_size();
    }
    RASTERDB_LOG_DEBUG("[RDB_DEBUG]     gpu_upload_detail: {} cols, {} bytes, {:8.2f} ms",
                       gpu_tbl->num_columns(), total_bytes,
                       std::chrono::duration<double, std::milli>(t_up_end - t_up).count());
  }
  return gpu_tbl;
}

// ============================================================================
// FILTER — evaluate WHERE clause predicates on GPU
std::unique_ptr<gpu_table> gpu_executor::execute_filter(duckdb::LogicalFilter& op)
{
  RASTERDB_LOG_DEBUG("GPU execute_filter");
  D_ASSERT(op.children.size() == 1);
  auto input = execute_operator(*op.children[0]);

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
    RASTERDB_LOG_DEBUG("[TIMER]     filter_compare             {:8.2f} ms", ms);
  }

  // Apply the mask to compact the table
  auto t_compact_start = std::chrono::high_resolution_clock::now();
  auto result = apply_filter_mask(*input, mask);
  {
    auto t_compact_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_compact_end - t_compact_start).count();
    RASTERDB_LOG_DEBUG("[TIMER]     filter_compact             {:8.2f} ms", ms);
  }
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
    disp.dispatch_scatter_if(spc, scatter_groups);
  }

  return result;
}

// ============================================================================
// Helper: unwrap BoundCastExpression to find the inner expression.
// DuckDB's optimizer inserts casts for type promotion (e.g., val > 5 becomes
// CAST(val AS BIGINT) > CAST(5 AS BIGINT)). We strip these to find the
// underlying column ref or constant.
// ============================================================================

static duckdb::Expression& unwrap_cast(duckdb::Expression& expr) {
  if (expr.expression_class == duckdb::ExpressionClass::BOUND_CAST) {
    auto& cast = expr.Cast<duckdb::BoundCastExpression>();
    return unwrap_cast(*cast.child);
  }
  return expr;
}

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
  auto& left = *func.children[0];
  auto& right = *func.children[1];

  rasterdf::data_type out_type = to_rdf_type(func.return_type);
  int32_t type_id = rdf_shader_type_id(out_type.id);
  auto result = allocate_column(_ctx, out_type, input.num_rows());

  binary_op_push_constants pc{};
  pc.output_addr = result.address();
  pc.size = static_cast<uint32_t>(input.num_rows());
  pc.op = op_code;
  pc.debug_mode = 0;
  pc.type_id = type_id;

  if (left.type == duckdb::ExpressionType::BOUND_REF &&
      right.type == duckdb::ExpressionType::BOUND_REF) {
    auto& l = left.Cast<duckdb::BoundReferenceExpression>();
    auto& r = right.Cast<duckdb::BoundReferenceExpression>();
    pc.input_a = input.col(l.index).address();
    pc.input_b = input.col(r.index).address();
    pc.mode = 0; // COL_COL
    pc.scalar_val = 0;
  } else if (left.type == duckdb::ExpressionType::BOUND_REF &&
             right.type == duckdb::ExpressionType::VALUE_CONSTANT) {
    auto& l = left.Cast<duckdb::BoundReferenceExpression>();
    auto& c = right.Cast<duckdb::BoundConstantExpression>();
    pc.input_a = input.col(l.index).address();
    pc.input_b = 0;
    pc.mode = 1; // COL_SCALAR
    if (type_id == 0) {
      pc.scalar_val = c.value.GetValue<int32_t>();
    } else {
      float fval = c.value.GetValue<float>();
      std::memcpy(&pc.scalar_val, &fval, sizeof(float));
    }
  } else {
    throw duckdb::NotImplementedException(
      "RasterDB GPU: unsupported binary_op operand types");
  }

  _ctx.dispatcher().dispatch_binary_op(pc);
  return result;
}

// ============================================================================
// AGGREGATE (ungrouped only for now)
// ============================================================================

std::unique_ptr<gpu_table> gpu_executor::execute_aggregate(duckdb::LogicalAggregate& op)
{
  stage_timer t("  aggregate");
  RASTERDB_LOG_DEBUG("GPU execute_aggregate");
  D_ASSERT(op.children.size() == 1);
  auto input = execute_operator(*op.children[0]);

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

    duckdb::idx_t col_idx = 0;
    bool is_count_star = false;

    if (expr.children.empty()) {
      is_count_star = (fname == "count" || fname == "count_star");
      if (!is_count_star) {
        throw duckdb::NotImplementedException(
          "RasterDB GPU: aggregate '%s' with no children", fname.c_str());
      }
    } else if (expr.children[0]->type == duckdb::ExpressionType::BOUND_REF) {
      col_idx = expr.children[0]->Cast<duckdb::BoundReferenceExpression>().index;
    } else {
      throw duckdb::NotImplementedException(
        "RasterDB GPU: aggregate with complex child expression");
    }

    if (is_count_star) {
      // count(*) — store directly as host-side scalar (no GPU needed)
      output.columns[i].type = {rasterdf::type_id::INT64};
      output.columns[i].num_rows = 1;
      output.columns[i].is_host_only = true;
      output.columns[i].host_data.resize(sizeof(int64_t));
      int64_t count = static_cast<int64_t>(n);
      std::memcpy(output.columns[i].host_data.data(), &count, sizeof(int64_t));
      RASTERDB_LOG_DEBUG("[RDB_DEBUG]     count_star = {} (host-only)", count);
      continue;
    }

    auto& in_col = input.col(col_idx);
    auto col_view = in_col.view();

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
    auto scalar = rasterdf::reduce(col_view, agg, in_col.type,
                                   _ctx.vk_context(), _ctx.dispatcher(), _ctx.workspace_mr());
    auto t_reduce_end = std::chrono::high_resolution_clock::now();
    RASTERDB_LOG_DEBUG("[RDB_DEBUG]     reduce({}) col={}      {:8.2f} ms",
                       fname, static_cast<size_t>(col_idx),
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
    RASTERDB_LOG_DEBUG("[RDB_DEBUG]     agg({}) stored host-only (no GPU alloc)", fname);
  }
}

// ============================================================================
// GROUP BY aggregate — hash-based groupby via rasterdf
// ============================================================================

void gpu_executor::execute_grouped_aggregate(
  const gpu_table& input,
  const duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& groups,
  const duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& aggregates,
  const duckdb::vector<duckdb::LogicalType>& result_types,
  gpu_table& output)
{
  RASTERDB_LOG_DEBUG("GPU execute_grouped_aggregate: {} groups, {} aggs",
                     groups.size(), aggregates.size());

  // Currently support single group-by column (INT32)
  if (groups.size() != 1) {
    throw duckdb::NotImplementedException(
      "RasterDB GPU: only single-column GROUP BY supported");
  }

  auto& group_expr = *groups[0];
  if (group_expr.type != duckdb::ExpressionType::BOUND_REF) {
    throw duckdb::NotImplementedException(
      "RasterDB GPU: GROUP BY expression must be a column reference");
  }

  auto group_col_idx = group_expr.Cast<duckdb::BoundReferenceExpression>().index;
  auto& group_col = input.col(group_col_idx);

  // Build table_view for group key
  auto group_key_view = group_col.view();
  std::vector<rasterdf::column_view> key_views = {group_key_view};
  rasterdf::table_view keys_tv(key_views);

  // Process each aggregate expression using rasterdf::groupby
  // Result layout: [group_key_col, agg1_col, agg2_col, ...]
  // Column index 0 = group key, 1..N = aggregates

  // We process one aggregate at a time since rasterdf::groupby reinitializes
  // the hash table per aggregate() call.
  // First, handle the first aggregate to get the keys.

  bool keys_set = false;
  rasterdf::size_type num_groups = 0;

  for (duckdb::idx_t i = 0; i < aggregates.size(); i++) {
    auto& expr = aggregates[i]->Cast<duckdb::BoundAggregateExpression>();
    auto& fname = expr.function.name;

    bool is_count_star = false;
    duckdb::idx_t val_col_idx = 0;

    if (expr.children.empty()) {
      is_count_star = (fname == "count" || fname == "count_star");
      if (!is_count_star) {
        throw duckdb::NotImplementedException(
          "RasterDB GPU: grouped aggregate '%s' with no children", fname.c_str());
      }
    } else if (expr.children[0]->type == duckdb::ExpressionType::BOUND_REF) {
      val_col_idx = expr.children[0]->Cast<duckdb::BoundReferenceExpression>().index;
    } else {
      throw duckdb::NotImplementedException(
        "RasterDB GPU: grouped aggregate with complex child expression");
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

    if (is_count_star) {
      // For count(*), use the group key column as values (counts rows per group)
      req.values = group_col.view();
    } else {
      req.values = input.col(val_col_idx).view();
    }
    req.aggregations.push_back(std::make_unique<rasterdf::groupby_aggregation>(kind));
    requests.push_back(std::move(req));

    auto agg_result = gb.aggregate(std::move(requests));

    // On first aggregate, extract the keys
    if (!keys_set) {
      auto key_cols = agg_result.keys->extract();
      if (!key_cols.empty()) {
        num_groups = key_cols[0]->size();
        output.columns[0] = gpu_column_from_rdf(std::move(*key_cols[0]));
      }
      keys_set = true;
    }

    // Store aggregate result (column index = groups.size() + i)
    size_t out_col_idx = groups.size() + i;
    if (!agg_result.results.empty() && agg_result.results[0]) {
      output.columns[out_col_idx] = gpu_column_from_rdf(std::move(*agg_result.results[0]));
    }
  }

  RASTERDB_LOG_DEBUG("GROUP BY result: {} groups, {} output cols",
                     num_groups, output.columns.size());
}

// ============================================================================
// ORDER BY — radix sort via rasterdf
// ============================================================================

std::unique_ptr<gpu_table> gpu_executor::execute_order(duckdb::LogicalOrder& op)
{
  stage_timer t("  order_by");
  RASTERDB_LOG_DEBUG("GPU execute_order");
  D_ASSERT(op.children.size() == 1);
  auto input = execute_operator(*op.children[0]);

  if (input->num_rows() <= 1) return input;

  if (op.orders.size() != 1) {
    throw duckdb::NotImplementedException(
      "RasterDB GPU: multi-column ORDER BY not yet implemented");
  }

  auto& order = op.orders[0];
  if (order.expression->type != duckdb::ExpressionType::BOUND_REF) {
    throw duckdb::NotImplementedException(
      "RasterDB GPU: ORDER BY non-column expression not supported");
  }

  auto& ref = order.expression->Cast<duckdb::BoundReferenceExpression>();

  // Build a single-column table_view for the sort key
  auto sort_key_view = input->col(ref.index).view();
  std::vector<rasterdf::column_view> key_views = {sort_key_view};
  rasterdf::table_view keys_tv(key_views);

  // Determine sort order
  std::vector<rasterdf::order> col_order = {
    (order.type == duckdb::OrderType::DESCENDING)
      ? rasterdf::order::DESCENDING
      : rasterdf::order::ASCENDING
  };

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
  stage_timer t("  limit");
  RASTERDB_LOG_DEBUG("GPU execute_limit");
  D_ASSERT(op.children.size() == 1);
  auto input = execute_operator(*op.children[0]);

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
    disp.dispatch_gather_indices(gc, ng);
  }

  return result;
}

// ============================================================================
// JOIN — hash join via rasterdf
// ============================================================================

std::unique_ptr<gpu_table> gpu_executor::execute_join(duckdb::LogicalComparisonJoin& op)
{
  stage_timer t("  join");
  RASTERDB_LOG_DEBUG("GPU execute_join");
  D_ASSERT(op.children.size() == 2);

  if (op.join_type != duckdb::JoinType::INNER) {
    throw duckdb::NotImplementedException(
      "RasterDB GPU: only INNER JOIN supported, got %s",
      duckdb::JoinTypeToString(op.join_type).c_str());
  }

  if (op.conditions.size() != 1) {
    throw duckdb::NotImplementedException(
      "RasterDB GPU: only single-condition JOIN supported");
  }

  auto& cond = op.conditions[0];
  if (cond.comparison != duckdb::ExpressionType::COMPARE_EQUAL) {
    throw duckdb::NotImplementedException(
      "RasterDB GPU: only equi-join supported");
  }

  // Execute both children
  auto left_table = execute_operator(*op.children[0]);
  auto right_table = execute_operator(*op.children[1]);

  RASTERDB_LOG_DEBUG("JOIN: left {} rows x {} cols, right {} rows x {} cols",
                     left_table->num_rows(), left_table->num_columns(),
                     right_table->num_rows(), right_table->num_columns());

  // Get join key column indices from conditions
  auto& left_expr = unwrap_cast(*cond.left);
  auto& right_expr = unwrap_cast(*cond.right);

  if (left_expr.type != duckdb::ExpressionType::BOUND_REF ||
      right_expr.type != duckdb::ExpressionType::BOUND_REF) {
    throw duckdb::NotImplementedException(
      "RasterDB GPU: join conditions must be column references");
  }

  auto left_key_idx = left_expr.Cast<duckdb::BoundReferenceExpression>().index;
  auto right_key_idx = right_expr.Cast<duckdb::BoundReferenceExpression>().index;

  // Build table_views for join keys (single-column INT32 keys)
  auto left_key_view = left_table->col(left_key_idx).view();
  auto right_key_view = right_table->col(right_key_idx).view();
  std::vector<rasterdf::column_view> lk = {left_key_view};
  std::vector<rasterdf::column_view> rk = {right_key_view};
  rasterdf::table_view left_keys_tv(lk);
  rasterdf::table_view right_keys_tv(rk);

  // Perform hash join — builds on right (build), probes with left (probe)
  auto [left_indices, right_indices] = rasterdf::inner_join(
    left_keys_tv, right_keys_tv,
    _ctx.vk_context(), _ctx.dispatcher(), _ctx.workspace_mr());

  rasterdf::size_type match_count = left_indices->size();
  RASTERDB_LOG_DEBUG("JOIN: {} matches", match_count);

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

  // Build result: left columns then right columns
  auto result = std::make_unique<gpu_table>();
  result->duckdb_types = op.types;

  auto left_cols = left_gathered->extract();
  auto right_cols = right_gathered->extract();

  size_t total_cols = left_cols.size() + right_cols.size();
  result->columns.resize(total_cols);
  for (size_t i = 0; i < left_cols.size(); i++) {
    result->columns[i] = gpu_column_from_rdf(std::move(*left_cols[i]));
  }
  for (size_t i = 0; i < right_cols.size(); i++) {
    result->columns[left_cols.size() + i] = gpu_column_from_rdf(std::move(*right_cols[i]));
  }

  RASTERDB_LOG_DEBUG("JOIN result: {} rows x {} cols", match_count, total_cols);
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
  RASTERDB_LOG_DEBUG("[RDB_DEBUG]     to_query_result: {} cols, {} rows",
                     num_cols, total_rows);
  for (size_t c = 0; c < num_cols; c++) {
    RASTERDB_LOG_DEBUG("[RDB_DEBUG]       col[{}]: num_rows={}, is_host_only={}, host_data_sz={}, byte_size={}",
                       c, table->col(c).num_rows, table->col(c).is_host_only,
                       table->col(c).host_data.size(), table->col(c).byte_size());
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
