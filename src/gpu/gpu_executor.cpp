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
#include "gpu/gpu_types.hpp"
#include "log/logging.hpp"

#include <rasterdf/execution/dispatcher.hpp>

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

#include <algorithm>
#include <chrono>
#include <cstring>
#include <vector>

namespace rasterdb {
namespace gpu {

using namespace rasterdf::execution;

static constexpr uint32_t WG_SIZE = 256;
static uint32_t div_ceil(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

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
  auto start = std::chrono::high_resolution_clock::now();
  auto result = execute_operator(plan);
  auto end = std::chrono::high_resolution_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
  RASTERDB_LOG_INFO("GPU execution completed in {:.2f} ms", ms);
  return result;
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
  auto conn = duckdb::make_uniq<duckdb::Connection>(*_client_ctx.db);

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
  std::string col_list;
  for (size_t i = 0; i < col_ids.size(); i++) {
    if (i > 0) col_list += ", ";
    auto col_idx = col_ids[i].GetPrimaryIndex();
    if (col_idx < op.names.size()) {
      col_list += "\"" + op.names[col_idx] + "\"";
    } else {
      col_list += "\"" + op.names[0] + "\"";
    }
  }
  if (col_list.empty()) col_list = "*";

  std::string scan_query = "SELECT " + col_list + " FROM \"" + table_name + "\"";
  RASTERDB_LOG_DEBUG("GPU scan: {}", scan_query);
  auto result = conn->Query(scan_query);
  if (result->HasError()) {
    throw std::runtime_error("GPU scan failed: " + result->GetError());
  }

  // Collect all chunks
  std::vector<std::unique_ptr<duckdb::DataChunk>> chunks;
  while (true) {
    auto chunk = result->Fetch();
    if (!chunk || chunk->size() == 0) break;
    chunks.push_back(std::move(chunk));
  }

  rasterdf::size_type total_scanned = 0;
  for (auto& c : chunks) total_scanned += static_cast<rasterdf::size_type>(c->size());
  RASTERDB_LOG_DEBUG("Scanned {} chunks ({} rows, {} cols) from '{}'",
                     chunks.size(), total_scanned, types.size(), table_name);
  return gpu_table::from_data_chunks(_ctx, types, chunks);
}

// ============================================================================
// FILTER — evaluate WHERE clause predicates on GPU
// ============================================================================

std::unique_ptr<gpu_table> gpu_executor::execute_filter(duckdb::LogicalFilter& op)
{
  RASTERDB_LOG_DEBUG("GPU execute_filter");
  D_ASSERT(op.children.size() == 1);
  auto input = execute_operator(*op.children[0]);

  if (input->num_rows() == 0) return input;

  // Evaluate each filter expression to produce a boolean (int32 0/1) mask
  gpu_column mask;
  bool first = true;

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

  // Apply the mask to compact the table
  return apply_filter_mask(*input, mask);
}

// ============================================================================
// Apply boolean mask — prefix scan + scatter (stream compaction)
// ============================================================================

std::unique_ptr<gpu_table> gpu_executor::apply_filter_mask(const gpu_table& input, gpu_column& mask)
{
  uint32_t n = static_cast<uint32_t>(input.num_rows());
  auto& disp = _ctx.dispatcher();

  // Copy mask into scan buffer for prefix scan (scan operates in-place)
  auto scan_buf = allocate_column(_ctx, {rasterdf::type_id::INT32},
                                   static_cast<rasterdf::size_type>(n + 1));
  {
    binary_op_push_constants pc{};
    pc.input_a = mask.address();
    pc.input_b = 0;
    pc.output_addr = scan_buf.address();
    pc.size = n;
    pc.op = 0; pc.scalar_val = 0; pc.mode = 1; pc.debug_mode = 0; pc.type_id = 0;
    disp.dispatch_binary_op(pc);
  }

  // Run 3-pass exclusive prefix scan
  uint32_t num_blocks = std::max(1u, div_ceil(n, WG_SIZE * 2));
  auto block_sums = allocate_buffer(_ctx, (num_blocks + 1) * sizeof(uint32_t));
  auto total_sum_buf = allocate_buffer(_ctx, sizeof(uint32_t));

  {
    prefix_scan_pc pc{};
    pc.data_ptr = scan_buf.address();
    pc.block_sums_ptr = block_sums.data();
    pc.total_sum_ptr = total_sum_buf.data();
    pc.numElements = n;
    pc.blockCount = num_blocks;
    disp.dispatch_prefix_scan_local(pc, num_blocks);
    if (num_blocks > 1) {
      disp.dispatch_prefix_scan_global(pc);
      disp.dispatch_prefix_scan_add(pc, num_blocks);
    }
  }

  // Read total count: for exclusive prefix scan, total = scan[n-1] + mask[n-1]
  uint32_t output_count = 0;
  {
    std::vector<int32_t> scan_vals(n + 1);
    const_cast<rasterdf::device_buffer&>(scan_buf.data).copy_to_host(
      scan_vals.data(), (n + 1) * sizeof(int32_t),
      _ctx.device(), _ctx.queue(), _ctx.command_pool());
    std::vector<int32_t> mask_vals(n);
    const_cast<rasterdf::device_buffer&>(mask.data).copy_to_host(
      mask_vals.data(), n * sizeof(int32_t),
      _ctx.device(), _ctx.queue(), _ctx.command_pool());
    output_count = static_cast<uint32_t>(scan_vals[n - 1] + mask_vals[n - 1]);
  }

  RASTERDB_LOG_DEBUG("Filter: {} -> {} rows", n, output_count);

  if (output_count == 0) {
    auto result = std::make_unique<gpu_table>();
    result->duckdb_types = input.duckdb_types;
    result->columns.resize(input.num_columns());
    for (size_t c = 0; c < input.num_columns(); c++) {
      result->columns[c].type = input.col(c).type;
      result->columns[c].num_rows = 0;
    }
    return result;
  }

  // Gather each column using mask + prefix-scanned offsets
  auto result = std::make_unique<gpu_table>();
  result->duckdb_types = input.duckdb_types;
  result->columns.resize(input.num_columns());

  for (size_t c = 0; c < input.num_columns(); c++) {
    auto& in_col = input.col(c);
    result->columns[c] = allocate_column(_ctx, in_col.type,
                                          static_cast<rasterdf::size_type>(output_count));
    gather_push_constants gc{};
    gc.input_addr = in_col.address();
    gc.mask_addr = mask.address();
    gc.output_addr = result->columns[c].address();
    gc.counter_addr = scan_buf.address();
    gc.size = n;
    disp.dispatch_gather(gc);
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

  auto result = std::make_unique<gpu_table>();
  result->duckdb_types = op.types;
  result->columns.resize(op.expressions.size());

  for (size_t i = 0; i < op.expressions.size(); i++) {
    auto& expr = *op.expressions[i];

    if (expr.type == duckdb::ExpressionType::BOUND_REF) {
      auto& ref = expr.Cast<duckdb::BoundReferenceExpression>();
      auto& src = input->col(ref.index);
      // Copy column to output
      result->columns[i] = allocate_column(_ctx, src.type, src.num_rows);
      binary_op_push_constants pc{};
      pc.input_a = src.address();
      pc.input_b = 0;
      pc.output_addr = result->columns[i].address();
      pc.size = static_cast<uint32_t>(src.num_rows);
      pc.op = 0; pc.scalar_val = 0; pc.mode = 1; pc.debug_mode = 0;
      pc.type_id = rdf_shader_type_id(src.type.id);
      _ctx.dispatcher().dispatch_binary_op(pc);
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
  RASTERDB_LOG_DEBUG("GPU execute_aggregate");
  D_ASSERT(op.children.size() == 1);
  auto input = execute_operator(*op.children[0]);

  if (!op.groups.empty()) {
    throw duckdb::NotImplementedException(
      "RasterDB GPU: GROUP BY not yet implemented");
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
  auto& disp = _ctx.dispatcher();
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
      output.columns[i] = allocate_column(_ctx, {rasterdf::type_id::INT64}, 1);
      int64_t count = static_cast<int64_t>(n);
      output.columns[i].data.copy_from_host(&count, sizeof(int64_t),
                                              _ctx.device(), _ctx.queue(), _ctx.command_pool());
      continue;
    }

    auto& in_col = input.col(col_idx);
    int32_t type_id = rdf_shader_type_id(in_col.type.id);

    auto out_buf = allocate_buffer(_ctx, rdf_type_size(in_col.type.id));

    sum_push_constants pc{};
    pc.input_addr = in_col.address();
    pc.output_addr = out_buf.data();
    pc.size = n;
    pc.pad = 0;

    if (fname == "sum" || fname == "sum_no_overflow") {
      if (type_id == 0) disp.dispatch_sum(pc);
      else              disp.dispatch_sum_float(pc);
    } else if (fname == "min") {
      if (type_id == 0) disp.dispatch_min(pc);
      else              disp.dispatch_min_float(pc);
    } else if (fname == "max") {
      if (type_id == 0) disp.dispatch_max(pc);
      else              disp.dispatch_max_float(pc);
    } else if (fname == "count") {
      output.columns[i] = allocate_column(_ctx, {rasterdf::type_id::INT64}, 1);
      int64_t count = static_cast<int64_t>(n);
      output.columns[i].data.copy_from_host(&count, sizeof(int64_t),
                                              _ctx.device(), _ctx.queue(), _ctx.command_pool());
      continue;
    } else if (fname == "avg" || fname == "mean") {
      // AVG = SUM / COUNT (compute sum on GPU, divide on CPU)
      if (type_id == 0) disp.dispatch_sum(pc);
      else              disp.dispatch_sum_float(pc);

      if (type_id == 0) {
        int32_t sum_int = 0;
        out_buf.copy_to_host(&sum_int, sizeof(int32_t),
                              _ctx.device(), _ctx.queue(), _ctx.command_pool());
        double avg = static_cast<double>(sum_int) / static_cast<double>(n);
        output.columns[i] = allocate_column(_ctx, {rasterdf::type_id::FLOAT64}, 1);
        output.columns[i].data.copy_from_host(&avg, sizeof(double),
                                                _ctx.device(), _ctx.queue(), _ctx.command_pool());
      } else {
        float sum_val = 0;
        out_buf.copy_to_host(&sum_val, sizeof(float),
                              _ctx.device(), _ctx.queue(), _ctx.command_pool());
        double avg = static_cast<double>(sum_val) / static_cast<double>(n);
        output.columns[i] = allocate_column(_ctx, {rasterdf::type_id::FLOAT64}, 1);
        output.columns[i].data.copy_from_host(&avg, sizeof(double),
                                                _ctx.device(), _ctx.queue(), _ctx.command_pool());
      }
      continue;
    } else {
      throw duckdb::NotImplementedException(
        "RasterDB GPU: unsupported aggregate function '%s'", fname.c_str());
    }

    // Wrap the output buffer as a 1-element column
    output.columns[i].type = in_col.type;
    output.columns[i].num_rows = 1;
    output.columns[i].data = std::move(out_buf);
  }
}

// ============================================================================
// ORDER BY — radix sort via rasterdf
// ============================================================================

std::unique_ptr<gpu_table> gpu_executor::execute_order(duckdb::LogicalOrder& op)
{
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
  auto& sort_col = input->col(ref.index);
  int32_t type_id = rdf_shader_type_id(sort_col.type.id);

  uint32_t n = static_cast<uint32_t>(input->num_rows());
  auto& disp = _ctx.dispatcher();

  uint32_t num_groups = div_ceil(n, WG_SIZE);
  uint32_t num_blocks = std::max(1u, div_ceil(num_groups * 256, WG_SIZE * 2));

  auto data_a = allocate_buffer(_ctx, n * sizeof(uint32_t));
  auto data_b = allocate_buffer(_ctx, n * sizeof(uint32_t));
  auto payload_a = allocate_buffer(_ctx, n * sizeof(uint32_t));
  auto payload_b = allocate_buffer(_ctx, n * sizeof(uint32_t));
  auto hist_buf = allocate_buffer(_ctx, num_groups * 256 * sizeof(uint32_t));
  auto partial_buf = allocate_buffer(_ctx, (num_blocks + 1) * sizeof(uint32_t));
  auto bucket_totals = allocate_buffer(_ctx, 256 * sizeof(uint32_t));
  auto global_offsets = allocate_buffer(_ctx, 256 * sizeof(uint32_t));

  // Copy sort key to data_a
  {
    binary_op_push_constants pc{};
    pc.input_a = sort_col.address();
    pc.output_addr = data_a.data();
    pc.size = n;
    pc.op = 0; pc.scalar_val = 0; pc.mode = 1; pc.type_id = type_id;
    pc.input_b = 0; pc.debug_mode = 0;
    disp.dispatch_binary_op(pc);
  }

  // Float: convert to sortable unsigned
  if (type_id == 1) {
    radix_init_indices_pc pc{};
    pc.indices_ptr = data_a.data();
    pc.numElements = n;
    disp.dispatch_float_to_sortable(pc, num_groups);
  }

  // Initialize index payload [0, 1, 2, ..., n-1]
  {
    radix_init_indices_pc pc{};
    pc.indices_ptr = payload_a.data();
    pc.numElements = n;
    disp.dispatch_radix_init_indices(pc, num_groups);
  }

  // Run batched radix sort
  disp.dispatch_radix_sort_batched(
    data_a.data(), data_b.data(), payload_a.data(), payload_b.data(),
    hist_buf.data(), partial_buf.data(), bucket_totals.data(), global_offsets.data(),
    n, num_groups, num_blocks);

  // Gather all columns using sorted indices (result in payload_a after 8 passes)
  auto result = std::make_unique<gpu_table>();
  result->duckdb_types = input->duckdb_types;
  result->columns.resize(input->num_columns());

  for (size_t c = 0; c < input->num_columns(); c++) {
    auto& in_col = input->col(c);
    result->columns[c] = allocate_column(_ctx, in_col.type, input->num_rows());

    gather_indices_pc gc{};
    gc.input_addr = in_col.address();
    gc.indices_addr = payload_a.data();
    gc.output_addr = result->columns[c].address();
    gc.size = n;
    disp.dispatch_gather_indices(gc, num_groups);
  }

  return result;
}

// ============================================================================
// LIMIT — take first N rows
// ============================================================================

std::unique_ptr<gpu_table> gpu_executor::execute_limit(duckdb::LogicalLimit& op)
{
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
// Convert gpu_table -> DuckDB QueryResult
// ============================================================================

duckdb::unique_ptr<duckdb::QueryResult> gpu_executor::to_query_result(
  std::unique_ptr<gpu_table> table,
  const duckdb::vector<duckdb::string>& names,
  const duckdb::vector<duckdb::LogicalType>& types)
{
  // Download full columns to CPU first, then build DataChunks
  size_t num_cols = table->num_columns();
  rasterdf::size_type total_rows = table->num_rows();

  // Download all column data to host buffers
  std::vector<std::vector<uint8_t>> host_data(num_cols);
  for (size_t c = 0; c < num_cols; c++) {
    auto& col = table->col(c);
    size_t bytes = col.byte_size();
    host_data[c].resize(bytes);
    if (bytes > 0) {
      download_column(_ctx, col, host_data[c].data(), bytes);
    }
  }

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
      size_t elem_size = rdf_type_size(table->col(c).type.id);
      size_t byte_offset = static_cast<size_t>(offset) * elem_size;
      size_t byte_count = static_cast<size_t>(count) * elem_size;

      auto dst = reinterpret_cast<uint8_t*>(chunk.data[c].GetData());
      std::memcpy(dst, host_data[c].data() + byte_offset, byte_count);
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
