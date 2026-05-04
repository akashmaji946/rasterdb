/*
 * Copyright 2025, RasterDB Contributors.
 * Split from src/gpu/gpu_executor.cpp.
 */

#include "gpu/gpu_executor_internal.hpp"

namespace rasterdb {
namespace gpu {

// ============================================================================
// JOIN — hash join via rasterdf
// ============================================================================

// Toggle between compute-shader hash join and graphics-pipeline simple garuda join
static constexpr bool USE_SIMPLE_GFX_JOIN = true;

std::unique_ptr<gpu_table> gpu_executor::execute_join(duckdb::LogicalComparisonJoin& op)
{
  RASTERDB_LOG_DEBUG("GPU execute_join (simple_garuda={})", USE_SIMPLE_GFX_JOIN ? "true" : "false");
  D_ASSERT(op.children.size() == 2);

  if (op.join_type != duckdb::JoinType::INNER) {
    throw duckdb::NotImplementedException("RasterDB GPU: only INNER JOIN supported, got %s",
                                          duckdb::JoinTypeToString(op.join_type).c_str());
  }

  // Validate all conditions are equi-joins on column references
  RASTERDB_LOG_DEBUG("[RDB_DEBUG] JOIN: {} conditions", op.conditions.size());
  for (size_t ci = 0; ci < op.conditions.size(); ci++) {
    auto& c = op.conditions[ci];
    RASTERDB_LOG_DEBUG("[RDB_DEBUG]   cond[{}]: cmp={} left={} right={}",
                       ci,
                       duckdb::ExpressionTypeToString(c.comparison),
                       c.left->ToString(),
                       c.right->ToString());
  }
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
  auto left_table  = execute_operator(*op.children[0]);
  auto right_table = execute_operator(*op.children[1]);

  stage_timer t("  join");  // Timer starts AFTER child execution

  RASTERDB_LOG_DEBUG("JOIN: left {} rows x {} cols, right {} rows x {} cols",
                     left_table->num_rows(),
                     left_table->num_columns(),
                     right_table->num_rows(),
                     right_table->num_columns());

  // Join on FIRST condition
  auto& cond0        = op.conditions[0];
  auto left_key_idx  = unwrap_cast(*cond0.left).Cast<duckdb::BoundReferenceExpression>().index;
  auto right_key_idx = unwrap_cast(*cond0.right).Cast<duckdb::BoundReferenceExpression>().index;

  auto left_key_view  = left_table->col(left_key_idx).view();
  auto right_key_view = right_table->col(right_key_idx).view();
  RASTERDB_LOG_DEBUG(
    "[RDB_DEBUG] JOIN keys: L col[{}] addr=0x{:x} size={}, R col[{}] addr=0x{:x} size={}",
    static_cast<size_t>(left_key_idx),
    static_cast<uint64_t>(left_key_view.data()),
    left_key_view.size(),
    static_cast<size_t>(right_key_idx),
    static_cast<uint64_t>(right_key_view.data()),
    right_key_view.size());

  std::unique_ptr<rasterdf::column> left_indices;
  std::unique_ptr<rasterdf::column> right_indices;
  rasterdf::size_type match_count = 0;

  if constexpr (USE_SIMPLE_GFX_JOIN) {
    // ── Simple Garuda Join (graphics-pipeline, vertex shader hash join) ──
    uint32_t left_n  = static_cast<uint32_t>(left_key_view.size());
    uint32_t right_n = static_cast<uint32_t>(right_key_view.size());

    auto sg_result = rasterdf::simple_garuda_inner_join(left_key_view.data(),
                                                        left_n,
                                                        right_key_view.data(),
                                                        right_n,
                                                        _ctx.vk_context(),
                                                        _ctx.dispatcher(),
                                                        _ctx.workspace_mr());

    match_count = static_cast<rasterdf::size_type>(sg_result.num_matches);

    if (match_count > 0) {
      left_indices =
        std::make_unique<rasterdf::column>(rasterdf::data_type{rasterdf::type_id::INT32},
                                           match_count,
                                           std::move(*sg_result.left_indices));
      right_indices =
        std::make_unique<rasterdf::column>(rasterdf::data_type{rasterdf::type_id::INT32},
                                           match_count,
                                           std::move(*sg_result.right_indices));
    }
  } else {
    // ── Compute-shader hash join ──
    std::vector<rasterdf::column_view> lk = {left_key_view};
    std::vector<rasterdf::column_view> rk = {right_key_view};
    rasterdf::table_view left_keys_tv(lk);
    rasterdf::table_view right_keys_tv(rk);

    auto join_result = rasterdf::inner_join(
      left_keys_tv, right_keys_tv, _ctx.vk_context(), _ctx.dispatcher(), _ctx.workspace_mr());

    left_indices  = std::move(join_result.first);
    right_indices = std::move(join_result.second);
    match_count   = left_indices ? left_indices->size() : 0;
  }
  RASTERDB_LOG_DEBUG("JOIN: {} matches after first condition", match_count);

  if (match_count == 0) {
    auto result          = std::make_unique<gpu_table>();
    result->duckdb_types = op.types;
    result->columns.resize(op.types.size());
    for (size_t i = 0; i < op.types.size(); i++) {
      result->columns[i].type     = to_rdf_type(op.types[i]);
      result->columns[i].num_rows = 0;
    }
    return result;
  }

  auto left_idx_view  = left_indices->view();
  auto right_idx_view = right_indices->view();

  // Gather from both tables using the matched indices
  auto left_gathered = rasterdf::gather(
    left_table->view(), left_idx_view, _ctx.vk_context(), _ctx.dispatcher(), _ctx.workspace_mr());
  auto right_gathered = rasterdf::gather(
    right_table->view(), right_idx_view, _ctx.vk_context(), _ctx.dispatcher(), _ctx.workspace_mr());

  auto left_cols  = left_gathered->extract();
  auto right_cols = right_gathered->extract();

  // Build result: left columns then right columns
  auto result          = std::make_unique<gpu_table>();
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
    auto& cond  = op.conditions[ci];
    auto lk_idx = unwrap_cast(*cond.left).Cast<duckdb::BoundReferenceExpression>().index;
    auto rk_idx = unwrap_cast(*cond.right).Cast<duckdb::BoundReferenceExpression>().index;

    // In the merged table: left cols at [0..num_left-1], right cols at [num_left..]
    auto& left_key_col  = result->col(lk_idx);
    auto& right_key_col = result->col(num_left_cols + rk_idx);

    uint32_t n = static_cast<uint32_t>(result->num_rows());
    auto mask  = allocate_column(_ctx, {rasterdf::type_id::INT32}, n);

    compare_columns_push_constants cpc{};
    cpc.input_a     = left_key_col.address();
    cpc.input_b     = right_key_col.address();
    cpc.output_addr = mask.address();
    cpc.size        = n;
    cpc.op          = 4;                                                    // EQ
    cpc.type_id     = static_cast<int32_t>(rasterdf::ShaderTypeId::INT32);  // INT32
    _ctx.dispatcher().dispatch_compare_columns(cpc);

    result = apply_filter_mask(*result, mask);
    RASTERDB_LOG_DEBUG("JOIN: {} rows after condition {}", result->num_rows(), ci);
  }

  return result;
}

}  // namespace gpu
}  // namespace rasterdb
