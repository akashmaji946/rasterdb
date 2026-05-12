/*
 * Copyright 2025, RasterDB Contributors.
 * Split from src/gpu/gpu_executor.cpp.
 */

#include "gpu/gpu_executor_internal.hpp"

namespace rasterdb {
namespace gpu {

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

  // Check for STRING sort keys or STRING columns in table — not yet supported
  for (auto& order : op.orders) {
    auto& expr = unwrap_cast(*order.expression);
    if (expr.type == duckdb::ExpressionType::BOUND_REF) {
      auto& ref = expr.Cast<duckdb::BoundReferenceExpression>();
      if (input->col(ref.index).is_string()) {
        throw duckdb::NotImplementedException(
          "RasterDB GPU: ORDER BY on STRING columns not yet supported");
      }
    }
  }
  for (size_t c = 0; c < input->num_columns(); c++) {
    if (input->col(c).is_string()) {
      throw duckdb::NotImplementedException(
        "RasterDB GPU: ORDER BY with STRING columns in table not yet supported");
    }
  }

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

} // namespace gpu
} // namespace rasterdb
