/*
 * Copyright 2025, RasterDB Contributors.
 * Split from src/gpu/gpu_executor.cpp.
 */

#include "gpu/gpu_executor_internal.hpp"

namespace rasterdb {
namespace gpu {

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
          RASTERDB_LOG_DEBUG("[TIMER]   hint: count(*) only - scan 1 col");
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
          RASTERDB_LOG_DEBUG("[TIMER]   hint: count(*) only - scan 1 col");
        }
      }
    }
  }
}

// ============================================================================
// Recursive operator dispatch
// ============================================================================

} // namespace gpu
} // namespace rasterdb
