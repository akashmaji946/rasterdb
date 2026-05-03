/*
 * Copyright 2025, RasterDB Contributors.
 * Split from src/gpu/gpu_executor.cpp.
 */

#include "gpu/gpu_executor_internal.hpp"

namespace rasterdb {
namespace gpu {

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
      RASTERDB_LOG_DEBUG("[RDB_DEBUG]     count_star = {} (host-only)", count);
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
    RASTERDB_LOG_DEBUG("[RDB_DEBUG]     reduce({})              {:8.2f} ms",
                       fname,
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

} // namespace gpu
} // namespace rasterdb
