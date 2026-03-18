/*
 * Copyright 2025, RasterDB Contributors.
 * GPU Executor — walks a DuckDB logical plan and executes it on GPU via rasterdf.
 *
 * Supported operators: scan, filter, projection, aggregate, order, limit.
 * Unsupported operators throw NotImplementedException → CPU fallback.
 */

#pragma once

#include "gpu_context.hpp"
#include "gpu_table.hpp"

#include <duckdb/common/types/data_chunk.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/main/query_result.hpp>
#include <duckdb/planner/expression.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/planner/operator/logical_filter.hpp>
#include <duckdb/planner/operator/logical_projection.hpp>
#include <duckdb/planner/operator/logical_aggregate.hpp>
#include <duckdb/planner/operator/logical_order.hpp>
#include <duckdb/planner/operator/logical_limit.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>

#include <memory>

namespace rasterdb {
namespace gpu {

/// Execute a DuckDB logical plan on GPU using rasterdf Vulkan compute.
/// Returns a gpu_table with results, or throws NotImplementedException
/// for unsupported operators (caught by extension to fall back to CPU).
class gpu_executor {
public:
  explicit gpu_executor(gpu_context& ctx, duckdb::ClientContext& client_ctx);

  /// Execute the full logical plan tree, returning results as a gpu_table.
  std::unique_ptr<gpu_table> execute(duckdb::LogicalOperator& plan);

  /// Convert a gpu_table to a DuckDB MaterializedQueryResult.
  duckdb::unique_ptr<duckdb::QueryResult> to_query_result(
    std::unique_ptr<gpu_table> table,
    const duckdb::vector<duckdb::string>& names,
    const duckdb::vector<duckdb::LogicalType>& types);

private:
  gpu_context& _ctx;
  duckdb::ClientContext& _client_ctx;

  // Recursive plan executor
  std::unique_ptr<gpu_table> execute_operator(duckdb::LogicalOperator& op);

  // Individual operator implementations
  std::unique_ptr<gpu_table> execute_get(duckdb::LogicalGet& op);
  std::unique_ptr<gpu_table> execute_filter(duckdb::LogicalFilter& op);
  std::unique_ptr<gpu_table> execute_projection(duckdb::LogicalProjection& op);
  std::unique_ptr<gpu_table> execute_aggregate(duckdb::LogicalAggregate& op);
  std::unique_ptr<gpu_table> execute_order(duckdb::LogicalOrder& op);
  std::unique_ptr<gpu_table> execute_limit(duckdb::LogicalLimit& op);

  // Expression evaluation helpers
  gpu_column evaluate_comparison(const gpu_table& input, duckdb::Expression& expr);
  gpu_column evaluate_binary_op(const gpu_table& input, duckdb::Expression& expr);

  // Filter: apply a boolean mask to compact a table
  std::unique_ptr<gpu_table> apply_filter_mask(const gpu_table& input, gpu_column& mask);

  // Aggregate helpers
  void execute_ungrouped_aggregate(const gpu_table& input,
                                    const duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& aggregates,
                                    gpu_table& output);
};

} // namespace gpu
} // namespace rasterdb
