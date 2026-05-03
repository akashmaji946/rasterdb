/*
 * Copyright 2025, RasterDB Contributors.
 * Split from src/gpu/gpu_executor.cpp.
 */

#include "gpu/gpu_executor_internal.hpp"

namespace rasterdb {
namespace gpu {

// ============================================================================

gpu_executor::gpu_executor(gpu_context& ctx, duckdb::ClientContext& client_ctx)
  : _ctx(ctx), _client_ctx(client_ctx) {}


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

  if (debug_logging_enabled()) {
    debug_print_plan(plan);
  }
  analyze_plan_hints(plan);
  auto result = execute_operator(plan);
  return result;
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

} // namespace gpu
} // namespace rasterdb
