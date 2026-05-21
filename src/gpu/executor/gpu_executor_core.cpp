/*
 * Copyright 2025, RasterDB Contributors.
 * Split from src/gpu/gpu_executor.cpp.
 */

#include "gpu/gpu_executor_internal.hpp"
#include "config.hpp"
#include "gpu/physical/raster_physical_operator.hpp"
#include "gpu/physical/raster_physical_plan_generator.hpp"
#include "gpu/pipeline/raster_pipeline.hpp"
#include "gpu/pipeline/raster_pipeline_executor.hpp"

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

  std::string logical_plan_text;
  append_logical_plan(plan, logical_plan_text);
  RASTERDB_LOG_INFO("[RDB_PLAN] Executor logical plan:\n{}", logical_plan_text);
  if (debug_logging_enabled()) { debug_print_plan(plan); }
  analyze_plan_hints(plan);

  if (duckdb::Config::MODIFIED_PIPELINE) {
    raster_physical_plan_generator plan_generator;
    auto physical_plan = plan_generator.create_plan(plan);
    std::string physical_plan_text;
    physical_plan->append_plan(physical_plan_text);
    RASTERDB_LOG_INFO("[RDB_PLAN] Raster physical plan:\n{}", physical_plan_text);

    raster_pipeline_builder pipeline_builder;
    auto pipeline = pipeline_builder.build(*physical_plan);
    RASTERDB_LOG_INFO("[RDB_PLAN] Raster pipeline:\n{}", pipeline.describe());

    raster_pipeline_executor pipeline_executor(*this);
    return pipeline_executor.execute(*physical_plan, pipeline);
  }

  return execute_operator(plan);
}

std::unique_ptr<gpu_table> gpu_executor::execute_physical_operator(raster_physical_operator& op)
{
  return execute_operator(op.logical());
}

std::unique_ptr<gpu_table> gpu_executor::execute_operator(duckdb::LogicalOperator& op)
{
  auto op_name = duckdb::LogicalOperatorToString(op.type);
  int64_t rows_in = -1;
  if (!op.children.empty() && op.children[0]) {
    rows_in = static_cast<int64_t>(op.children[0]->estimated_cardinality);
  }
  auto t0 = std::chrono::high_resolution_clock::now();
  std::unique_ptr<gpu_table> result;
  switch (op.type) {
    case duckdb::LogicalOperatorType::LOGICAL_GET:
      result = execute_get(op.Cast<duckdb::LogicalGet>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_FILTER:
      result = execute_filter(op.Cast<duckdb::LogicalFilter>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_PROJECTION:
      result = execute_projection(op.Cast<duckdb::LogicalProjection>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
      result = execute_aggregate(op.Cast<duckdb::LogicalAggregate>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_ORDER_BY:
      result = execute_order(op.Cast<duckdb::LogicalOrder>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_LIMIT:
      result = execute_limit(op.Cast<duckdb::LogicalLimit>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_TOP_N:
      result = execute_top_n(op.Cast<duckdb::LogicalTopN>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
      result = execute_join(op.Cast<duckdb::LogicalComparisonJoin>());
      break;
    default:
      throw duckdb::NotImplementedException(
        "RasterDB GPU: unsupported operator %s",
        duckdb::LogicalOperatorToString(op.type).c_str());
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  auto rows_out = result ? result->num_rows() : 0;
  RASTERDB_LOG_INFO("[RDB_OP] op={} rows_in={} rows_out={} time_ms={:.3f}",
                    op_name, rows_in, rows_out, ms);
  return result;
}

} // namespace gpu
} // namespace rasterdb
