/*
 * Copyright 2025, RasterDB Contributors.
 * Internal header shared across gpu_executor_*.cpp files.
 * NOT part of the public API — only included by executor implementation files.
 */

#pragma once

#include "gpu_executor.hpp"
#include "gpu_buffer_manager.hpp"
#include "gpu_types.hpp"
#include "log/logging.hpp"

#include <rasterdf/execution/dispatcher.hpp>
#include <rasterdf/reduction.hpp>
#include <rasterdf/sorting.hpp>
#include <rasterdf/copying.hpp>
#include <rasterdf/stream_compaction.hpp>
#include <rasterdf/join.hpp>
#include <rasterdf/simple_garuda_join.hpp>
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
#include <duckdb/function/table_function.hpp>
#include <duckdb/common/types/hugeint.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <vector>

namespace rasterdb {
namespace gpu {

using namespace rasterdf::execution;

static constexpr uint32_t WG_SIZE = 256;
inline uint32_t div_ceil(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// Per-stage timing helper — prints to stderr with [TIMER] prefix
struct stage_timer {
  const char* name;
  std::chrono::high_resolution_clock::time_point t0;
  stage_timer(const char* n) : name(n), t0(std::chrono::high_resolution_clock::now()) {}
  ~stage_timer() {
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fprintf(stderr, "[TIMER] %-30s %8.2f ms\n", name, ms);
  }
};

// Helper: unwrap BoundCastExpression to find the inner expression.
static inline duckdb::Expression& unwrap_cast(duckdb::Expression& expr) {
  if (expr.expression_class == duckdb::ExpressionClass::BOUND_CAST) {
    auto& cast = expr.Cast<duckdb::BoundCastExpression>();
    return unwrap_cast(*cast.child);
  }
  return expr;
}

} // namespace gpu
} // namespace rasterdb
