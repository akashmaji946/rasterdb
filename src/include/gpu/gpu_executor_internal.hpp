/*
 * Copyright 2026, RasterDB Contributors.
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
#include <rasterdf/non_equi_join.hpp>
#include <rasterdf/groupby.hpp>

#include <duckdb/common/exception.hpp>
#include <duckdb/common/types/data_chunk.hpp>
#include <duckdb/common/types/column/column_data_collection.hpp>
#include <duckdb/main/connection.hpp>
#include <duckdb/main/materialized_query_result.hpp>
#include <duckdb/planner/expression/bound_comparison_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_conjunction_expression.hpp>
#include <duckdb/planner/expression/bound_case_expression.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <duckdb/planner/expression/bound_aggregate_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/expression/bound_cast_expression.hpp>
#include <duckdb/planner/expression/bound_operator_expression.hpp>
#include <duckdb/planner/expression/bound_between_expression.hpp>
#include <duckdb/planner/bound_result_modifier.hpp>
#include <duckdb/catalog/catalog_entry/table_catalog_entry.hpp>
#include <duckdb/function/table_function.hpp>
#include <duckdb/common/types/hugeint.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rasterdb {
namespace gpu {

using namespace rasterdf::execution;

static constexpr uint32_t WG_SIZE = 256;
inline uint32_t div_ceil(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
inline bool debug_logging_enabled() { return duckdb::RasterDBShouldLog(spdlog::level::debug); }

inline bool can_alias_fixed_width_column(const gpu_column& src) {
  return !src.is_host_only && !src.is_string() &&
         src.cached_buffer != VK_NULL_HANDLE && src.cached_address != 0;
}

inline gpu_column alias_fixed_width_column(const gpu_column& src) {
  gpu_column out;
  out.type = src.type;
  out.num_rows = src.num_rows;
  out.cached_address = src.cached_address;
  out.cached_buffer = src.cached_buffer;
  out.cached_offset = src.cached_offset;
  out.has_i32_minmax = src.has_i32_minmax;
  out.i32_min = src.i32_min;
  out.i32_max = src.i32_max;
  return out;
}

struct scoped_bool_setter {
  bool& target;
  bool old_value;

  scoped_bool_setter(bool& target, bool new_value)
      : target(target), old_value(target) {
    target = new_value;
  }

  ~scoped_bool_setter() {
    target = old_value;
  }
};

void debug_print_plan(duckdb::LogicalOperator& op, int depth = 0);
void append_logical_plan(duckdb::LogicalOperator& op, std::string& out, int depth = 0);

// Per-stage timing helper — uses RASTERDB_LOG_INFO with [TIMER] prefix
struct stage_timer {
  const char* name;
  std::chrono::high_resolution_clock::time_point t0;
  stage_timer(const char* n) : name(n), t0(std::chrono::high_resolution_clock::now()) {}
  ~stage_timer() {
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    RASTERDB_LOG_INFO("[TIMER] {:<30s} {:8.2f} ms", name, ms);
  }
};

// Helper: unwrap BoundCastExpression to find the inner expression.
static inline duckdb::Expression& unwrap_cast(duckdb::Expression& expr) {
  if (expr.expression_class == duckdb::ExpressionClass::BOUND_CAST) {
    auto& cast = expr.Cast<duckdb::BoundCastExpression>();
    if ((expr.return_type.id() == duckdb::LogicalTypeId::DECIMAL ||
         cast.child->return_type.id() == duckdb::LogicalTypeId::DECIMAL) &&
        !(expr.return_type == cast.child->return_type)) {
      return expr;
    }
    return unwrap_cast(*cast.child);
  }
  return expr;
}

} // namespace gpu
} // namespace rasterdb
