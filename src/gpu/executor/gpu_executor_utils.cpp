/*
 * Copyright 2025, RasterDB Contributors.
 * Shared executor implementation helpers.
 */

#include "gpu/gpu_executor_internal.hpp"

namespace rasterdb {
namespace gpu {

void debug_print_plan(duckdb::LogicalOperator& op, int depth) {
  std::string indent(depth * 2, ' ');
  RASTERDB_LOG_DEBUG("[RDB_PLAN] {}{} (types={}, children={})",
                     indent, duckdb::LogicalOperatorToString(op.type),
                     op.types.size(), op.children.size());
  for (auto& child : op.children) {
    debug_print_plan(*child, depth + 1);
  }
}


} // namespace gpu
} // namespace rasterdb
