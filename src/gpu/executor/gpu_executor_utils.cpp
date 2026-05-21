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

void append_logical_plan(duckdb::LogicalOperator& op, std::string& out, int depth) {
  out.append(static_cast<size_t>(depth * 2), ' ');
  out += duckdb::LogicalOperatorToString(op.type);
  out += " types=";
  out += std::to_string(op.types.size());
  out += " children=";
  out += std::to_string(op.children.size());
  out += " est_card=";
  out += std::to_string(static_cast<uint64_t>(op.estimated_cardinality));
  out += "\n";
  for (auto& child : op.children) {
    append_logical_plan(*child, out, depth + 1);
  }
}


} // namespace gpu
} // namespace rasterdb
