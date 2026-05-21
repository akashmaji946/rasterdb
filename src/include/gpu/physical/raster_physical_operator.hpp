#pragma once

#include <duckdb/common/common.hpp>
#include <duckdb/planner/logical_operator.hpp>

#include <memory>
#include <string>
#include <vector>

namespace rasterdb {
namespace gpu {

class gpu_executor;
class gpu_table;

enum class raster_physical_operator_type {
  GET,
  FILTER,
  PROJECTION,
  AGGREGATE,
  ORDER_BY,
  LIMIT,
  COMPARISON_JOIN,
  TOP_N,
  UNKNOWN
};

class raster_physical_operator {
public:
  raster_physical_operator(raster_physical_operator_type type,
                           duckdb::LogicalOperator &logical);
  virtual ~raster_physical_operator() = default;

  raster_physical_operator_type type() const;
  duckdb::LogicalOperator &logical();
  const duckdb::LogicalOperator &logical() const;

  void add_child(std::unique_ptr<raster_physical_operator> child);
  const std::vector<std::unique_ptr<raster_physical_operator>> &children() const;

  std::string name() const;
  void append_plan(std::string &out, int depth = 0) const;

private:
  raster_physical_operator_type _type;
  duckdb::LogicalOperator *_logical;
  std::vector<std::unique_ptr<raster_physical_operator>> _children;
};

} // namespace gpu
} // namespace rasterdb
