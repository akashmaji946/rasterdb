#pragma once

#include "gpu/physical/raster_physical_operator.hpp"

#include <duckdb/planner/logical_operator.hpp>

#include <memory>

namespace rasterdb {
namespace gpu {

class raster_physical_plan_generator {
public:
  std::unique_ptr<raster_physical_operator> create_plan(duckdb::LogicalOperator &logical);

private:
  raster_physical_operator_type map_type(duckdb::LogicalOperatorType type) const;
};

} // namespace gpu
} // namespace rasterdb
