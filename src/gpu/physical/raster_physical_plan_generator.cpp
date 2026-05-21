#include "gpu/physical/raster_physical_plan_generator.hpp"

namespace rasterdb {
namespace gpu {

std::unique_ptr<raster_physical_operator>
raster_physical_plan_generator::create_plan(duckdb::LogicalOperator &logical) {
  auto physical = std::make_unique<raster_physical_operator>(map_type(logical.type), logical);
  for (auto &child : logical.children) {
    physical->add_child(create_plan(*child));
  }
  return physical;
}

raster_physical_operator_type
raster_physical_plan_generator::map_type(duckdb::LogicalOperatorType type) const {
  switch (type) {
  case duckdb::LogicalOperatorType::LOGICAL_GET:
    return raster_physical_operator_type::GET;
  case duckdb::LogicalOperatorType::LOGICAL_FILTER:
    return raster_physical_operator_type::FILTER;
  case duckdb::LogicalOperatorType::LOGICAL_PROJECTION:
    return raster_physical_operator_type::PROJECTION;
  case duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
    return raster_physical_operator_type::AGGREGATE;
  case duckdb::LogicalOperatorType::LOGICAL_ORDER_BY:
    return raster_physical_operator_type::ORDER_BY;
  case duckdb::LogicalOperatorType::LOGICAL_LIMIT:
    return raster_physical_operator_type::LIMIT;
  case duckdb::LogicalOperatorType::LOGICAL_TOP_N:
    return raster_physical_operator_type::TOP_N;
  case duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
    return raster_physical_operator_type::COMPARISON_JOIN;
  default:
    return raster_physical_operator_type::UNKNOWN;
  }
}

} // namespace gpu
} // namespace rasterdb
