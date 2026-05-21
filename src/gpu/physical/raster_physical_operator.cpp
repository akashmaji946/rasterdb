#include "gpu/physical/raster_physical_operator.hpp"

#include <duckdb/planner/logical_operator.hpp>

namespace rasterdb {
namespace gpu {

raster_physical_operator::raster_physical_operator(raster_physical_operator_type type,
                                                   duckdb::LogicalOperator &logical)
    : _type(type), _logical(&logical) {}

raster_physical_operator_type raster_physical_operator::type() const { return _type; }

duckdb::LogicalOperator &raster_physical_operator::logical() { return *_logical; }

const duckdb::LogicalOperator &raster_physical_operator::logical() const { return *_logical; }

void raster_physical_operator::add_child(std::unique_ptr<raster_physical_operator> child) {
  _children.push_back(std::move(child));
}

const std::vector<std::unique_ptr<raster_physical_operator>> &raster_physical_operator::children() const {
  return _children;
}

std::string raster_physical_operator::name() const {
  switch (_type) {
  case raster_physical_operator_type::GET:
    return "GET";
  case raster_physical_operator_type::FILTER:
    return "FILTER";
  case raster_physical_operator_type::PROJECTION:
    return "PROJECTION";
  case raster_physical_operator_type::AGGREGATE:
    return "AGGREGATE";
  case raster_physical_operator_type::ORDER_BY:
    return "ORDER_BY";
  case raster_physical_operator_type::LIMIT:
    return "LIMIT";
  case raster_physical_operator_type::COMPARISON_JOIN:
    return "COMPARISON_JOIN";
  case raster_physical_operator_type::TOP_N:
    return "TOP_N";
  case raster_physical_operator_type::UNKNOWN:
  default:
    return "UNKNOWN";
  }
}

void raster_physical_operator::append_plan(std::string &out, int depth) const {
  out.append(static_cast<size_t>(depth * 2), ' ');
  out += name();
  out += " [";
  out += duckdb::LogicalOperatorToString(_logical->type);
  out += "]\n";
  for (auto const &child : _children) {
    child->append_plan(out, depth + 1);
  }
}

} // namespace gpu
} // namespace rasterdb
