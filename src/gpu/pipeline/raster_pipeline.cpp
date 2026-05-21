#include "gpu/pipeline/raster_pipeline.hpp"

namespace rasterdb {
namespace gpu {

void raster_pipeline::add_node(const raster_physical_operator &op, raster_pipeline_role role) {
  _nodes.push_back(raster_pipeline_node{&op, role});
}

const std::vector<raster_pipeline_node> &raster_pipeline::nodes() const { return _nodes; }

std::string raster_pipeline::describe() const {
  std::string out;
  for (auto const &node : _nodes) {
    switch (node.role) {
    case raster_pipeline_role::SOURCE:
      out += "SOURCE";
      break;
    case raster_pipeline_role::SINK:
      out += "SINK";
      break;
    case raster_pipeline_role::OPERATOR:
    default:
      out += "OPERATOR";
      break;
    }
    out += ": ";
    out += node.op ? node.op->name() : "<null>";
    out += "\n";
  }
  return out;
}

raster_pipeline raster_pipeline_builder::build(const raster_physical_operator &root) const {
  raster_pipeline pipeline;
  visit(root, pipeline);
  return pipeline;
}

void raster_pipeline_builder::visit(const raster_physical_operator &op, raster_pipeline &pipeline) const {
  for (auto const &child : op.children()) {
    visit(*child, pipeline);
  }
  pipeline.add_node(op, role_for(op));
}

raster_pipeline_role raster_pipeline_builder::role_for(const raster_physical_operator &op) const {
  switch (op.type()) {
  case raster_physical_operator_type::GET:
    return raster_pipeline_role::SOURCE;
  case raster_physical_operator_type::AGGREGATE:
  case raster_physical_operator_type::ORDER_BY:
  case raster_physical_operator_type::LIMIT:
  case raster_physical_operator_type::TOP_N:
    return raster_pipeline_role::SINK;
  default:
    return raster_pipeline_role::OPERATOR;
  }
}

} // namespace gpu
} // namespace rasterdb
