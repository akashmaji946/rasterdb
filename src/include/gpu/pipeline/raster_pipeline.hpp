#pragma once

#include "gpu/physical/raster_physical_operator.hpp"

#include <string>
#include <vector>

namespace rasterdb {
namespace gpu {

enum class raster_pipeline_role {
  SOURCE,
  OPERATOR,
  SINK
};

struct raster_pipeline_node {
  const raster_physical_operator *op{nullptr};
  raster_pipeline_role role{raster_pipeline_role::OPERATOR};
};

class raster_pipeline {
public:
  void add_node(const raster_physical_operator &op, raster_pipeline_role role);
  const std::vector<raster_pipeline_node> &nodes() const;
  std::string describe() const;

private:
  std::vector<raster_pipeline_node> _nodes;
};

class raster_pipeline_builder {
public:
  raster_pipeline build(const raster_physical_operator &root) const;

private:
  void visit(const raster_physical_operator &op, raster_pipeline &pipeline) const;
  raster_pipeline_role role_for(const raster_physical_operator &op) const;
};

} // namespace gpu
} // namespace rasterdb
