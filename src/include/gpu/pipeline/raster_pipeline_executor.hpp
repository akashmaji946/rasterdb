#pragma once

#include "gpu/gpu_executor.hpp"
#include "gpu/pipeline/raster_pipeline.hpp"
#include "gpu/physical/raster_physical_operator.hpp"

#include <memory>

namespace rasterdb {
namespace gpu {

class raster_pipeline_executor {
public:
  explicit raster_pipeline_executor(gpu_executor &executor);

  std::unique_ptr<gpu_table> execute(raster_physical_operator &root,
                                     const raster_pipeline &pipeline);

private:
  gpu_executor &_executor;
};

} // namespace gpu
} // namespace rasterdb
