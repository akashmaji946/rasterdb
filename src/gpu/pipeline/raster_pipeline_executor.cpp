#include "gpu/pipeline/raster_pipeline_executor.hpp"

#include "log/logging.hpp"
#include <chrono>

namespace rasterdb {
namespace gpu {

raster_pipeline_executor::raster_pipeline_executor(gpu_executor &executor) : _executor(executor) {}

std::unique_ptr<gpu_table> raster_pipeline_executor::execute(raster_physical_operator &root,
                                                             const raster_pipeline &pipeline) {
  RASTERDB_LOG_INFO("[RDB_PLAN] Physical pipeline nodes:\n{}", pipeline.describe());
  for (auto const &node : pipeline.nodes()) {
    RASTERDB_LOG_INFO("[RDB_OP] scheduled op={} role={}",
                      node.op ? node.op->name() : "<null>",
                      node.role == raster_pipeline_role::SOURCE ? "SOURCE" :
                      node.role == raster_pipeline_role::SINK ? "SINK" : "OPERATOR");
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  auto result = _executor.execute_physical_operator(root);
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  auto rows_out = result ? result->num_rows() : 0;

  RASTERDB_LOG_INFO("[RDB_OP] op={} rows_in={} rows_out={} time_ms={:.3f}",
                    root.name(), -1, rows_out, ms);
  return result;
}

} // namespace gpu
} // namespace rasterdb
