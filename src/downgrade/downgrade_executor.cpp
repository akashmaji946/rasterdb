/*
 * STUBBED for RasterDB — CUDA memory downgrade system removed.
 * Memory tiering handled by rasterdf (Vulkan + VMA).
 */
#include <downgrade/downgrade_executor.hpp>

namespace rasterdb {
namespace parallel {

// All downgrade operations are no-ops in RasterDB.
// Memory management is handled by rasterdf's VMA-based allocator.

} // namespace parallel
} // namespace rasterdb
