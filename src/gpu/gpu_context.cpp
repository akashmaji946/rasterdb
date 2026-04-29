/*
 * Copyright 2025, RasterDB Contributors.
 * GPU Context implementation — initializes rasterdf Vulkan backend.
 */

#include "gpu/gpu_context.hpp"
#include "log/logging.hpp"

#include <rasterdf/simple_garuda_join.hpp>
#include <cstdlib>

namespace rasterdb {
namespace gpu {

std::unique_ptr<gpu_context> gpu_context::_instance;
std::once_flag gpu_context::_init_flag;

gpu_context::gpu_context(size_t memory_limit)
{
  RASTERDB_LOG_INFO("Initializing RasterDB GPU context (Vulkan/rasterdf)...");

  // Create Vulkan context — use NVIDIA discrete GPU
  _ctx = std::make_unique<rasterdf::context>(rasterdf::DeviceVendor::NVIDIA);
  RASTERDB_LOG_INFO("GPU device: {}", _ctx->device_name());
  RASTERDB_LOG_INFO("GPU memory: {} MB", _ctx->device_memory_bytes() / (1024 * 1024));

  // Set shader directory if not already set
  if (!std::getenv("RASTERDF_SHADER_DIR")) {
    // Try to find shaders relative to rasterdf source
    // The CMake will set this properly; this is a fallback
    RASTERDB_LOG_DEBUG("RASTERDF_SHADER_DIR not set; dispatcher will use fallback paths");
  }

  // Create dispatcher (loads all compute shader pipelines)
  _dispatcher = std::make_unique<rasterdf::execution::dispatcher>(*_ctx);
  RASTERDB_LOG_INFO("Vulkan compute pipelines loaded");

  // Eagerly create simple_garuda_engine (graphics pipelines, render pass, etc.)
  // so the first join call doesn't pay the ~15ms init cost.
  rasterdf::simple_garuda_engine_init(*_ctx);
  RASTERDB_LOG_INFO("Simple Garuda join pipelines loaded");

  // Create memory manager
  if (memory_limit == 0) {
    memory_limit = static_cast<size_t>(_ctx->device_memory_bytes() * 0.8);
  }
  _mem_mgr = std::make_unique<rasterdf::memory_manager>(*_ctx, memory_limit);
  RASTERDB_LOG_INFO("GPU memory manager initialized ({} MB limit)",
                   memory_limit / (1024 * 1024));
}

gpu_context::~gpu_context()
{
  // Destroy in reverse order. Do NOT log here — the static singleton
  // may be destroyed after spdlog's global registry is torn down.
  rasterdf::simple_garuda_engine_reset();
  _mem_mgr.reset();
  _dispatcher.reset();
  _ctx.reset();
}

gpu_context& gpu_context::instance()
{
  if (!_instance) {
    throw std::runtime_error("GPU context not initialized. Call gpu_context::initialize() first.");
  }
  return *_instance;
}

bool gpu_context::is_initialized()
{
  return _instance != nullptr;
}

void gpu_context::initialize(size_t memory_limit)
{
  std::call_once(_init_flag, [memory_limit]() {
    _instance = std::make_unique<gpu_context>(memory_limit);
  });
}

void gpu_context::shutdown()
{
  rasterdf::simple_garuda_engine_reset();
  _instance.reset();
}

} // namespace gpu
} // namespace rasterdb
