/*
 * Copyright 2026, RasterDB Contributors.
 * GPU Context implementation — initializes rasterdf Vulkan backend.
 */

#include "gpu/gpu_context.hpp"
#include "log/logging.hpp"

#include <rasterdf/simple_garuda_join.hpp>
#include <rasterdf/gfx_groupby_engine.hpp>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace rasterdb {
namespace gpu {

std::unique_ptr<gpu_context> gpu_context::_instance;
std::once_flag gpu_context::_init_flag;

#ifndef RASTERDB_RASTERDF_SHADER_DIR
#define RASTERDB_RASTERDF_SHADER_DIR ""
#endif

namespace {

bool shader_dir_is_usable(const std::filesystem::path& dir)
{
  return !dir.empty() && std::filesystem::exists(dir / "transform.spv");
}

std::string normalize_device_name(const char* value)
{
  std::string normalized = value ? value : "";
  for (auto& ch : normalized) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return normalized;
}

rasterdf::DeviceVendor configured_device_vendor()
{
  const auto device = normalize_device_name(std::getenv("RASTERDF_DEVICE"));
  if (device.empty() || device == "nvidia") {
    return rasterdf::DeviceVendor::NVIDIA;
  }
  if (device == "amd") {
    return rasterdf::DeviceVendor::AMD;
  }
  if (device == "intel") {
    return rasterdf::DeviceVendor::INTEL;
  }
  if (device == "any" || device == "software" || device == "llvmpipe") {
    return rasterdf::DeviceVendor::ANY;
  }

  RASTERDB_LOG_WARN("Unknown RASTERDF_DEVICE='{}'; falling back to NVIDIA", device);
  return rasterdf::DeviceVendor::NVIDIA;
}

void configure_rasterdf_shader_dir()
{
  const char* env = std::getenv("RASTERDF_SHADER_DIR");
  if (env && shader_dir_is_usable(env)) {
    RASTERDB_LOG_DEBUG("Using RASTERDF_SHADER_DIR={}", env);
    return;
  }

  std::vector<std::filesystem::path> candidates;
  if (std::string(RASTERDB_RASTERDF_SHADER_DIR).size() > 0) {
    candidates.emplace_back(RASTERDB_RASTERDF_SHADER_DIR);
  }
  candidates.emplace_back("../rasterdf/shaders/compiled");
  candidates.emplace_back("rasterdf/shaders/compiled");
  candidates.emplace_back("shaders/compiled");

  for (auto& candidate : candidates) {
    if (!shader_dir_is_usable(candidate)) {
      continue;
    }
    auto shader_dir = std::filesystem::absolute(candidate).lexically_normal().string();
#if defined(_WIN32)
    _putenv_s("RASTERDF_SHADER_DIR", shader_dir.c_str());
#else
    setenv("RASTERDF_SHADER_DIR", shader_dir.c_str(), 1);
#endif
    RASTERDB_LOG_INFO("RasterDF shader dir: {}", shader_dir);
    return;
  }

  if (env) {
    RASTERDB_LOG_WARN(
      "RASTERDF_SHADER_DIR={} does not contain transform.spv; dispatcher will report the shader error",
      env);
  } else {
    RASTERDB_LOG_WARN(
      "RASTERDF_SHADER_DIR not set and no RasterDF shader directory was found; dispatcher will use fallback paths");
  }
}

} // namespace

gpu_context::gpu_context(size_t memory_limit)
{
  RASTERDB_LOG_INFO("Initializing RasterDB GPU context (Vulkan/rasterdf)...");

  // Create Vulkan context. RASTERDF_DEVICE accepts: nvidia, amd, intel, any/software/llvmpipe.
  const auto preferred_vendor = configured_device_vendor();
  _ctx = std::make_unique<rasterdf::context>(preferred_vendor);
  RASTERDB_LOG_INFO("GPU device: {}", _ctx->device_name());
  RASTERDB_LOG_INFO("GPU memory: {} MB", _ctx->device_memory_bytes() / (1024 * 1024));

  configure_rasterdf_shader_dir();

  // Create dispatcher (loads all compute shader pipelines)
  _dispatcher = std::make_unique<rasterdf::execution::dispatcher>(*_ctx);
  RASTERDB_LOG_INFO("Vulkan Compute pipelines loaded.");

  // Eagerly create simple_garuda_engine (graphics pipelines, render pass, etc.)
  // so the first join call doesn't pay the ~15ms init cost.
  rasterdf::simple_garuda_engine_init(*_ctx);
  RASTERDB_LOG_INFO("Simple Garuda Join (SGJ) pipelines loaded.");

  // Create memory manager
  if (memory_limit == 0) {
    memory_limit = static_cast<size_t>(_ctx->device_memory_bytes() * 0.8);
  }
  _mem_mgr = std::make_unique<rasterdf::memory_manager>(*_ctx, memory_limit);
  RASTERDB_LOG_INFO("GPU Memory Manager: initialized ({} MB limit.)",
                   memory_limit / (1024 * 1024));
}

gpu_context::~gpu_context()
{
  // Destroy in reverse order. Do NOT log here — the static singleton
  // may be destroyed after spdlog's global registry is torn down.
  rasterdf::gfx_groupby_engine_reset();
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
  rasterdf::gfx_groupby_engine_reset();
  rasterdf::simple_garuda_engine_reset();
  _instance.reset();
}

} // namespace gpu
} // namespace rasterdb
