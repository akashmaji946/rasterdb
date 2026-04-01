/*
 * Copyright 2025, RasterDB Contributors.
 * GPU Context — wraps rasterdf Vulkan context, dispatcher, and memory manager.
 */

#pragma once

#include <rasterdf/core/context.hpp>
#include <rasterdf/execution/dispatcher.hpp>
#include <rasterdf/memory/memory_manager.hpp>

#include <memory>
#include <mutex>
#include <string>

namespace rasterdb {
namespace gpu {

class gpu_context {
public:
  /// Initialize the GPU context with a memory limit (bytes).
  /// Set memory_limit = 0 to use 80% of device memory.
  explicit gpu_context(size_t memory_limit = 0);
  ~gpu_context();

  gpu_context(const gpu_context&) = delete;
  gpu_context& operator=(const gpu_context&) = delete;

  rasterdf::context& vk_context() { return *_ctx; }
  rasterdf::execution::dispatcher& dispatcher() { return *_dispatcher; }
  rasterdf::memory_manager& memory() { return *_mem_mgr; }
  rasterdf::memory_resource* workspace_mr() { return _mem_mgr->workspace_resource(); }
  rasterdf::memory_resource* data_mr() { return _mem_mgr->data_resource(); }
  rasterdf::memory_resource* host_resource() { return _mem_mgr->host_resource(); }

  VkDevice device() const { return _ctx->device(); }
  VkQueue queue() const { return _ctx->compute_queue(); }
  VkCommandPool command_pool() const { return _dispatcher->_command_pool; }

  /// Singleton access (created on first call)
  static gpu_context& instance();
  static bool is_initialized();
  static void initialize(size_t memory_limit = 0);
  static void shutdown();

private:
  std::unique_ptr<rasterdf::context> _ctx;
  std::unique_ptr<rasterdf::execution::dispatcher> _dispatcher;
  std::unique_ptr<rasterdf::memory_manager> _mem_mgr;

  static std::unique_ptr<gpu_context> _instance;
  static std::once_flag _init_flag;
};

} // namespace gpu
} // namespace rasterdb
