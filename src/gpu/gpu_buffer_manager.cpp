/*
 * Copyright 2025, RasterDB Contributors.
 * GPUBufferManager implementation — Vulkan equivalent of Sirius's CUDA GPUBufferManager.
 */

#include "gpu/gpu_buffer_manager.hpp"
#include "gpu/gpu_context.hpp"
#include "log/logging.hpp"

#include <rasterdf/core/device_buffer.hpp>
#include <vulkan/vulkan.h>

#include <cstring>
#include <stdexcept>

namespace rasterdb {
namespace gpu {

bool GPUBufferManager::_initialized = false;

GPUBufferManager& GPUBufferManager::GetInstance(
    size_t cache_size_per_gpu,
    size_t processing_size_per_gpu,
    size_t processing_size_per_cpu)
{
  static GPUBufferManager instance(cache_size_per_gpu, processing_size_per_gpu, processing_size_per_cpu);
  return instance;
}

bool GPUBufferManager::is_initialized() {
  return _initialized;
}

GPUBufferManager::GPUBufferManager(size_t cache_size, size_t processing_size, size_t cpu_size)
    : cache_size_per_gpu(cache_size),
      processing_size_per_gpu(processing_size),
      processing_size_per_cpu(cpu_size),
      cpuProcessing(nullptr)
{
  RASTERDB_LOG_INFO("Initializing GPUBufferManager: cache={}MB, processing={}MB, staging={}MB",
                    cache_size / (1024 * 1024),
                    processing_size / (1024 * 1024),
                    cpu_size / (1024 * 1024));

  auto& ctx = gpu_context::instance();
  auto* mr = ctx.host_resource();  // vulkan_memory_resource (not pool) — handles both host and device

  // 1. GPU Cache: device-local buffer (equivalent of Sirius's cudaMalloc for gpuCache)
  //    Used for caching columns persistently across queries.
  _gpu_cache_alloc = mr->allocate(
      cache_size,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT |
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
  RASTERDB_LOG_INFO("  GPU cache: {}MB device-local buffer allocated (addr=0x{:x})",
                    cache_size / (1024 * 1024), _gpu_cache_alloc.address);

  // 2. CPU Staging: host-visible, persistently mapped (equivalent of Sirius's cudaHostAlloc)
  //    Used for CPU→GPU data transfers. Persistently mapped = no map/unmap overhead.
  _cpu_staging_alloc = mr->allocate(
      cpu_size,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
  if (!_cpu_staging_alloc.mapped_ptr) {
    mr->deallocate(_gpu_cache_alloc);
    throw std::runtime_error("GPUBufferManager: failed to map CPU staging buffer");
  }
  cpuProcessing = reinterpret_cast<uint8_t*>(_cpu_staging_alloc.mapped_ptr);
  RASTERDB_LOG_INFO("  CPU staging: {}MB host-visible buffer allocated (mapped={})",
                    cpu_size / (1024 * 1024), (void*)cpuProcessing);

  // 3. GPU Processing: device-local buffer (equivalent of Sirius's RMM pool for gpuProcessing)
  //    Used for intermediate GPU computation results (reset between queries).
  _gpu_processing_alloc = mr->allocate(
      processing_size,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT |
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
  RASTERDB_LOG_INFO("  GPU processing: {}MB device-local buffer allocated (addr=0x{:x})",
                    processing_size / (1024 * 1024), _gpu_processing_alloc.address);

  _initialized = true;
  RASTERDB_LOG_INFO("GPUBufferManager initialized successfully");
}

GPUBufferManager::~GPUBufferManager() {
  auto& ctx = gpu_context::instance();
  auto* mr = ctx.host_resource();
  // Deallocate in reverse order
  if (_gpu_processing_alloc.buffer) mr->deallocate(_gpu_processing_alloc);
  if (_cpu_staging_alloc.buffer) mr->deallocate(_cpu_staging_alloc);
  if (_gpu_cache_alloc.buffer) mr->deallocate(_gpu_cache_alloc);
  _initialized = false;
}

void GPUBufferManager::ResetBuffer() {
  gpuProcessingPointer.store(0, std::memory_order_relaxed);
  cpuProcessingPointer.store(0, std::memory_order_relaxed);
  RASTERDB_LOG_DEBUG("GPUBufferManager: processing buffers reset");
}

void GPUBufferManager::ResetCache() {
  gpuCachingPointer.store(0, std::memory_order_relaxed);
  cached_columns.clear();
  RASTERDB_LOG_DEBUG("GPUBufferManager: cache reset");
}

void GPUBufferManager::Print() {
  fprintf(stderr, "[GPUBufferManager]\n");
  fprintf(stderr, "  GPU cache:      %zu / %zu MB (%.1f%%)\n",
          gpuCachingPointer.load() / (1024*1024), cache_size_per_gpu / (1024*1024),
          100.0 * gpuCachingPointer.load() / cache_size_per_gpu);
  fprintf(stderr, "  GPU processing: %zu / %zu MB (%.1f%%)\n",
          gpuProcessingPointer.load() / (1024*1024), processing_size_per_gpu / (1024*1024),
          100.0 * gpuProcessingPointer.load() / processing_size_per_gpu);
  fprintf(stderr, "  CPU staging:    %zu / %zu MB (%.1f%%)\n",
          cpuProcessingPointer.load() / (1024*1024), processing_size_per_cpu / (1024*1024),
          100.0 * cpuProcessingPointer.load() / processing_size_per_cpu);
  fprintf(stderr, "  Cached columns: ");
  size_t total = 0;
  for (auto& [tbl, cols] : cached_columns) {
    for (auto& [col, info] : cols) {
      fprintf(stderr, "%s.%s (%zuMB), ", tbl.c_str(), col.c_str(), info.byte_size / (1024*1024));
      total += info.byte_size;
    }
  }
  if (total == 0) fprintf(stderr, "(none)");
  fprintf(stderr, "\n");
}

bool GPUBufferManager::checkIfColumnCached(const std::string& table_name,
                                           const std::string& column_name) {
  auto tbl_it = cached_columns.find(table_name);
  if (tbl_it == cached_columns.end()) return false;
  return tbl_it->second.find(column_name) != tbl_it->second.end();
}

GPUBufferManager::CachedColumn* GPUBufferManager::getCachedColumn(
    const std::string& table_name, const std::string& column_name) {
  auto tbl_it = cached_columns.find(table_name);
  if (tbl_it == cached_columns.end()) return nullptr;
  auto col_it = tbl_it->second.find(column_name);
  if (col_it == tbl_it->second.end()) return nullptr;
  return &col_it->second;
}

void GPUBufferManager::batchTransfer(
    gpu_context& ctx,
    const std::vector<size_t>& src_offsets,
    const std::vector<size_t>& dst_offsets,
    const std::vector<size_t>& sizes)
{
  if (src_offsets.empty()) return;

  // We wrap the existing gpuCache buffer into a dummy device_buffer to use
  // its built-in volk-initialized batch copy helper. This is required because
  // the duckdb extension lacks initialized volk function pointers.
  rasterdf::device_buffer dummy_cache(
      ctx.host_resource(),
      _gpu_cache_alloc.buffer,
      _gpu_cache_alloc.allocation,
      cache_size_per_gpu);

  dummy_cache.batch_copy_from_host(
      src_offsets, dst_offsets, sizes, 
      cpuStagingBuffer(), 
      ctx.device(), ctx.queue(), ctx.command_pool());

  dummy_cache.release();
}

} // namespace gpu
} // namespace rasterdb
