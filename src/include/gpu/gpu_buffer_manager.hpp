/*
 * Copyright 2025, RasterDB Contributors.
 * GPUBufferManager — Pre-allocated GPU buffer management (Vulkan equivalent of Sirius's CUDA GPUBufferManager).
 *
 * Manages three pre-allocated memory regions:
 *   1. gpuCache     — device-local, for caching table columns across queries
 *   2. cpuProcessing — host-visible (persistently mapped), staging for CPU→GPU transfers
 *   3. gpuProcessing — device-local, for intermediate GPU computation
 *
 * Uses bump allocators (customVkMalloc, customVkHostAlloc) for zero-overhead allocation.
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <rasterdf/memory/memory_resource.hpp>
#include <rasterdf/core/types.hpp>

namespace rasterdb {
namespace gpu {

class gpu_context;

class GPUBufferManager {
public:
  /// Get the singleton instance.  First call must provide sizes > 0.
  static GPUBufferManager& GetInstance(
      size_t cache_size_per_gpu      = 0,
      size_t processing_size_per_gpu = 0,
      size_t processing_size_per_cpu = 0);

  // Delete copy/move
  GPUBufferManager(const GPUBufferManager&)            = delete;
  GPUBufferManager& operator=(const GPUBufferManager&) = delete;

  /// Reset processing bump pointers (called between queries).
  void ResetBuffer();

  /// Reset cache bump pointer and invalidate all cached columns.
  void ResetCache();

  /// Print memory usage summary to stderr.
  void Print();

  // ---- Pre-allocated region pointers ----
  // cpuProcessing is host-mapped (staging); gpuCache/gpuProcessing are device-only.
  uint8_t* cpuProcessing;        // host-visible, persistently mapped

  // Bump pointers (byte offsets into each region)
  std::atomic<size_t> gpuCachingPointer{0};
  std::atomic<size_t> gpuProcessingPointer{0};
  std::atomic<size_t> cpuProcessingPointer{0};

  size_t cache_size_per_gpu;
  size_t processing_size_per_gpu;
  size_t processing_size_per_cpu;

  // ---- Bump allocators (matching Sirius API) ----

  /// Allocate `count` elements of type T from GPU cache or processing memory.
  /// If caching=true, allocates from gpuCache (persistent across queries).
  /// If caching=false, allocates from gpuProcessing (reset between queries).
  /// Returns byte offset into the corresponding VkBuffer.
  template<typename T>
  size_t customVkMalloc(size_t count, bool caching);

  /// Allocate `count` elements of type T from CPU staging (host-visible, pinned).
  /// Returns a host pointer within cpuProcessing.
  template<typename T>
  T* customVkHostAlloc(size_t count);

  // ---- Column caching ----
  struct CachedColumn {
    size_t gpu_offset;       // Byte offset within gpuCache VkBuffer
    size_t num_rows;
    size_t byte_size;
    rasterdf::data_type type;
  };

  // table_name -> column_name -> CachedColumn
  std::map<std::string, std::map<std::string, CachedColumn>> cached_columns;

  bool checkIfColumnCached(const std::string& table_name,
                           const std::string& column_name);
  CachedColumn* getCachedColumn(const std::string& table_name,
                                const std::string& column_name);

  // ---- Vulkan buffer handles (for vkCmdCopyBuffer) ----
  VkBuffer gpuCacheBuffer()     const { return _gpu_cache_alloc.buffer; }
  VkBuffer cpuStagingBuffer()   const { return _cpu_staging_alloc.buffer; }
  VkBuffer gpuProcessingBuffer() const { return _gpu_processing_alloc.buffer; }

  VkDeviceAddress gpuCacheAddress()      const { return _gpu_cache_alloc.address; }
  VkDeviceAddress gpuProcessingAddress() const { return _gpu_processing_alloc.address; }

  /// Batch transfer: multiple regions from cpuProcessing → gpuCache in ONE vkQueueSubmit.
  void batchTransfer(gpu_context& ctx,
                     const std::vector<size_t>& src_offsets,  // offsets in cpuProcessing
                     const std::vector<size_t>& dst_offsets,  // offsets in gpuCache
                     const std::vector<size_t>& sizes);

  static bool is_initialized();

private:
  GPUBufferManager(size_t cache_size, size_t processing_size, size_t cpu_size);
  ~GPUBufferManager();

  // VMA allocations (one big buffer per region)
  rasterdf::allocation_info _gpu_cache_alloc{};
  rasterdf::allocation_info _cpu_staging_alloc{};
  rasterdf::allocation_info _gpu_processing_alloc{};

  static bool _initialized;
};

// ---- Template implementations ----

template<typename T>
size_t GPUBufferManager::customVkMalloc(size_t count, bool caching) {
  size_t alloc_bytes = count * sizeof(T);
  // Align to 256 bytes (Vulkan minStorageBufferOffsetAlignment)
  constexpr size_t ALIGNMENT = 256;
  alloc_bytes = (alloc_bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);

  if (caching) {
    size_t start = gpuCachingPointer.fetch_add(alloc_bytes, std::memory_order_relaxed);
    if (start + alloc_bytes > cache_size_per_gpu) {
      gpuCachingPointer.fetch_sub(alloc_bytes, std::memory_order_relaxed);
      throw std::runtime_error("GPUBufferManager: out of GPU cache memory");
    }
    return start;
  } else {
    size_t start = gpuProcessingPointer.fetch_add(alloc_bytes, std::memory_order_relaxed);
    if (start + alloc_bytes > processing_size_per_gpu) {
      gpuProcessingPointer.fetch_sub(alloc_bytes, std::memory_order_relaxed);
      throw std::runtime_error("GPUBufferManager: out of GPU processing memory");
    }
    return start;
  }
}

template<typename T>
T* GPUBufferManager::customVkHostAlloc(size_t count) {
  size_t alloc_bytes = count * sizeof(T);
  // Align to 256 bytes
  constexpr size_t ALIGNMENT = 256;
  alloc_bytes = (alloc_bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);

  size_t start = cpuProcessingPointer.fetch_add(alloc_bytes, std::memory_order_relaxed);
  if (start + alloc_bytes > processing_size_per_cpu) {
    cpuProcessingPointer.fetch_sub(alloc_bytes, std::memory_order_relaxed);
    throw std::runtime_error("GPUBufferManager: out of CPU staging memory");
  }
  return reinterpret_cast<T*>(cpuProcessing + start);
}

} // namespace gpu
} // namespace rasterdb
