/*
 * Copyright 2025, RasterDB Contributors.
 * GPU Table implementation — upload/download DuckDB data ↔ GPU device buffers.
 */

#include "gpu/gpu_table.hpp"
#include "gpu/gpu_buffer_manager.hpp"
#include "log/logging.hpp"

#include <duckdb/common/types/vector.hpp>
#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>
#include <future>
#include <latch>

#include <vulkan/vulkan.h>

namespace rasterdb {
namespace gpu {

gpu_column allocate_column(gpu_context& ctx, rasterdf::data_type type, rasterdf::size_type num_rows)
{
  size_t bytes = static_cast<size_t>(num_rows) * rdf_type_size(type.id);
  gpu_column col;
  col.type = type;
  col.num_rows = num_rows;
  col.data = rasterdf::device_buffer(ctx.workspace_mr(), bytes);
  return col;
}

rasterdf::device_buffer allocate_buffer(gpu_context& ctx, size_t bytes)
{
  return rasterdf::device_buffer(ctx.workspace_mr(), bytes);
}

size_t download_column(gpu_context& ctx, const gpu_column& col, void* dst, size_t dst_size)
{
  size_t bytes = col.byte_size();
  if (bytes > dst_size) {
    throw std::runtime_error("download_column: destination buffer too small");
  }
  if (bytes == 0) return 0;

  if (col.cached_address != 0) {
    // Column is backed by GPUBufferManager cache — use staging copy
    // Allocate a temporary staging buffer for download
    auto* mr = ctx.host_resource();
    auto staging = mr->allocate(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
    if (!staging.mapped_ptr) {
      mr->deallocate(staging);
      throw std::runtime_error("download_column: failed to map staging buffer");
    }

    // Record copy from cached buffer → staging
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = ctx.command_pool();
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmdBuf;
    vkAllocateCommandBuffers(ctx.device(), &allocInfo, &cmdBuf);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmdBuf, &beginInfo);

    // Compute offset within the cache buffer
    auto& bufMgr = GPUBufferManager::GetInstance();
    VkDeviceSize srcOffset = col.cached_address - bufMgr.gpuCacheAddress();

    VkBufferCopy region{};
    region.srcOffset = srcOffset;
    region.dstOffset = 0;
    region.size = bytes;
    vkCmdCopyBuffer(cmdBuf, col.cached_buffer, staging.buffer, 1, &region);

    vkEndCommandBuffer(cmdBuf);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuf;
    vkQueueSubmit(ctx.queue(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx.queue());

    std::memcpy(dst, staging.mapped_ptr, bytes);

    vkFreeCommandBuffers(ctx.device(), ctx.command_pool(), 1, &cmdBuf);
    mr->deallocate(staging);
  } else {
    // Standard path: device_buffer owns the data
    const_cast<rasterdf::device_buffer&>(col.data).copy_to_host(
      dst, bytes, ctx.device(), ctx.queue(), ctx.command_pool());
  }
  return bytes;
}

std::vector<std::vector<uint8_t>> batch_download_columns(gpu_context& ctx, const gpu_table& table)
{
  size_t ncols = table.num_columns();
  std::vector<std::vector<uint8_t>> host_bufs(ncols);

  if (table.num_rows() == 0) return host_bufs;

  // Compute per-column byte sizes, handle host-only / ReBAR columns immediately.
  struct DeviceCol { size_t col_idx; size_t bytes; };
  std::vector<DeviceCol> device_cols;
  std::vector<size_t> col_bytes(ncols);
  size_t max_device_col_bytes = 0;

  for (size_t c = 0; c < ncols; c++) {
    auto& col = table.col(c);
    col_bytes[c] = static_cast<size_t>(col.num_rows) * rdf_type_size(col.type.id);
    host_bufs[c].resize(col_bytes[c]);

    if (col.is_host_only) {
      std::memcpy(host_bufs[c].data(), col.host_data.data(), col_bytes[c]);
    } else if (col.data.mapped_data()) {
      // ReBAR path: direct memcpy from host-visible device memory
      std::memcpy(host_bufs[c].data(), col.data.mapped_data(), col_bytes[c]);
    } else {
      device_cols.push_back({c, col_bytes[c]});
      max_device_col_bytes = std::max(max_device_col_bytes, col_bytes[c]);
    }
  }

  if (device_cols.empty()) return host_bufs;

  // Allocate ONE reusable staging buffer sized to the largest device-resident
  // column. A single 1.8GB staging allocation was unstable on this path, while
  // the existing per-column 600MB download_column path is known to work. This
  // preserves the safe copy size while eliminating repeated VMA alloc/dealloc.
  auto* mr = ctx.host_resource();
  auto staging = mr->allocate(max_device_col_bytes,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
  if (!staging.mapped_ptr) {
    mr->deallocate(staging);
    throw std::runtime_error("batch_download_columns: failed to map staging buffer");
  }

  for (auto& dc : device_cols) {
    auto& col = table.col(dc.col_idx);
    if (col.cached_buffer != VK_NULL_HANDLE) {
      download_column(ctx, col, host_bufs[dc.col_idx].data(), dc.bytes);
      continue;
    }

    const_cast<rasterdf::device_buffer&>(col.data).copy_to_host_with_staging(
      host_bufs[dc.col_idx].data(), dc.bytes, staging,
      ctx.device(), ctx.queue(), ctx.command_pool());
  }

  mr->deallocate(staging);
  return host_bufs;
}

/// Upload a flat host buffer to a new gpu_column.
static gpu_column upload_column(gpu_context& ctx, rasterdf::data_type type,
                                rasterdf::size_type num_rows, const void* host_data)
{
  gpu_column col = allocate_column(ctx, type, num_rows);
  size_t bytes = col.byte_size();
  if (bytes > 0 && host_data) {
    col.data.copy_from_host(host_data, bytes, ctx.device(), ctx.queue(), ctx.command_pool());
  }
  return col;
}

/// Extract flat data from a DuckDB Vector into a contiguous host buffer.
/// Returns the number of valid rows extracted.
static rasterdf::size_type flatten_vector(const duckdb::Vector& vec,
                                           rasterdf::size_type count,
                                           rasterdf::data_type rdf_type,
                                           std::vector<uint8_t>& out_buf)
{
  size_t elem_size = rdf_type_size(rdf_type.id);
  size_t offset = out_buf.size();
  out_buf.resize(offset + static_cast<size_t>(count) * elem_size);

  auto data_ptr = reinterpret_cast<const uint8_t*>(vec.GetData());
  std::memcpy(out_buf.data() + offset, data_ptr, static_cast<size_t>(count) * elem_size);
  return count;
}

/// Overload: write directly to a raw destination pointer. Returns bytes written.
static size_t flatten_vector_raw(const duckdb::Vector& vec,
                                 rasterdf::size_type count,
                                 rasterdf::data_type rdf_type,
                                 uint8_t* dst)
{
  size_t elem_size = rdf_type_size(rdf_type.id);
  size_t bytes = static_cast<size_t>(count) * elem_size;
  auto data_ptr = reinterpret_cast<const uint8_t*>(vec.GetData());
  std::memcpy(dst, data_ptr, bytes);
  return bytes;
}

std::unique_ptr<gpu_table> gpu_table::from_data_chunks(
  gpu_context& ctx,
  const std::vector<duckdb::LogicalType>& types,
  std::vector<std::unique_ptr<duckdb::DataChunk>>& chunks)
{
  auto table = std::make_unique<gpu_table>();
  table->duckdb_types = types;

  if (chunks.empty()) {
    table->_num_rows = 0;
    table->columns.resize(types.size());
    for (size_t c = 0; c < types.size(); c++) {
      table->columns[c].type = to_rdf_type(types[c]);
      table->columns[c].num_rows = 0;
    }
    return table;
  }

  size_t num_cols = types.size();

  // First pass: compute total rows and validate types
  rasterdf::size_type total_rows = 0;
  std::vector<rasterdf::data_type> rdf_types(num_cols);
  for (size_t c = 0; c < num_cols; c++) {
    rdf_types[c] = to_rdf_type(types[c]); // throws for unsupported types
  }
  for (auto& chunk : chunks) {
    total_rows += static_cast<rasterdf::size_type>(chunk->size());
  }

  if (total_rows == 0) {
    table->_num_rows = 0;
    table->columns.resize(num_cols);
    for (size_t c = 0; c < num_cols; c++) {
      table->columns[c].type = rdf_types[c];
      table->columns[c].num_rows = 0;
    }
    return table;
  }

  // Second pass: flatten each column across all chunks, then upload to GPU
  // Multi-threaded: process columns in parallel for faster CPU-side flattening
  table->columns.resize(num_cols);

  // Determine number of threads (cap at hardware concurrency, min 1)
  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) num_threads = 4;
  // Don't create more threads than columns
  if (num_threads > num_cols) num_threads = static_cast<unsigned int>(num_cols);

  if (num_cols == 1 || num_threads == 1) {
    // Single-threaded path for simplicity with few columns
    for (size_t c = 0; c < num_cols; c++) {
      std::vector<uint8_t> host_buf;
      host_buf.reserve(static_cast<size_t>(total_rows) * rdf_type_size(rdf_types[c].id));

      for (auto& chunk : chunks) {
        chunk->Flatten(); // ensure flat vectors
        flatten_vector(chunk->data[c], static_cast<rasterdf::size_type>(chunk->size()),
                       rdf_types[c], host_buf);
      }
      table->columns[c] = upload_column(ctx, rdf_types[c], total_rows, host_buf.data());
    }
  } else {
    // Multi-threaded path: parallelize across columns
    std::vector<std::vector<uint8_t>> host_bufs(num_cols);
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    // Create work items: each thread processes a subset of columns
    auto worker = [&](size_t start_col, size_t end_col) {
      for (size_t c = start_col; c < end_col; c++) {
        host_bufs[c].reserve(static_cast<size_t>(total_rows) * rdf_type_size(rdf_types[c].id));
        for (auto& chunk : chunks) {
          chunk->Flatten(); // ensure flat vectors
          flatten_vector(chunk->data[c], static_cast<rasterdf::size_type>(chunk->size()),
                         rdf_types[c], host_bufs[c]);
        }
      }
    };

    // Launch threads with column range assignments
    size_t cols_per_thread = num_cols / num_threads;
    size_t remainder = num_cols % num_threads;
    size_t start = 0;

    for (unsigned int t = 0; t < num_threads; t++) {
      size_t count = cols_per_thread + (t < remainder ? 1 : 0);
      if (count == 0) break;
      size_t end = start + count;
      threads.emplace_back(worker, start, end);
      start = end;
    }

    // Wait for all flattening to complete
    for (auto& t : threads) {
      t.join();
    }

    // Upload columns to GPU (can also be parallelized, but keep simple for now)
    for (size_t c = 0; c < num_cols; c++) {
      table->columns[c] = upload_column(ctx, rdf_types[c], total_rows, host_bufs[c].data());
    }
  }

  table->_num_rows = total_rows;
  RASTERDB_LOG_DEBUG("GPU table uploaded: {} rows x {} cols", total_rows, num_cols);
  return table;
}

// ============================================================================
// from_buffer_manager — upload via GPUBufferManager (pinned staging + batch transfer)
// ============================================================================

std::unique_ptr<gpu_table> gpu_table::from_buffer_manager(
  gpu_context& ctx,
  const std::string& table_name,
  const std::vector<std::string>& column_names,
  const std::vector<duckdb::LogicalType>& types,
  std::vector<std::unique_ptr<duckdb::DataChunk>>& chunks)
{
  auto& bufMgr = GPUBufferManager::GetInstance();
  auto table = std::make_unique<gpu_table>();
  table->duckdb_types = types;
  size_t num_cols = types.size();

  // Compute rasterdf types
  std::vector<rasterdf::data_type> rdf_types(num_cols);
  for (size_t c = 0; c < num_cols; c++) {
    rdf_types[c] = to_rdf_type(types[c]);
  }

  // Count total rows
  rasterdf::size_type total_rows = 0;
  for (auto& chunk : chunks) {
    total_rows += static_cast<rasterdf::size_type>(chunk->size());
  }

  table->columns.resize(num_cols);

  // First pass: identify which columns need uploading vs cached, pre-calculate offsets
  struct upload_col_info {
    size_t c;
    size_t col_bytes;
    size_t staging_off;
    uint8_t* staging_dst;
  };
  std::vector<upload_col_info> upload_cols;
  std::vector<size_t> upload_sizes;

  for (size_t c = 0; c < num_cols; c++) {
    size_t col_bytes = static_cast<size_t>(total_rows) * rdf_type_size(rdf_types[c].id);

    if (bufMgr.checkIfColumnCached(table_name, column_names[c])) {
      // Use cached column — zero-cost
      auto* cached = bufMgr.getCachedColumn(table_name, column_names[c]);
      table->columns[c].type = cached->type;
      table->columns[c].num_rows = static_cast<rasterdf::size_type>(cached->num_rows);
      table->columns[c].cached_address = bufMgr.gpuCacheAddress() + cached->gpu_offset;
      table->columns[c].cached_buffer = bufMgr.gpuCacheBuffer();
      RASTERDB_LOG_DEBUG("  col {} ({}) CACHED at offset {} ({} bytes)",
                         c, column_names[c], cached->gpu_offset, cached->byte_size);
    } else {
      // Pre-allocate staging + cache slot (single-threaded bump allocator)
      uint8_t* staging_dst = bufMgr.customVkHostAlloc<uint8_t>(col_bytes);
      size_t staging_off = static_cast<size_t>(staging_dst - bufMgr.cpuProcessing);
      [[maybe_unused]] size_t cache_off = bufMgr.customVkMalloc<uint8_t>(col_bytes, /*caching=*/true);

      upload_cols.push_back({c, col_bytes, staging_off, staging_dst});
      upload_sizes.push_back(col_bytes);

      // Set column metadata now (staging pointer is valid)
      table->columns[c].type = rdf_types[c];
      table->columns[c].num_rows = total_rows;
      table->columns[c].cached_address = bufMgr.cpuStagingAddress() + staging_off;
      table->columns[c].cached_buffer = bufMgr.cpuStagingBuffer();
    }
  }

  // Second pass: multi-threaded flattening into pre-allocated staging buffers
  if (!upload_cols.empty()) {
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    if (num_threads > upload_cols.size()) num_threads = static_cast<unsigned int>(upload_cols.size());

    if (upload_cols.size() == 1 || num_threads == 1) {
      // Single-threaded path
      for (auto& info : upload_cols) {
        size_t write_pos = 0;
        for (auto& chunk : chunks) {
          chunk->Flatten();
          write_pos += flatten_vector_raw(chunk->data[info.c],
                         static_cast<rasterdf::size_type>(chunk->size()),
                         rdf_types[info.c], info.staging_dst + write_pos);
        }
        RASTERDB_LOG_DEBUG("  col {} ({}) ZERO-COPY STAGING: offset={} bytes={}",
                           info.c, column_names[info.c], info.staging_off, info.col_bytes);
      }
    } else {
      // Multi-threaded path: parallelize across columns to flatten
      std::vector<std::thread> threads;
      threads.reserve(num_threads);

      auto worker = [&](size_t start_idx, size_t end_idx) {
        for (size_t i = start_idx; i < end_idx; i++) {
          auto& info = upload_cols[i];
          size_t write_pos = 0;
          for (auto& chunk : chunks) {
            chunk->Flatten();
            write_pos += flatten_vector_raw(chunk->data[info.c],
                           static_cast<rasterdf::size_type>(chunk->size()),
                           rdf_types[info.c], info.staging_dst + write_pos);
          }
        }
      };

      // Launch threads with work range assignments
      size_t cols_per_thread = upload_cols.size() / num_threads;
      size_t remainder = upload_cols.size() % num_threads;
      size_t start = 0;

      for (unsigned int t = 0; t < num_threads; t++) {
        size_t count = cols_per_thread + (t < remainder ? 1 : 0);
        if (count == 0) break;
        size_t end = start + count;
        threads.emplace_back(worker, start, end);
        start = end;
      }

      for (auto& t : threads) {
        t.join();
      }

      for (auto& info : upload_cols) {
        RASTERDB_LOG_DEBUG("  col {} ({}) ZERO-COPY STAGING: offset={} bytes={}",
                           info.c, column_names[info.c], info.staging_off, info.col_bytes);
      }
    }
  }

  // Skip batchTransfer entirely! Zero-Copy Optimization matches Sirius 0ms upload.

  table->_num_rows = total_rows;
  RASTERDB_LOG_DEBUG("GPU table from_buffer_manager: {} rows x {} cols ({} uploaded, {} cached)",
                     total_rows, num_cols, upload_sizes.size(), num_cols - upload_sizes.size());
  return table;
}

void gpu_table::append_chunk(gpu_context& ctx, const duckdb::DataChunk& chunk,
                              const std::vector<duckdb::LogicalType>& types)
{
  (void)ctx; (void)chunk; (void)types;
  throw duckdb::NotImplementedException("gpu_table::append_chunk not yet implemented");
}

gpu_column gpu_column_from_rdf(rasterdf::column&& col)
{
  gpu_column result;
  result.type = col.type();
  result.num_rows = col.size();
  result.data = std::move(col.device_data());
  return result;
}

std::unique_ptr<gpu_table> gpu_table_from_rdf(std::unique_ptr<rasterdf::table> tbl,
                                               const std::vector<duckdb::LogicalType>& duckdb_types)
{
  auto result = std::make_unique<gpu_table>();
  result->duckdb_types = duckdb_types;

  auto cols = tbl->extract();
  result->columns.resize(cols.size());
  for (size_t i = 0; i < cols.size(); i++) {
    result->columns[i] = gpu_column_from_rdf(std::move(*cols[i]));
  }
  return result;
}

rasterdf::table_view gpu_table::view() const
{
  std::vector<rasterdf::column_view> views;
  views.reserve(columns.size());
  for (auto& col : columns) {
    views.push_back(col.view());
  }
  return rasterdf::table_view(views);
}

} // namespace gpu
} // namespace rasterdb
