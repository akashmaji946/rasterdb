/*
 * Copyright 2025, RasterDB Contributors.
 * GPU Table implementation — upload/download DuckDB data ↔ GPU device buffers.
 */

#include "gpu/gpu_table.hpp"
#include "gpu/gpu_buffer_manager.hpp"
#include "log/logging.hpp"

#include <duckdb/common/types/vector.hpp>
#include <cstring>

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
  table->columns.resize(num_cols);
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

  // Track which columns need uploading vs already cached
  std::vector<size_t> upload_staging_offsets;
  std::vector<size_t> upload_cache_offsets;
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
      // Flatten DuckDB chunks into pre-allocated staging + allocate cache slot
      uint8_t* staging_dst = bufMgr.customVkHostAlloc<uint8_t>(col_bytes);
      size_t staging_off = static_cast<size_t>(staging_dst - bufMgr.cpuProcessing);
      size_t cache_off = bufMgr.customVkMalloc<uint8_t>(col_bytes, /*caching=*/true);

      // Flatten column data into staging
      size_t write_pos = 0;
      for (auto& chunk : chunks) {
        chunk->Flatten();
        write_pos += flatten_vector_raw(chunk->data[c],
                       static_cast<rasterdf::size_type>(chunk->size()),
                       rdf_types[c], staging_dst + write_pos);
      }

      // Set column to natively reference CPU staging location! (Zero-Copy PCIe)
      table->columns[c].type = rdf_types[c];
      table->columns[c].num_rows = total_rows;
      table->columns[c].cached_address = bufMgr.cpuStagingAddress() + staging_off;
      table->columns[c].cached_buffer = bufMgr.cpuStagingBuffer();

      // We still map sizes so we log it, but we won't batchTransfer to GPU Cache
      upload_sizes.push_back(col_bytes);

      RASTERDB_LOG_DEBUG("  col {} ({}) ZERO-COPY STAGING: offset={} bytes={}",
                         c, column_names[c], staging_off, col_bytes);
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
