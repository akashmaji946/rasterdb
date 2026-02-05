/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// sirius
#include <data/host_parquet_representation.hpp>
#include <data/host_parquet_representation_converters.hpp>

// cucascade
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_space.hpp>

// cudf
#include <cudf/io/parquet.hpp>

// rmm
#include <rmm/detail/error.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

// standard library
#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace sirius {

namespace detail {

/**
 * @brief Convert host_parquet_representation to gpu_table_representation
 */
std::unique_ptr<cucascade::idata_representation> convert_host_parquet_to_gpu(
  cucascade::idata_representation& source,
  cucascade::memory::memory_space const* target_memory_space,
  rmm::cuda_stream_view stream)
{
  auto& host_src          = source.cast<host_parquet_representation>();
  auto const& byte_ranges = host_src.get_column_chunk_byte_ranges();
  auto& reader            = host_src.get_parquet_reader();

  rmm::device_async_resource_ref mr_ref(target_memory_space->get_default_allocator());
  int previous_device = -1;
  RMM_CUDA_TRY(cudaGetDevice(&previous_device));
  RMM_CUDA_TRY(cudaSetDevice(target_memory_space->get_device_id()));

  auto const& allocation = host_src.get_column_chunks();

  // Allocate the device buffers
  std::vector<rmm::device_buffer> device_buffers;
  device_buffers.reserve(byte_ranges.size());
  std::for_each(
    byte_ranges.begin(), byte_ranges.end(), [&](cudf::io::text::byte_range_info const& byte_range) {
      device_buffers.emplace_back(byte_range.size(), stream, mr_ref);
    });

  // Copy each chunk from multiple-block host allocation into its device buffer
  auto const block_size = allocation->block_size();
  for (size_t i = 0; i < byte_ranges.size(); ++i) {
    auto const& byte_range = byte_ranges[i];
    auto const chunk_size  = byte_range.size();
    auto const chunk_off   = byte_range.offset();  // packed-buffer offset

    size_t remaining   = chunk_size;
    size_t dst_off     = 0;
    size_t block_index = chunk_off / block_size;
    size_t block_off   = chunk_off % block_size;

    while (remaining > 0) {
      auto src_block     = allocation->at(block_index);  // span<std::byte>
      auto const avail   = block_size - block_off;
      auto const to_copy = std::min(remaining, avail);

      RMM_CUDA_TRY(
        cudaMemcpyAsync(static_cast<std::uint8_t*>(device_buffers[i].data()) + dst_off,
                        reinterpret_cast<std::uint8_t const*>(src_block.data()) + block_off,
                        to_copy,
                        cudaMemcpyHostToDevice,
                        stream.value()));

      remaining -= to_copy;
      dst_off += to_copy;
      ++block_index;
      block_off = 0;
    }
  }

  auto result =
    reader.materialize_payload_columns(host_src.get_rg_span(),
                                       std::move(device_buffers),
                                       cudf::column_view{},
                                       cudf::io::parquet::experimental::use_data_page_mask::NO,
                                       host_src.get_reader_options(),
                                       stream);
  auto new_table = std::move(*result.tbl);  // Discard metadata
  stream.synchronize();

  RMM_CUDA_TRY(cudaSetDevice(previous_device));
  return std::make_unique<cucascade::gpu_table_representation>(
    std::move(new_table), *const_cast<cucascade::memory::memory_space*>(target_memory_space));
}

/**
 * @brief Convert host_parquet_representation to host_parquet_representation (cross-host copy)
 */
std::unique_ptr<cucascade::idata_representation> convert_host_parquet_to_host_parquet(
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  rmm::cuda_stream_view /* stream */)
{
  auto& host_src       = source.cast<host_parquet_representation>();
  auto const data_size = host_src.get_size_in_bytes();

  assert(source.get_device_id() != target_memory_space->get_device_id());
  auto* mr = target_memory_space
               ->get_memory_resource_as<cucascade::memory::fixed_size_host_memory_resource>();
  if (mr == nullptr) {
    throw std::runtime_error(
      "Target HOST memory_space does not have a fixed_size_host_memory_resource");
  }

  auto const& src_allocation  = host_src.get_column_chunks();
  auto dst_allocation         = mr->allocate_multiple_blocks(data_size);
  size_t src_block_index      = 0;
  size_t src_block_offset     = 0;
  size_t dst_block_index      = 0;
  size_t dst_block_offset     = 0;
  size_t const src_block_size = src_allocation->block_size();
  size_t const dst_block_size = dst_allocation->block_size();
  size_t copied               = 0;
  while (copied < data_size) {
    size_t remaining     = data_size - copied;
    size_t src_avail     = src_block_size - src_block_offset;
    size_t dst_avail     = dst_block_size - dst_block_offset;
    size_t bytes_to_copy = std::min({remaining, src_avail, dst_avail});
    auto* src_ptr        = src_allocation->at(src_block_index).data() + src_block_offset;
    auto* dst_ptr        = dst_allocation->at(dst_block_index).data() + dst_block_offset;
    std::memcpy(dst_ptr, src_ptr, bytes_to_copy);
    copied += bytes_to_copy;
    src_block_offset += bytes_to_copy;
    dst_block_offset += bytes_to_copy;
    if (src_block_offset == src_block_size) {
      src_block_index++;
      src_block_offset = 0;
    }
    if (dst_block_offset == dst_block_size) {
      dst_block_index++;
      dst_block_offset = 0;
    }
  }
  return std::make_unique<host_parquet_representation>(
    const_cast<cucascade::memory::memory_space*>(target_memory_space),
    std::move(dst_allocation),
    std::move(host_src.move_parquet_reader()),
    host_src.get_reader_options(),
    std::move(host_src.get_row_group_indices()),
    std::move(host_src.get_column_chunk_byte_ranges()),
    data_size);
}

}  // namespace detail

void register_parquet_converters(cucascade::representation_converter_registry& registry)
{
  // HOST Parquet -> GPU
  if (!registry.has_converter<host_parquet_representation, cucascade::gpu_table_representation>()) {
    registry.register_converter<host_parquet_representation, cucascade::gpu_table_representation>(
      detail::convert_host_parquet_to_gpu);
  }

  // HOST Parquet -> HOST Parquet (cross-host copy)
  if (!registry.has_converter<host_parquet_representation, host_parquet_representation>()) {
    registry.register_converter<host_parquet_representation, host_parquet_representation>(
      detail::convert_host_parquet_to_host_parquet);
  }
}

}  // namespace sirius
