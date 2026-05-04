/*
 * Copyright 2025, RasterDB Contributors.
 * Split from src/gpu/gpu_executor.cpp.
 */

#include "gpu/gpu_executor_internal.hpp"

namespace rasterdb {
namespace gpu {

// ============================================================================
// FILTER — evaluate WHERE clause predicates on GPU
// ============================================================================
std::unique_ptr<gpu_table> gpu_executor::execute_filter(duckdb::LogicalFilter& op)
{
  RASTERDB_LOG_DEBUG("GPU execute_filter");
  D_ASSERT(op.children.size() == 1);
  auto input = execute_operator(*op.children[0]);

  RASTERDB_LOG_DEBUG("Filter input: {} rows x {} cols", input->num_rows(), input->num_columns());
  for (size_t c = 0; c < input->num_columns(); c++) {
    RASTERDB_LOG_DEBUG("  input col {}: {} rows, type={}, addr={}", c, input->col(c).num_rows,
                       static_cast<int>(input->col(c).type.id), input->col(c).address());
  }

  RASTERDB_LOG_DEBUG("[RDB_DEBUG] FILTER: {} expressions, {} rows x {} cols",
                     op.expressions.size(), input->num_rows(), input->num_columns());
  stage_timer t("  filter (total)");

  if (input->num_rows() == 0) return input;

  // Evaluate each filter expression to produce a boolean (int32 0/1) mask
  gpu_column mask;
  bool first = true;

  auto t_compare_start = std::chrono::high_resolution_clock::now();
  for (auto& expr : op.expressions) {
    gpu_column expr_mask = evaluate_comparison(*input, *expr);

    if (first) {
      mask = std::move(expr_mask);
      first = false;
    } else {
      // AND masks: multiply element-wise (0/1 * 0/1 = AND)
      auto result = allocate_column(_ctx, {rasterdf::type_id::INT32}, mask.num_rows);
      binary_op_push_constants pc{};
      pc.input_a = mask.address();
      pc.input_b = expr_mask.address();
      pc.output_addr = result.address();
      pc.size = static_cast<uint32_t>(mask.num_rows);
      pc.op = 2;       // MUL
      pc.scalar_val = 0;
      pc.mode = 0;     // COL_COL
      pc.debug_mode = 0;
      pc.type_id = static_cast<int32_t>(rasterdf::ShaderTypeId::INT32);  // int32
      _ctx.dispatcher().dispatch_binary_op(pc);
      mask = std::move(result);
    }
  }

  {
    auto t_compare_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_compare_end - t_compare_start).count();
    RASTERDB_LOG_DEBUG("[TIMER]     filter_compare             {:8.2f} ms", ms);
  }

  // Apply the mask to compact the table
  auto t_compact_start = std::chrono::high_resolution_clock::now();
  auto result = apply_filter_mask(*input, mask);
  {
    auto t_compact_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_compact_end - t_compact_start).count();
    RASTERDB_LOG_DEBUG("[TIMER]     filter_compact             {:8.2f} ms", ms);
  }
  RASTERDB_LOG_DEBUG("[RDB_DEBUG] filter: {} rows => {} rows",
                     input->num_rows(), result->num_rows());
  return result;
}

// ============================================================================
// Apply boolean mask — GPU-side order-preserving stream compaction.
// Uses prefix scan on mask, then scatter_if per column. No CPU transfer.
// ============================================================================

std::unique_ptr<gpu_table> gpu_executor::apply_filter_mask(const gpu_table& input, gpu_column& mask)
{
  uint32_t n = static_cast<uint32_t>(input.num_rows());
  auto& disp = _ctx.dispatcher();
  auto* mr = _ctx.workspace_mr();

  // --- Step 1: GPU-side copy mask → prefix buffer (no host round-trip) ----
  // prefix_buf[0..N-1] = mask[0..N-1], prefix_buf[N] = 0 (sentinel)
  uint32_t prefix_count = n + 1;

  rasterdf::device_buffer prefix_buf(
    mr, prefix_count * sizeof(uint32_t),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

  // GPU-to-GPU copy: mask[0..N-1] → prefix_buf[0..N-1]
  disp.copy_buffer(mask.data.buffer(), prefix_buf.buffer(), n * sizeof(uint32_t),
                   mask.data.offset(), prefix_buf.offset());
  // Zero the sentinel element prefix_buf[N]
  disp.fill_buffer(prefix_buf.buffer(), 0u, sizeof(uint32_t),
                   prefix_buf.offset() + n * sizeof(uint32_t));

  // --- Step 2: 3-pass prefix scan on prefix_buf ---------------------------
  uint32_t scan_wg = 256;
  uint32_t scan_groups = (prefix_count + scan_wg - 1) / scan_wg;

  rasterdf::device_buffer block_sums(
    mr, scan_groups * sizeof(uint32_t),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

  // Device buffer for total_sum (written by scan_global shader)
  rasterdf::device_buffer total_sum_buf(
    mr, sizeof(uint32_t),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

  prefix_scan_pc scan_pc{};
  scan_pc.data_ptr = prefix_buf.data();
  scan_pc.block_sums_ptr = block_sums.data();
  scan_pc.total_sum_ptr = total_sum_buf.data();
  scan_pc.numElements = prefix_count;
  scan_pc.blockCount = scan_groups;

  disp.dispatch_prefix_scan_local(scan_pc, scan_groups);
  disp.dispatch_prefix_scan_global(scan_pc);
  disp.dispatch_prefix_scan_add(scan_pc, scan_groups);

  // --- Step 3: Read total count via copy_to_host --------------------------
  uint32_t output_count = 0;
  total_sum_buf.copy_to_host(&output_count, sizeof(uint32_t),
                             _ctx.device(), _ctx.queue(), _ctx.command_pool());

  RASTERDB_LOG_DEBUG("Filter (GPU compaction): {} -> {} rows", n, output_count);

  auto result = std::make_unique<gpu_table>();
  result->duckdb_types = input.duckdb_types;
  result->columns.resize(input.num_columns());

  if (output_count == 0) {
    for (size_t c = 0; c < input.num_columns(); c++) {
      result->columns[c].type = input.col(c).type;
      result->columns[c].num_rows = 0;
    }
    RASTERDB_LOG_DEBUG("Filter result: 0 rows (early return)");
    return result;
  }

  // --- Step 4: Scatter each column using prefix sums ----------------------
  uint32_t scatter_groups = (n + 255) / 256;
  
  // Skip GPU scatter for very small output tables (driver crash on tiny dispatches)
  // For tiny outputs, use CPU-based scatter
  if (output_count < 256) {
    RASTERDB_LOG_DEBUG("Filter (CPU scatter for tiny output): {} -> {} rows", n, output_count);
    for (size_t c = 0; c < input.num_columns(); c++) {
      auto& in_col = input.col(c);
      result->columns[c] = allocate_column(_ctx, in_col.type,
                                            static_cast<rasterdf::size_type>(output_count));
      
      // Copy prefix to host to find which rows passed
      std::vector<uint32_t> host_prefix(prefix_count);
      prefix_buf.copy_to_host(host_prefix.data(), prefix_count * sizeof(uint32_t),
                             _ctx.device(), _ctx.queue(), _ctx.command_pool());
      
      // Copy input column to host (need non-const access)
      size_t elem_size = rdf_type_size(in_col.type.id);
      std::vector<uint8_t> host_input(n * elem_size);
      const_cast<rasterdf::device_buffer&>(in_col.data).copy_to_host(host_input.data(), n * elem_size,
                               _ctx.device(), _ctx.queue(), _ctx.command_pool());
      
      // Scatter on CPU
      std::vector<uint8_t> host_output(output_count * elem_size);
      for (uint32_t i = 0; i < n; i++) {
        if (host_prefix[i + 1] > host_prefix[i]) {
          uint32_t dst_idx = host_prefix[i];
          memcpy(&host_output[dst_idx * elem_size], &host_input[i * elem_size], elem_size);
        }
      }
      
      // Copy output to GPU
      result->columns[c].data.copy_from_host(host_output.data(), output_count * elem_size,
                                            _ctx.device(), _ctx.queue(), _ctx.command_pool());
    }
  } else {
    for (size_t c = 0; c < input.num_columns(); c++) {
      auto& in_col = input.col(c);
      result->columns[c] = allocate_column(_ctx, in_col.type,
                                            static_cast<rasterdf::size_type>(output_count));

      scatter_if_pc spc{};
      spc.input_ptr = in_col.address();
      spc.prefix_ptr = prefix_buf.data();
      spc.output_ptr = result->columns[c].address();
      spc.size = n;
      if (rdf_type_size(in_col.type.id) == 8) {
        disp.dispatch_scatter_if_64(spc, scatter_groups);
      } else {
        disp.dispatch_scatter_if(spc, scatter_groups);
      }
    }
  }

  result->set_num_rows(static_cast<rasterdf::size_type>(output_count));
  RASTERDB_LOG_DEBUG("Filter result: {} rows x {} cols", result->num_rows(), result->num_columns());
  return result;
}

} // namespace gpu
} // namespace rasterdb
