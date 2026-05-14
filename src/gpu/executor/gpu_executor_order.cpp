/*
 * Copyright 2025, RasterDB Contributors.
 * Split from src/gpu/gpu_executor.cpp.
 *
 * GPU VARCHAR sorting via Multi-Pass Prefix Radix Sort (Strategy B):
 *   - Extract 8-byte lexicographic prefix chunks from strings
 *   - Use existing 64-bit GPU radix sort (stable) with row_ids as payload
 *   - LSD-first multi-pass: sort by least-significant chunk first, then
 *     progressively higher-order chunks.  Stability ensures final ordering
 *     is fully lexicographic across all bytes.
 *   - Strings are never moved; only row_ids are permuted.
 */

#include "gpu/gpu_executor_internal.hpp"

namespace rasterdb {
namespace gpu {

// ============================================================================
// ORDER BY — radix sort via rasterdf
// Supports both numeric and STRING sort keys on GPU (no CPU fallback).
// ============================================================================

std::unique_ptr<gpu_table> gpu_executor::execute_order(duckdb::LogicalOrder& op)
{
  RASTERDB_LOG_DEBUG("GPU execute_order");
  D_ASSERT(op.children.size() == 1);
  auto input = execute_operator(*op.children[0]);

  stage_timer t("  order_by");

  const auto N = input->num_rows();
  if (N <= 1) return input;

  auto& disp = _ctx.dispatcher();
  auto* mr   = _ctx.workspace_mr();
  VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  uint32_t n = static_cast<uint32_t>(N);

  // ── Identify sort key columns and directions ──────────────────────
  bool has_string_key = false;
  struct sort_key_info {
    size_t col_idx;
    bool is_string;
    bool descending;
  };
  std::vector<sort_key_info> sort_keys;

  for (auto& order : op.orders) {
    auto& expr = unwrap_cast(*order.expression);
    if (expr.type == duckdb::ExpressionType::BOUND_REF) {
      auto& ref = expr.Cast<duckdb::BoundReferenceExpression>();
      bool is_str = input->col(ref.index).is_string();
      bool desc = (order.type == duckdb::OrderType::DESCENDING);
      sort_keys.push_back({ref.index, is_str, desc});
      if (is_str) has_string_key = true;
    }
  }

  // ── Fast path: no string keys — use existing rasterdf sorted_order ──
  if (!has_string_key) {
    // Also check if any non-key column is a string — gather doesn't support it
    bool has_string_col = false;
    for (size_t c = 0; c < input->num_columns(); c++) {
      if (input->col(c).is_string()) { has_string_col = true; break; }
    }
    if (!has_string_col) {
      std::vector<rasterdf::column_view> key_views;
      std::vector<rasterdf::order> col_order;
      std::vector<gpu_column> expr_temps;
      for (auto& order : op.orders) {
        auto& expr = unwrap_cast(*order.expression);
        if (expr.type == duckdb::ExpressionType::BOUND_REF) {
          auto& ref = expr.Cast<duckdb::BoundReferenceExpression>();
          key_views.push_back(input->col(ref.index).view());
        } else {
          expr_temps.push_back(evaluate_expression(*input, expr));
          key_views.push_back(expr_temps.back().view());
        }
        col_order.push_back(
          (order.type == duckdb::OrderType::DESCENDING)
            ? rasterdf::order::DESCENDING : rasterdf::order::ASCENDING);
      }
      rasterdf::table_view keys_tv(key_views);
      auto indices = rasterdf::sorted_order(keys_tv, col_order,
                         _ctx.vk_context(), disp, mr);
      auto sorted_table = rasterdf::gather(input->view(), indices->view(),
                               _ctx.vk_context(), disp, mr);
      return gpu_table_from_rdf(std::move(sorted_table), input->duckdb_types);
    }
    // Fall through to string-aware path if table has string columns
  }

  // ── String-aware ORDER BY path (Multi-Pass Prefix Radix Sort) ─────

  // 1. Initialize row_ids = [0, 1, 2, ..., N-1]
  rasterdf::device_buffer row_ids_buf(mr, n * sizeof(uint32_t), usage);
  {
    uint32_t ng = div_ceil(n, WG_SIZE);
    radix_init_indices_pc pc{};
    pc.indices_ptr = row_ids_buf.data();
    pc.numElements = n;
    disp.dispatch_radix_init_indices(pc, ng);
  }

  // 2. Process sort keys in reverse priority order (last key first)
  //    so the most significant key is sorted last (LSD-first stable sort).
  for (int ki = static_cast<int>(sort_keys.size()) - 1; ki >= 0; --ki) {
    auto& sk = sort_keys[ki];
    auto& col = input->col(sk.col_idx);

    if (sk.is_string) {
      // ── Multi-pass prefix radix sort for this string column ──────
      // Compute max string length on GPU via atomicMax reduction.
      rasterdf::device_buffer max_len_buf(mr, sizeof(uint32_t), usage);
      // Zero-initialize the output buffer
      uint32_t zero = 0;
      max_len_buf.copy_from_host(&zero, sizeof(uint32_t),
          _ctx.device(), _ctx.queue(), _ctx.command_pool());

      string_max_length_pc mlpc{};
      mlpc.offsets_ptr = col.str_offsets.data();
      mlpc.output_ptr  = max_len_buf.data();
      mlpc.num_rows    = n;
      disp.dispatch_string_max_length(mlpc);

      uint32_t max_len = 0;
      max_len_buf.copy_to_host(&max_len, sizeof(uint32_t),
          _ctx.device(), _ctx.queue(), _ctx.command_pool());
      if (max_len == 0) max_len = 1;

      uint32_t num_passes = (max_len + 7) / 8;
      if (num_passes == 0) num_passes = 1;

      // Allocate radix sort temporary buffers
      uint32_t numGroups = div_ceil(n, WG_SIZE);
      uint32_t numBlocks = div_ceil(numGroups, WG_SIZE);

      rasterdf::device_buffer keys_a(mr, n * sizeof(uint64_t), usage);
      rasterdf::device_buffer keys_b(mr, n * sizeof(uint64_t), usage);
      rasterdf::device_buffer payload_b(mr, n * sizeof(uint32_t), usage);
      rasterdf::device_buffer hist_buf(mr, numGroups * 16 * sizeof(uint32_t), usage);
      rasterdf::device_buffer partial_buf(mr, numBlocks * 16 * sizeof(uint32_t), usage);
      rasterdf::device_buffer bucket_totals(mr, 16 * sizeof(uint32_t), usage);
      rasterdf::device_buffer global_offsets(mr, 16 * sizeof(uint32_t), usage);

      // LSD-first: process from highest byte_offset down to 0
      for (int pass = static_cast<int>(num_passes) - 1; pass >= 0; --pass) {
        uint32_t byte_offset = static_cast<uint32_t>(pass) * 8;

        // Extract 8-byte prefix at byte_offset for each row (via row_ids indirection)
        string_extract_prefix_pc epc{};
        epc.offsets_ptr  = col.str_offsets.data();
        epc.chars_ptr    = col.str_chars.data();
        epc.row_ids_ptr  = row_ids_buf.data();
        epc.output_ptr   = keys_a.data();
        epc.num_rows     = n;
        epc.byte_offset  = byte_offset;
        disp.dispatch_string_extract_prefix(epc);

        // For DESC ordering, flip all bits so ascending radix sort yields descending
        if (sk.descending) {
          radix_init_indices_pc fpc{};
          fpc.indices_ptr = keys_a.data();
          fpc.numElements = n;
          disp.dispatch_flip_bits_64(fpc, numGroups);
        }

        // 64-bit radix sort: sorts keys_a → keys_b → ... (ping-pong),
        // with row_ids as payload. After 16 passes (64 bits), result is in
        // the "source" buffer (even number of swaps).
        disp.dispatch_radix_sort_batched_64(
            keys_a.data(), keys_b.data(),
            row_ids_buf.data(), payload_b.data(),
            hist_buf.data(), partial_buf.data(),
            bucket_totals.data(), global_offsets.data(),
            n, numGroups, numBlocks);
        // After batched_64 (16 passes = even), result keys are in keys_a,
        // and result payload (sorted row_ids) is in row_ids_buf.
        // (16 swaps = even → data ends in original src buffer)
      }
    } else {
      // ── Numeric sort key — extract values, radix sort with row_ids ──
      auto& data_col = col;
      uint32_t numGroups = div_ceil(n, WG_SIZE);
      uint32_t numBlocks = div_ceil(numGroups, WG_SIZE);

      bool is_64bit = (data_col.type.id == rasterdf::type_id::INT64 ||
                       data_col.type.id == rasterdf::type_id::FLOAT64);
      bool is_float = (data_col.type.id == rasterdf::type_id::FLOAT32 ||
                       data_col.type.id == rasterdf::type_id::FLOAT64);

      if (is_64bit) {
        // 64-bit numeric sort
        rasterdf::device_buffer keys_a(mr, n * sizeof(uint64_t), usage);
        rasterdf::device_buffer keys_b(mr, n * sizeof(uint64_t), usage);
        rasterdf::device_buffer payload_b(mr, n * sizeof(uint32_t), usage);
        rasterdf::device_buffer hist_buf(mr, numGroups * 16 * sizeof(uint32_t), usage);
        rasterdf::device_buffer partial_buf(mr, numBlocks * 16 * sizeof(uint32_t), usage);
        rasterdf::device_buffer bucket_totals(mr, 16 * sizeof(uint32_t), usage);
        rasterdf::device_buffer global_offsets(mr, 16 * sizeof(uint32_t), usage);

        // Copy data to keys_a
        disp.copy_buffer(data_col.buffer(), keys_a.buffer(),
                         n * sizeof(uint64_t), 0, 0);

        if (is_float) {
          radix_init_indices_pc fpc{};
          fpc.indices_ptr = keys_a.data();
          fpc.numElements = n;
          disp.dispatch_double_to_sortable(fpc, numGroups);
        }
        if (sk.descending) {
          radix_init_indices_pc fpc{};
          fpc.indices_ptr = keys_a.data();
          fpc.numElements = n;
          disp.dispatch_flip_bits_64(fpc, numGroups);
        }
        disp.dispatch_radix_sort_batched_64(
            keys_a.data(), keys_b.data(),
            row_ids_buf.data(), payload_b.data(),
            hist_buf.data(), partial_buf.data(),
            bucket_totals.data(), global_offsets.data(),
            n, numGroups, numBlocks);
      } else {
        // 32-bit numeric sort
        rasterdf::device_buffer keys_a(mr, n * sizeof(uint32_t), usage);
        rasterdf::device_buffer keys_b(mr, n * sizeof(uint32_t), usage);
        rasterdf::device_buffer payload_b(mr, n * sizeof(uint32_t), usage);
        rasterdf::device_buffer hist_buf(mr, numGroups * 16 * sizeof(uint32_t), usage);
        rasterdf::device_buffer partial_buf(mr, numBlocks * 16 * sizeof(uint32_t), usage);
        rasterdf::device_buffer bucket_totals(mr, 16 * sizeof(uint32_t), usage);
        rasterdf::device_buffer global_offsets(mr, 16 * sizeof(uint32_t), usage);

        // Copy data to keys_a
        disp.copy_buffer(data_col.buffer(), keys_a.buffer(),
                         n * sizeof(uint32_t), 0, 0);

        if (is_float) {
          radix_init_indices_pc fpc{};
          fpc.indices_ptr = keys_a.data();
          fpc.numElements = n;
          disp.dispatch_float_to_sortable(fpc, numGroups);
        }
        if (sk.descending) {
          radix_init_indices_pc fpc{};
          fpc.indices_ptr = keys_a.data();
          fpc.numElements = n;
          disp.dispatch_flip_bits(fpc, numGroups);
        }
        disp.dispatch_radix_sort_batched(
            keys_a.data(), keys_b.data(),
            row_ids_buf.data(), payload_b.data(),
            hist_buf.data(), partial_buf.data(),
            bucket_totals.data(), global_offsets.data(),
            n, numGroups, numBlocks);
      }
    }
  }

  // ── Gather all columns using sorted row_ids ───────────────────────
  auto result = std::make_unique<gpu_table>();
  result->columns.resize(input->num_columns());
  result->duckdb_types = input->duckdb_types;
  result->set_num_rows(N);

  rasterdf::column_view idx_view(
      rasterdf::data_type{rasterdf::type_id::INT32}, N,
      row_ids_buf.data(), 0, 0, 0);

  for (size_t c = 0; c < input->num_columns(); c++) {
    auto& in_col = input->col(c);

    if (in_col.is_string()) {
      // ── Gather string column via lengths + prefix scan + copy ──
      uint32_t nc = n;

      rasterdf::device_buffer out_offsets(mr, (nc + 1) * sizeof(int32_t), usage);

      // Write lengths into out_offsets[0..N-1]
      string_lengths_pc lpc{};
      lpc.offsets_ptr  = in_col.str_offsets.data();
      lpc.indices_ptr  = idx_view.data();
      lpc.num_indices  = nc;
      lpc.output_ptr   = out_offsets.data();
      disp.dispatch_string_lengths(lpc);

      // Zero element N, then exclusive prefix scan on N+1 elements
      disp.fill_buffer(out_offsets.buffer(), 0u, sizeof(int32_t),
                       out_offsets.offset() + nc * sizeof(int32_t));

      uint32_t scan_elems = nc + 1;
      uint32_t scan_ngroups = div_ceil(scan_elems, WG_SIZE);
      rasterdf::device_buffer scan_bsums(mr, scan_ngroups * sizeof(uint32_t), usage);
      rasterdf::device_buffer scan_total(mr, sizeof(uint32_t), usage);
      prefix_scan_pc opc{};
      opc.data_ptr       = out_offsets.data();
      opc.block_sums_ptr = scan_bsums.data();
      opc.total_sum_ptr  = scan_total.data();
      opc.numElements    = scan_elems;
      opc.blockCount     = scan_ngroups;
      disp.dispatch_prefix_scan_local(opc, scan_ngroups);
      disp.dispatch_prefix_scan_global(opc);
      disp.dispatch_prefix_scan_add(opc, scan_ngroups);

      int32_t total_chars = 0;
      out_offsets.copy_to_host(&total_chars, sizeof(int32_t),
          static_cast<size_t>(nc) * sizeof(int32_t),
          _ctx.device(), _ctx.queue(), _ctx.command_pool());

      rasterdf::device_buffer out_chars(mr, std::max(total_chars, 1), usage);

      string_copy_pc cpc{};
      cpc.in_offsets_ptr  = in_col.str_offsets.data();
      cpc.in_chars_ptr    = in_col.str_chars.data();
      cpc.indices_ptr     = idx_view.data();
      cpc.out_offsets_ptr = out_offsets.data();
      cpc.out_chars_ptr   = out_chars.data();
      cpc.num_indices     = nc;
      disp.dispatch_string_copy(cpc);

      gpu_column out;
      out.type = rasterdf::data_type{rasterdf::type_id::STRING};
      out.num_rows = N;
      out.str_offsets = std::move(out_offsets);
      out.str_chars = std::move(out_chars);
      out.str_total_chars = total_chars;
      result->columns[c] = std::move(out);

    } else {
      // ── Gather numeric column via gather_indices ──
      result->columns[c] = allocate_column(_ctx, in_col.type, N);
      uint32_t ng = div_ceil(n, WG_SIZE);
      gather_indices_pc gpc{};
      gpc.input_addr   = in_col.address();
      gpc.indices_addr = row_ids_buf.data();
      gpc.output_addr  = result->columns[c].address();
      gpc.size         = n;

      bool is_64bit = (in_col.type.id == rasterdf::type_id::INT64 ||
                       in_col.type.id == rasterdf::type_id::FLOAT64);
      if (is_64bit)
        disp.dispatch_gather_indices_64(gpc, ng);
      else
        disp.dispatch_gather_indices(gpc, ng);
    }
  }

  RASTERDB_LOG_DEBUG("GPU ORDER BY complete: {} rows, {} columns", N, input->num_columns());
  return result;
}

} // namespace gpu
} // namespace rasterdb
