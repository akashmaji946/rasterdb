/*
 * Copyright 2025, RasterDB Contributors.
 * Split from src/gpu/gpu_executor.cpp.
 */

#include "gpu/gpu_executor_internal.hpp"

#include <rasterdf/gfx_groupby_engine.hpp>

namespace rasterdb {
namespace gpu {

// ============================================================================
// GROUP BY aggregate — hash-based groupby via rasterdf
// Supports 1-, 2-, and 3-column GROUP BY keys.
// Multi-column keys use a composite INT32 key: col0*M+col1 (2-col) or
// (col0*M+col1)*M+col2 (3-col), then decompose after groupby.
// ============================================================================

// Composite key multipliers.
// INT32 path (GPU): fast, but limited to small-range group columns.
static constexpr int32_t GROUPBY_COMPOSITE_M_I32 = 10007;
// INT64 path (CPU): handles any value range, used when INT32 would overflow.
static constexpr int64_t GROUPBY_COMPOSITE_M_I64 = 100000007LL;

// Toggle between compute-shader groupby and mesh-shader gfxm groupby
static constexpr bool USE_SIMPLE_GFX_AGGR = true;

void gpu_executor::execute_grouped_aggregate(
  const gpu_table& input,
  const duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& groups,
  const duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& aggregates,
  const duckdb::vector<duckdb::LogicalType>& result_types,
  gpu_table& output)
{
  RASTERDB_LOG_DEBUG(
    "GPU execute_grouped_aggregate: {} groups, {} aggs", groups.size(), aggregates.size());

  size_t num_group_cols = groups.size();
  if (num_group_cols < 1 || num_group_cols > 3) {
    throw duckdb::NotImplementedException("RasterDB GPU: GROUP BY supports 1-3 columns, got %zu",
                                          num_group_cols);
  }

  // Extract group column indices and validate
  std::vector<duckdb::idx_t> group_col_indices;
  for (size_t g = 0; g < num_group_cols; g++) {
    auto& group_expr = *groups[g];
    if (group_expr.type != duckdb::ExpressionType::BOUND_REF) {
      throw duckdb::NotImplementedException(
        "RasterDB GPU: GROUP BY expression must be a column reference");
    }
    group_col_indices.push_back(group_expr.Cast<duckdb::BoundReferenceExpression>().index);
  }

  RASTERDB_LOG_DEBUG("GROUP BY {} cols, {} rows", num_group_cols, input.num_rows());
  {
    std::ostringstream oss;
    oss << "[RDB_DEBUG] GROUP BY col indices:";
    for (auto idx : group_col_indices)
      oss << " " << idx;
    oss << " (input has " << input.num_columns() << " cols)";
    RASTERDB_LOG_DEBUG("{}", oss.str());
  }

  // Build the effective group key column (single or composite)
  auto n_rows = input.num_rows();
  gpu_column composite_key_storage;  // owns memory for multi-col case
  const gpu_column* key_col_ptr;
  bool composite_is_int64 = false;                 // tracks which decomposition M to use
  int64_t decompose_base1 = 0;                     // base for 2-col, or middle base for 3-col
  int64_t decompose_base2 = 0;                     // last base for 3-col (unused for 2-col)
  std::vector<int64_t> surrogate_id_to_composite;  // INT64 surrogate mapping

  bool single_col_int32 =
    (num_group_cols == 1 && (input.col(group_col_indices[0]).type.id == rasterdf::type_id::INT32 ||
                             input.col(group_col_indices[0]).type.id == rasterdf::type_id::DICTIONARY32));

  if (single_col_int32) {
    key_col_ptr = &input.col(group_col_indices[0]);
  } else if (num_group_cols == 1 && !USE_SIMPLE_GFX_AGGR &&
             input.col(group_col_indices[0]).type.id == rasterdf::type_id::INT64) {
    // Compute path supports single INT64 column directly
    key_col_ptr = &input.col(group_col_indices[0]);
  } else {
    stage_timer tc("    groupby_composite_key");
    auto& disp  = _ctx.dispatcher();
    uint32_t sz = static_cast<uint32_t>(n_rows);

    // Quick max-reduction on each group column to decide INT32 vs INT64.
    // FLOAT32 group columns force the INT64/surrogate path because they
    // can't participate in integer composite key arithmetic.
    bool has_float_group_col       = false;
    int64_t max_composite_estimate = 1;
    std::vector<int64_t> max_vals(num_group_cols, 0);
    {
      for (size_t g = 0; g < num_group_cols; g++) {
        auto& gcol    = input.col(group_col_indices[g]);
        auto col_view = gcol.view();
        rasterdf::reduce_aggregation max_agg(rasterdf::aggregation_kind::MAX);
        int64_t mv = 0;
        if (gcol.type.id == rasterdf::type_id::FLOAT32) {
          has_float_group_col = true;
          auto max_s          = rasterdf::reduce(col_view,
                                                 max_agg,
                                                 rasterdf::data_type{rasterdf::type_id::FLOAT32},
                                                 _ctx.vk_context(),
                                                 _ctx.dispatcher(),
                                                 _ctx.workspace_mr());
          // Bit-cast float max to int32 for range estimation
          float fv = max_s->as<float>();
          int32_t iv;
          std::memcpy(&iv, &fv, sizeof(int32_t));
          mv = static_cast<int64_t>(iv < 0 ? -iv : iv);
        } else {
          auto max_s = rasterdf::reduce(col_view,
                                        max_agg,
                                        rasterdf::data_type{rasterdf::type_id::INT32},
                                        _ctx.vk_context(),
                                        _ctx.dispatcher(),
                                        _ctx.workspace_mr());
          mv         = static_cast<int64_t>(max_s->as<int32_t>());
        }
        max_vals[g] = mv;
        if (g == 0) {
          max_composite_estimate = mv;
        } else {
          max_composite_estimate = max_composite_estimate * GROUPBY_COMPOSITE_M_I32 + mv;
        }
      }
    }
    // Force INT64/surrogate path when FLOAT columns are in GROUP BY
    composite_is_int64 = has_float_group_col || (max_composite_estimate > INT32_MAX / 2);
    {
      std::ostringstream oss;
      oss << "[RDB_DEBUG] GROUP BY max_vals:";
      for (size_t g = 0; g < num_group_cols; g++)
        oss << " col[" << group_col_indices[g] << "]=" << max_vals[g];
      oss << " composite_estimate=" << max_composite_estimate << " is_int64=" << composite_is_int64;
      RASTERDB_LOG_DEBUG("{}", oss.str());
    }

    if (!composite_is_int64) {
      // ---- INT32 GPU path (fast) ----
      decompose_base1 = GROUPBY_COMPOSITE_M_I32;
      decompose_base2 = GROUPBY_COMPOSITE_M_I32;

      // Step 1: temp = col0 * M  (COL_SCALAR multiply)
      auto temp = allocate_column(_ctx, {rasterdf::type_id::INT32}, n_rows);
      {
        binary_op_push_constants pc{};
        pc.input_a     = input.col(group_col_indices[0]).address();
        pc.input_b     = 0;
        pc.output_addr = temp.address();
        pc.size        = sz;
        pc.op          = 2;  // MUL
        pc.scalar_val  = GROUPBY_COMPOSITE_M_I32;
        pc.mode        = 1;                                                    // COL_SCALAR
        pc.type_id     = static_cast<int32_t>(rasterdf::ShaderTypeId::INT32);  // INT32
        pc.debug_mode  = 0;
        disp.dispatch_binary_op(pc);
      }

      // Step 2: composite = temp + col1  (COL_COL add)
      composite_key_storage = allocate_column(_ctx, {rasterdf::type_id::INT32}, n_rows);
      {
        binary_op_push_constants pc{};
        pc.input_a     = temp.address();
        pc.input_b     = input.col(group_col_indices[1]).address();
        pc.output_addr = composite_key_storage.address();
        pc.size        = sz;
        pc.op          = 0;                                                    // ADD
        pc.mode        = 0;                                                    // COL_COL
        pc.type_id     = static_cast<int32_t>(rasterdf::ShaderTypeId::INT32);  // INT32
        pc.debug_mode  = 0;
        disp.dispatch_binary_op(pc);
      }

      if (num_group_cols == 3) {
        // Step 3: temp2 = composite * M
        auto temp2 = allocate_column(_ctx, {rasterdf::type_id::INT32}, n_rows);
        {
          binary_op_push_constants pc{};
          pc.input_a     = composite_key_storage.address();
          pc.input_b     = 0;
          pc.output_addr = temp2.address();
          pc.size        = sz;
          pc.op          = 2;  // MUL
          pc.scalar_val  = GROUPBY_COMPOSITE_M_I32;
          pc.mode        = 1;                                                    // COL_SCALAR
          pc.type_id     = static_cast<int32_t>(rasterdf::ShaderTypeId::INT32);  // INT32
          pc.debug_mode  = 0;
          disp.dispatch_binary_op(pc);
        }
        // Step 4: composite = temp2 + col2
        {
          binary_op_push_constants pc{};
          pc.input_a     = temp2.address();
          pc.input_b     = input.col(group_col_indices[2]).address();
          pc.output_addr = composite_key_storage.address();
          pc.size        = sz;
          pc.op          = 0;                                                    // ADD
          pc.mode        = 0;                                                    // COL_COL
          pc.type_id     = static_cast<int32_t>(rasterdf::ShaderTypeId::INT32);  // INT32
          pc.debug_mode  = 0;
          disp.dispatch_binary_op(pc);
        }
      }
      key_col_ptr = &composite_key_storage;

      if (debug_logging_enabled()) {
        auto sample = std::min<rasterdf::size_type>(10, n_rows);
        std::vector<int32_t> h_composite_debug(sample);
        if (sample > 0) {
          composite_key_storage.data.copy_to_host(h_composite_debug.data(),
                                                  sample * sizeof(int32_t),
                                                  _ctx.device(),
                                                  _ctx.queue(),
                                                  _ctx.command_pool());
        }
        std::ostringstream line;
        line << "[RDB_DEBUG] GPU composite keys (first " << sample << "):";
        for (auto key : h_composite_debug) {
          line << ' ' << key;
        }
        RASTERDB_LOG_DEBUG("{}", line.str());
      }
    } else {
      // ---- INT64 composite key (handles both INT32 and FLOAT32 group columns) ----
      // For hash-based groupby, we only need equality (not ordering).
      // FLOAT32 raw bit patterns are injective: two floats are equal iff their
      // bits are equal. So we treat FLOAT32 as uint32 and use base = 2^32.
      // INT32 columns use base = max(col) + 1.
      //
      // 2-col: key = col0_u * base1 + col1_u
      // 3-col: key = (col0_u * base1 + col1_u) * base2 + col2_u
      //
      // where col_u = uint32(raw_bits) for FLOAT32, or raw value for INT32.

      // Compute bases: FLOAT32 columns need full 32-bit range, INT32 use max+1
      std::vector<bool> col_is_float(num_group_cols);
      for (size_t g = 0; g < num_group_cols; g++) {
        col_is_float[g] = (input.col(group_col_indices[g]).type.id == rasterdf::type_id::FLOAT32);
      }

      // For FLOAT32 cols: base = 2^32 (covers all uint32 bit patterns)
      // For INT32 cols: base = max_val + 1
      decompose_base1 = 1;
      decompose_base2 = 1;
      if (num_group_cols >= 2) {
        decompose_base1 = col_is_float[1] ? (1LL << 32) : (max_vals[1] + 1);
      }
      if (num_group_cols >= 3) {
        decompose_base2 = col_is_float[2] ? (1LL << 32) : (max_vals[2] + 1);
      }
      if (decompose_base1 <= 0 || decompose_base2 <= 0) {
        throw duckdb::NotImplementedException(
          "RasterDB GPU: non-positive GROUP BY base in INT64 composite path");
      }

      // Download filtered group columns (post-filter, typically small ~1M rows)
      std::vector<std::vector<int32_t>> h_group_cols(num_group_cols);
      for (size_t g = 0; g < num_group_cols; g++) {
        h_group_cols[g].resize(n_rows);
        const_cast<rasterdf::device_buffer&>(input.col(group_col_indices[g]).data)
          .copy_to_host(h_group_cols[g].data(),
                        n_rows * sizeof(int32_t),
                        _ctx.device(),
                        _ctx.queue(),
                        _ctx.command_pool());
      }

      // Helper: convert raw int32 bits to uint64 for composite key construction.
      // For FLOAT32 columns, reinterpret as uint32 to get a non-negative value.
      // For INT32 columns, use the raw value directly (must be non-negative for groupby keys).
      auto to_unsigned = [&](size_t col_idx, int32_t raw) -> int64_t {
        if (col_is_float[col_idx]) { return static_cast<int64_t>(static_cast<uint32_t>(raw)); }
        return static_cast<int64_t>(raw);
      };

      // Compute INT64 composite keys (simple vectorizable loop, no hash map)
      std::vector<int64_t> h_composite(n_rows);
      if (num_group_cols == 1) {
        for (rasterdf::size_type r = 0; r < n_rows; r++) {
          h_composite[r] = to_unsigned(0, h_group_cols[0][r]);
        }
      } else if (num_group_cols == 2) {
        for (rasterdf::size_type r = 0; r < n_rows; r++) {
          h_composite[r] = to_unsigned(0, h_group_cols[0][r]) * decompose_base1 +
                           to_unsigned(1, h_group_cols[1][r]);
        }
      } else {
        for (rasterdf::size_type r = 0; r < n_rows; r++) {
          h_composite[r] = (to_unsigned(0, h_group_cols[0][r]) * decompose_base1 +
                            to_unsigned(1, h_group_cols[1][r])) *
                             decompose_base2 +
                           to_unsigned(2, h_group_cols[2][r]);
        }
      }

      RASTERDB_LOG_DEBUG("[RDB_DEBUG] INT64 composite: {} rows, base1={}, base2={}, has_float={}",
                         n_rows,
                         decompose_base1,
                         decompose_base2,
                         has_float_group_col);

      if constexpr (USE_SIMPLE_GFX_AGGR) {
        // GFXM path: INT64 keys are slow/broken in mesh shaders.
        // Use a CPU-side surrogate mapping: INT64 composite -> INT32 ID.
        std::unordered_map<int64_t, int32_t> composite_to_id;
        std::vector<int32_t> h_surrogates(n_rows);
        for (rasterdf::size_type r = 0; r < n_rows; r++) {
          int64_t key = h_composite[r];
          auto it     = composite_to_id.find(key);
          if (it == composite_to_id.end()) {
            int32_t next_id      = static_cast<int32_t>(surrogate_id_to_composite.size());
            composite_to_id[key] = next_id;
            surrogate_id_to_composite.push_back(key);
            h_surrogates[r] = next_id;
          } else {
            h_surrogates[r] = it->second;
          }
        }

        RASTERDB_LOG_DEBUG("[RDB_DEBUG] GFXM surrogate mapping: {} unique keys",
                           surrogate_id_to_composite.size());

        // Upload INT32 surrogate IDs to GPU
        composite_key_storage = allocate_column(_ctx, {rasterdf::type_id::INT32}, n_rows);
        composite_key_storage.data.copy_from_host(h_surrogates.data(),
                                                  n_rows * sizeof(int32_t),
                                                  _ctx.device(),
                                                  _ctx.queue(),
                                                  _ctx.command_pool());
      } else {
        // Compute path: upload INT64 composite key as-is
        composite_key_storage = allocate_column(_ctx, {rasterdf::type_id::INT64}, n_rows);
        composite_key_storage.data.copy_from_host(h_composite.data(),
                                                  n_rows * sizeof(int64_t),
                                                  _ctx.device(),
                                                  _ctx.queue(),
                                                  _ctx.command_pool());
      }
      key_col_ptr = &composite_key_storage;

      // Store column types for decomposition (needed to reverse uint32 cast for FLOAT32)
      _group_col_types.resize(num_group_cols);
      for (size_t g = 0; g < num_group_cols; g++) {
        _group_col_types[g] = input.col(group_col_indices[g]).type.id;
      }
    }
  }

  // Build table_view for the effective group key
  auto group_key_view                          = key_col_ptr->view();
  std::vector<rasterdf::column_view> key_views = {group_key_view};
  rasterdf::table_view keys_tv(key_views);

  // Process each aggregate expression using GFXM mesh shader or compute shader
  // Result layout: [group_key_cols..., agg_cols...]

  bool keys_set                         = false;
  rasterdf::size_type num_groups_result = 0;

  if constexpr (USE_SIMPLE_GFX_AGGR) {
    // ── GFXM Mesh Shader Groupby (graphics-pipeline, mesh shader hash aggregation) ──
    // Initialize gfx engine
    rasterdf::gfx_groupby_engine_init(_ctx.vk_context());
    RASTERDB_LOG_DEBUG("     [GFXM] Engine initialized");

    for (duckdb::idx_t i = 0; i < aggregates.size(); i++) {
      auto& expr  = aggregates[i]->Cast<duckdb::BoundAggregateExpression>();
      auto& fname = expr.function.name;

      bool is_count_star = false;

      if (expr.children.empty()) {
        is_count_star = (fname == "count" || fname == "count_star");
        if (!is_count_star) {
          throw duckdb::NotImplementedException(
            "RasterDB GPU: grouped aggregate '%s' with no children", fname.c_str());
        }
      }

      // Map aggregate name to gfxm agg_type (0=sum, 1=count, 2=min, 3=max, 4=mean)
      int gfxm_agg_type;
      if (fname == "sum" || fname == "sum_no_overflow") {
        gfxm_agg_type = 0;
      } else if (fname == "min") {
        gfxm_agg_type = 2;
      } else if (fname == "max") {
        gfxm_agg_type = 3;
      } else if (fname == "count" || fname == "count_star") {
        gfxm_agg_type = 1;
      } else if (fname == "avg" || fname == "mean") {
        gfxm_agg_type = 4;
      } else {
        throw duckdb::NotImplementedException("RasterDB GPU: unsupported grouped aggregate '%s'",
                                              fname.c_str());
      }

      RASTERDB_LOG_DEBUG("     [GFXM] Aggregate {}/{}: {} (type={}, count_star={})",
                         i + 1,
                         aggregates.size(),
                         fname,
                         gfxm_agg_type,
                         is_count_star);

      // Get key and value device addresses
      VkDeviceAddress keys_addr = key_col_ptr->address();

      gpu_column val_temp;
      VkDeviceAddress values_addr     = 0;
      rasterdf::type_id value_type_id = rasterdf::type_id::INT32;
      if (!is_count_star) {
        val_temp      = evaluate_expression(input, *expr.children[0]);
        values_addr   = val_temp.address();
        value_type_id = val_temp.type.id;
        RASTERDB_LOG_DEBUG("     [GFXM] Value column evaluated, addr=0x{:x}",
                           static_cast<uint64_t>(values_addr));
      }

      uint32_t n = static_cast<uint32_t>(input.num_rows());
      RASTERDB_LOG_DEBUG("     [GFXM] Input rows: {}", n);

      // Call gfxm groupby - use INT32 mesh shaders with surrogate INT32 keys
      rasterdf::device_buffer out_keys(
        _ctx.workspace_mr(),
        0,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
      rasterdf::device_buffer out_values(
        _ctx.workspace_mr(),
        0,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
      uint32_t out_num_groups = 0;

      auto t_gfxm_start = std::chrono::high_resolution_clock::now();

      // Use INT32 mesh shaders with surrogate INT32 keys (INT64 atomics are broken)
      rasterdf::gfxm_groupby_aggregate(gfxm_agg_type,
                                       keys_addr,
                                       values_addr,
                                       n,
                                       _ctx.dispatcher(),
                                       _ctx.workspace_mr(),
                                       out_keys,
                                       out_values,
                                       out_num_groups,
                                       0,
                                       value_type_id);

      auto t_gfxm_end = std::chrono::high_resolution_clock::now();
      RASTERDB_LOG_DEBUG(
        "     [GFXM] gfxm_groupby_aggregate time: {:.2f} ms",
        std::chrono::duration<double, std::milli>(t_gfxm_end - t_gfxm_start).count());

      RASTERDB_LOG_DEBUG("[GFXM_DBG] agg={} out_num_groups={}", gfxm_agg_type, out_num_groups);

      if (out_num_groups == 0) {
        throw duckdb::NotImplementedException("RasterDB GPU: gfxm groupby produced zero groups");
      }

      // Debug: dump raw GFXM output keys and values
      if (debug_logging_enabled()) {
        std::vector<int32_t> dbg_keys(out_num_groups);
        out_keys.copy_to_host(dbg_keys.data(),
                              out_num_groups * sizeof(int32_t),
                              0,
                              _ctx.device(),
                              _ctx.queue(),
                              _ctx.command_pool());
        std::ostringstream keys_line;
        keys_line << "[GFXM_DBG] raw out_keys:";
        for (uint32_t j = 0; j < out_num_groups && j < 10; j++) {
          keys_line << ' ' << dbg_keys[j];
        }
        RASTERDB_LOG_DEBUG("{}", keys_line.str());

        if (gfxm_agg_type == 0) {  // sum -> INT64 values
          std::vector<int64_t> dbg_vals(out_num_groups);
          out_values.copy_to_host(dbg_vals.data(),
                                  out_num_groups * sizeof(int64_t),
                                  0,
                                  _ctx.device(),
                                  _ctx.queue(),
                                  _ctx.command_pool());
          std::ostringstream values_line;
          values_line << "[GFXM_DBG] raw out_values(i64):";
          for (uint32_t j = 0; j < out_num_groups && j < 10; j++) {
            values_line << ' ' << dbg_vals[j];
          }
          RASTERDB_LOG_DEBUG("{}", values_line.str());
        } else {
          std::vector<int32_t> dbg_vals(out_num_groups);
          out_values.copy_to_host(dbg_vals.data(),
                                  out_num_groups * sizeof(int32_t),
                                  0,
                                  _ctx.device(),
                                  _ctx.queue(),
                                  _ctx.command_pool());
          std::ostringstream values_line;
          values_line << "[GFXM_DBG] raw out_values(i32):";
          for (uint32_t j = 0; j < out_num_groups && j < 10; j++) {
            values_line << ' ' << dbg_vals[j];
          }
          RASTERDB_LOG_DEBUG("{}", values_line.str());
        }
      }

      RASTERDB_LOG_DEBUG("     [GFXM] Output groups: {}", out_num_groups);

      // Convert device buffers to rasterdf columns
      // Keys are INT32 surrogate IDs or original INT32s
      rasterdf::data_type key_type = key_col_ptr->type;
      rasterdf::column key_col_rdf(key_type, out_num_groups, std::move(out_keys));

      rasterdf::data_type val_type;
      if (gfxm_agg_type == 0) {
        val_type = rasterdf::data_type{value_type_id == rasterdf::type_id::FLOAT32
                                         ? rasterdf::type_id::FLOAT64
                                         : rasterdf::type_id::INT64};
      } else if (gfxm_agg_type == 4) {
        val_type = rasterdf::data_type{rasterdf::type_id::FLOAT32};
      } else {
        val_type = rasterdf::data_type{rasterdf::type_id::INT32};
      }
      rasterdf::column val_col_rdf(val_type, out_num_groups, std::move(out_values));

      // Download keys for sorting
      size_t key_elem_size = rasterdf::size_of(key_col_rdf.type());
      bool keys_are_int64  = (key_col_rdf.type().id == rasterdf::type_id::INT64);
      RASTERDB_LOG_DEBUG("     [GFXM] Key type: {}, elem_size={}",
                         keys_are_int64 ? "INT64" : "INT32",
                         key_elem_size);

      auto t_download_keys_start = std::chrono::high_resolution_clock::now();
      std::vector<uint8_t> h_keys_raw(out_num_groups * key_elem_size);
      key_col_rdf.device_data().copy_to_host(h_keys_raw.data(),
                                             out_num_groups * key_elem_size,
                                             0,
                                             _ctx.device(),
                                             _ctx.queue(),
                                             _ctx.command_pool());
      auto t_download_keys_end = std::chrono::high_resolution_clock::now();
      RASTERDB_LOG_DEBUG(
        "     [GFXM] Download keys time: {:.2f} ms",
        std::chrono::duration<double, std::milli>(t_download_keys_end - t_download_keys_start)
          .count());

      // Build a sort permutation (ascending by key)
      auto t_sort_start = std::chrono::high_resolution_clock::now();
      std::vector<size_t> perm(out_num_groups);
      std::iota(perm.begin(), perm.end(), 0);
      if (keys_are_int64) {
        auto* kp = reinterpret_cast<const int64_t*>(h_keys_raw.data());
        std::sort(perm.begin(), perm.end(), [kp](size_t a, size_t b) { return kp[a] < kp[b]; });
      } else {
        auto* kp = reinterpret_cast<const int32_t*>(h_keys_raw.data());
        std::sort(perm.begin(), perm.end(), [kp](size_t a, size_t b) { return kp[a] < kp[b]; });
      }
      auto t_sort_end = std::chrono::high_resolution_clock::now();
      RASTERDB_LOG_DEBUG(
        "     [GFXM] Sort permutation time: {:.2f} ms",
        std::chrono::duration<double, std::milli>(t_sort_end - t_sort_start).count());

      // Download value column to CPU, apply permutation, re-upload
      size_t val_elem_size = rasterdf::size_of(val_col_rdf.type());
      RASTERDB_LOG_DEBUG("     [GFXM] Value elem_size={}", val_elem_size);

      auto t_download_vals_start = std::chrono::high_resolution_clock::now();
      std::vector<uint8_t> h_vals(out_num_groups * val_elem_size);
      val_col_rdf.device_data().copy_to_host(h_vals.data(),
                                             out_num_groups * val_elem_size,
                                             0,
                                             _ctx.device(),
                                             _ctx.queue(),
                                             _ctx.command_pool());
      auto t_download_vals_end = std::chrono::high_resolution_clock::now();
      RASTERDB_LOG_DEBUG(
        "     [GFXM] Download values time: {:.2f} ms",
        std::chrono::duration<double, std::milli>(t_download_vals_end - t_download_vals_start)
          .count());

      // Apply permutation to keys and values
      auto t_permute_start = std::chrono::high_resolution_clock::now();
      std::vector<uint8_t> sorted_keys_raw(out_num_groups * key_elem_size);
      std::vector<uint8_t> sorted_vals(out_num_groups * val_elem_size);
      for (size_t j = 0; j < out_num_groups; j++) {
        std::memcpy(sorted_keys_raw.data() + j * key_elem_size,
                    h_keys_raw.data() + perm[j] * key_elem_size,
                    key_elem_size);
        std::memcpy(sorted_vals.data() + j * val_elem_size,
                    h_vals.data() + perm[j] * val_elem_size,
                    val_elem_size);
      }
      auto t_permute_end = std::chrono::high_resolution_clock::now();
      RASTERDB_LOG_DEBUG(
        "     [GFXM] Apply permutation time: {:.2f} ms",
        std::chrono::duration<double, std::milli>(t_permute_end - t_permute_start).count());

      // On first aggregate, store the sorted keys
      if (!keys_set) {
        num_groups_result = out_num_groups;
        if (num_group_cols == 1 && surrogate_id_to_composite.empty()) {
          auto t_upload_keys_start = std::chrono::high_resolution_clock::now();
          auto sorted_key_col      = allocate_column(_ctx, key_col_ptr->type, out_num_groups);
          sorted_key_col.data.copy_from_host(sorted_keys_raw.data(),
                                             out_num_groups * key_elem_size,
                                             _ctx.device(),
                                             _ctx.queue(),
                                             _ctx.command_pool());
          auto t_upload_keys_end = std::chrono::high_resolution_clock::now();
          RASTERDB_LOG_DEBUG(
            "     [GFXM] Upload sorted keys time: {:.2f} ms",
            std::chrono::duration<double, std::milli>(t_upload_keys_end - t_upload_keys_start)
              .count());
          output.columns[0] = std::move(sorted_key_col);
        } else {
          // Mixed-radix decomposition or Surrogate mapping back
          RASTERDB_LOG_DEBUG("     [GFXM] Mixed-radix decomposition / Surrogate map-back");
          std::vector<int64_t> sorted_composite_i64(out_num_groups);
          if (keys_are_int64) {
            auto* p = reinterpret_cast<const int64_t*>(sorted_keys_raw.data());
            for (uint32_t j = 0; j < out_num_groups; j++)
              sorted_composite_i64[j] = p[j];
          } else if (!surrogate_id_to_composite.empty()) {
            auto* p = reinterpret_cast<const int32_t*>(sorted_keys_raw.data());
            for (uint32_t j = 0; j < out_num_groups; j++) {
              int32_t id = p[j];
              if (id < 0 || static_cast<size_t>(id) >= surrogate_id_to_composite.size()) {
                throw duckdb::NotImplementedException(
                  "RasterDB GPU: invalid surrogate GROUP BY key id %d", id);
              }
              sorted_composite_i64[j] = surrogate_id_to_composite[static_cast<size_t>(id)];
            }
          } else {
            auto* p = reinterpret_cast<const int32_t*>(sorted_keys_raw.data());
            for (uint32_t j = 0; j < out_num_groups; j++)
              sorted_composite_i64[j] = static_cast<int64_t>(p[j]);
          }

          RASTERDB_LOG_DEBUG(
            "[GFXM_DBG] decompose: num_group_cols={} base1={} base2={} surr_empty={} keys_i64={}",
            num_group_cols,
            decompose_base1,
            decompose_base2,
            surrogate_id_to_composite.empty(),
            keys_are_int64);

          // Decompose and assign correct type (FLOAT32 cols get their original type)
          auto get_col_type = [&](size_t g) -> rasterdf::type_id {
            if (g < _group_col_types.size()) return _group_col_types[g];
            return rasterdf::type_id::INT32;
          };

          if (num_group_cols == 1) {
            std::vector<int32_t> col0(out_num_groups);
            for (uint32_t j = 0; j < out_num_groups; j++) {
              col0[j] = static_cast<int32_t>(sorted_composite_i64[j]);
            }
            output.columns[0] = allocate_column(_ctx, {get_col_type(0)}, out_num_groups);
            output.columns[0].data.copy_from_host(col0.data(),
                                                  out_num_groups * sizeof(int32_t),
                                                  _ctx.device(),
                                                  _ctx.queue(),
                                                  _ctx.command_pool());
          } else if (num_group_cols == 2) {
            std::vector<int32_t> col0(out_num_groups), col1(out_num_groups);
            for (uint32_t j = 0; j < out_num_groups; j++) {
              col1[j] = static_cast<int32_t>(sorted_composite_i64[j] % decompose_base1);
              col0[j] = static_cast<int32_t>(sorted_composite_i64[j] / decompose_base1);
            }
            output.columns[0] = allocate_column(_ctx, {get_col_type(0)}, out_num_groups);
            output.columns[0].data.copy_from_host(col0.data(),
                                                  out_num_groups * sizeof(int32_t),
                                                  _ctx.device(),
                                                  _ctx.queue(),
                                                  _ctx.command_pool());
            output.columns[1] = allocate_column(_ctx, {get_col_type(1)}, out_num_groups);
            output.columns[1].data.copy_from_host(col1.data(),
                                                  out_num_groups * sizeof(int32_t),
                                                  _ctx.device(),
                                                  _ctx.queue(),
                                                  _ctx.command_pool());
          } else {
            std::vector<int32_t> col0(out_num_groups), col1(out_num_groups), col2(out_num_groups);
            for (uint32_t j = 0; j < out_num_groups; j++) {
              int64_t c = sorted_composite_i64[j];
              col2[j]   = static_cast<int32_t>(c % decompose_base2);
              c /= decompose_base2;
              col1[j] = static_cast<int32_t>(c % decompose_base1);
              col0[j] = static_cast<int32_t>(c / decompose_base1);
            }
            output.columns[0] = allocate_column(_ctx, {get_col_type(0)}, out_num_groups);
            output.columns[0].data.copy_from_host(col0.data(),
                                                  out_num_groups * sizeof(int32_t),
                                                  _ctx.device(),
                                                  _ctx.queue(),
                                                  _ctx.command_pool());
            output.columns[1] = allocate_column(_ctx, {get_col_type(1)}, out_num_groups);
            output.columns[1].data.copy_from_host(col1.data(),
                                                  out_num_groups * sizeof(int32_t),
                                                  _ctx.device(),
                                                  _ctx.queue(),
                                                  _ctx.command_pool());
            output.columns[2] = allocate_column(_ctx, {get_col_type(2)}, out_num_groups);
            output.columns[2].data.copy_from_host(col2.data(),
                                                  out_num_groups * sizeof(int32_t),
                                                  _ctx.device(),
                                                  _ctx.queue(),
                                                  _ctx.command_pool());
          }
        }
        keys_set = true;
      }

      // Create gpu_column for sorted values and upload
      size_t out_col_idx       = num_group_cols + i;
      auto t_upload_vals_start = std::chrono::high_resolution_clock::now();
      auto sorted_val_col      = allocate_column(_ctx, val_col_rdf.type(), out_num_groups);
      sorted_val_col.data.copy_from_host(sorted_vals.data(),
                                         out_num_groups * val_elem_size,
                                         _ctx.device(),
                                         _ctx.queue(),
                                         _ctx.command_pool());
      auto t_upload_vals_end = std::chrono::high_resolution_clock::now();
      RASTERDB_LOG_DEBUG(
        "     [GFXM] Upload sorted values time: {:.2f} ms",
        std::chrono::duration<double, std::milli>(t_upload_vals_end - t_upload_vals_start).count());
      output.columns[out_col_idx] = std::move(sorted_val_col);
    }

  }  // end if constexpr (USE_SIMPLE_GFX_AGGR)
  else {
    // ── Compute Shader Groupby (rasterdf::groupby) ──
    RASTERDB_LOG_DEBUG("     [COMPUTE] Using compute shader groupby");

    for (duckdb::idx_t i = 0; i < aggregates.size(); i++) {
      auto& expr  = aggregates[i]->Cast<duckdb::BoundAggregateExpression>();
      auto& fname = expr.function.name;

      bool is_count_star = false;

      if (expr.children.empty()) {
        is_count_star = (fname == "count" || fname == "count_star");
        if (!is_count_star) {
          throw duckdb::NotImplementedException(
            "RasterDB GPU: grouped aggregate '%s' with no children", fname.c_str());
        }
      }

      // Map aggregate name to rasterdf kind
      rasterdf::aggregation_kind kind;
      if (fname == "sum" || fname == "sum_no_overflow") {
        kind = rasterdf::aggregation_kind::SUM;
      } else if (fname == "min") {
        kind = rasterdf::aggregation_kind::MIN;
      } else if (fname == "max") {
        kind = rasterdf::aggregation_kind::MAX;
      } else if (fname == "count" || fname == "count_star") {
        kind = rasterdf::aggregation_kind::COUNT_ALL;
      } else if (fname == "avg" || fname == "mean") {
        kind = rasterdf::aggregation_kind::MEAN;
      } else {
        throw duckdb::NotImplementedException("RasterDB GPU: unsupported grouped aggregate '%s'",
                                              fname.c_str());
      }

      // Build aggregation request
      rasterdf::groupby gb(keys_tv, _ctx.vk_context(), _ctx.dispatcher(), _ctx.workspace_mr());

      std::vector<rasterdf::aggregation_request> requests;
      rasterdf::aggregation_request req;

      // Evaluate the value expression (may be column ref or complex expression)
      gpu_column val_temp;  // keep alive for view validity
      if (is_count_star) {
        req.values = key_col_ptr->view();
      } else {
        val_temp   = evaluate_expression(input, *expr.children[0]);
        req.values = val_temp.view();
      }
      req.aggregations.push_back(std::make_unique<rasterdf::groupby_aggregation>(kind));
      requests.push_back(std::move(req));

      auto agg_result = gb.aggregate(std::move(requests));

      // Extract keys + value from this aggregate call
      auto this_key_cols = agg_result.keys->extract();
      if (this_key_cols.empty() || agg_result.results.empty() || !agg_result.results[0]) {
        throw duckdb::NotImplementedException(
          "RasterDB GPU: grouped aggregate '%s' produced empty keys/results", fname.c_str());
      }
      auto ng = this_key_cols[0]->size();

      // Download keys to CPU for sorting (INT32 or INT64 depending on key type)
      auto& key_col_rdf    = *this_key_cols[0];
      auto& val_col_rdf    = *agg_result.results[0];
      size_t key_elem_size = rasterdf::size_of(key_col_rdf.type());
      bool keys_are_int64  = (key_col_rdf.type().id == rasterdf::type_id::INT64);

      std::vector<uint8_t> h_keys_raw(ng * key_elem_size);
      key_col_rdf.device_data().copy_to_host(
        h_keys_raw.data(), ng * key_elem_size, 0, _ctx.device(), _ctx.queue(), _ctx.command_pool());

      // Build a sort permutation (ascending by key)
      std::vector<size_t> perm(ng);
      std::iota(perm.begin(), perm.end(), 0);
      if (keys_are_int64) {
        auto* kp = reinterpret_cast<const int64_t*>(h_keys_raw.data());
        std::sort(perm.begin(), perm.end(), [kp](size_t a, size_t b) { return kp[a] < kp[b]; });
      } else {
        auto* kp = reinterpret_cast<const int32_t*>(h_keys_raw.data());
        std::sort(perm.begin(), perm.end(), [kp](size_t a, size_t b) { return kp[a] < kp[b]; });
      }

      // Download value column to CPU, apply permutation, re-upload
      size_t val_elem_size = rasterdf::size_of(val_col_rdf.type());
      std::vector<uint8_t> h_vals(ng * val_elem_size);
      val_col_rdf.device_data().copy_to_host(
        h_vals.data(), ng * val_elem_size, 0, _ctx.device(), _ctx.queue(), _ctx.command_pool());

      // Apply permutation to keys and values
      std::vector<uint8_t> sorted_keys_raw(ng * key_elem_size);
      std::vector<uint8_t> sorted_vals(ng * val_elem_size);
      for (size_t j = 0; j < ng; j++) {
        std::memcpy(sorted_keys_raw.data() + j * key_elem_size,
                    h_keys_raw.data() + perm[j] * key_elem_size,
                    key_elem_size);
        std::memcpy(sorted_vals.data() + j * val_elem_size,
                    h_vals.data() + perm[j] * val_elem_size,
                    val_elem_size);
      }

      // On first aggregate, store the sorted keys
      if (!keys_set) {
        num_groups_result = ng;
        if (num_group_cols == 1) {
          auto sorted_key_col = allocate_column(_ctx, key_col_rdf.type(), ng);
          sorted_key_col.data.copy_from_host(sorted_keys_raw.data(),
                                             ng * key_elem_size,
                                             _ctx.device(),
                                             _ctx.queue(),
                                             _ctx.command_pool());
          output.columns[0] = std::move(sorted_key_col);
        } else {
          // Mixed-radix decomposition (INT64 composite keys, handles INT32 + FLOAT32)
          std::vector<int64_t> sorted_composite_i64(ng);
          if (keys_are_int64) {
            auto* p = reinterpret_cast<const int64_t*>(sorted_keys_raw.data());
            for (rasterdf::size_type j = 0; j < ng; j++)
              sorted_composite_i64[j] = p[j];
          } else if (!surrogate_id_to_composite.empty()) {
            auto* p = reinterpret_cast<const int32_t*>(sorted_keys_raw.data());
            for (rasterdf::size_type j = 0; j < ng; j++) {
              int32_t id = p[j];
              if (id < 0 || static_cast<size_t>(id) >= surrogate_id_to_composite.size()) {
                throw duckdb::NotImplementedException(
                  "RasterDB GPU: invalid surrogate GROUP BY key id %d", id);
              }
              sorted_composite_i64[j] = surrogate_id_to_composite[static_cast<size_t>(id)];
            }
          } else {
            auto* p = reinterpret_cast<const int32_t*>(sorted_keys_raw.data());
            for (rasterdf::size_type j = 0; j < ng; j++)
              sorted_composite_i64[j] = static_cast<int64_t>(p[j]);
          }

          // Decompose and assign correct type (FLOAT32 cols get their original type)
          auto get_col_type = [&](size_t g) -> rasterdf::type_id {
            if (g < _group_col_types.size()) return _group_col_types[g];
            return rasterdf::type_id::INT32;
          };

          if (num_group_cols == 2) {
            std::vector<int32_t> col0(ng), col1(ng);
            for (rasterdf::size_type j = 0; j < ng; j++) {
              col1[j] = static_cast<int32_t>(sorted_composite_i64[j] % decompose_base1);
              col0[j] = static_cast<int32_t>(sorted_composite_i64[j] / decompose_base1);
            }
            output.columns[0] = allocate_column(_ctx, {get_col_type(0)}, ng);
            output.columns[0].data.copy_from_host(
              col0.data(), ng * sizeof(int32_t), _ctx.device(), _ctx.queue(), _ctx.command_pool());
            output.columns[1] = allocate_column(_ctx, {get_col_type(1)}, ng);
            output.columns[1].data.copy_from_host(
              col1.data(), ng * sizeof(int32_t), _ctx.device(), _ctx.queue(), _ctx.command_pool());
          } else {
            std::vector<int32_t> col0(ng), col1(ng), col2(ng);
            for (rasterdf::size_type j = 0; j < ng; j++) {
              int64_t c = sorted_composite_i64[j];
              col2[j]   = static_cast<int32_t>(c % decompose_base2);
              c /= decompose_base2;
              col1[j] = static_cast<int32_t>(c % decompose_base1);
              col0[j] = static_cast<int32_t>(c / decompose_base1);
            }
            output.columns[0] = allocate_column(_ctx, {get_col_type(0)}, ng);
            output.columns[0].data.copy_from_host(
              col0.data(), ng * sizeof(int32_t), _ctx.device(), _ctx.queue(), _ctx.command_pool());
            output.columns[1] = allocate_column(_ctx, {get_col_type(1)}, ng);
            output.columns[1].data.copy_from_host(
              col1.data(), ng * sizeof(int32_t), _ctx.device(), _ctx.queue(), _ctx.command_pool());
            output.columns[2] = allocate_column(_ctx, {get_col_type(2)}, ng);
            output.columns[2].data.copy_from_host(
              col2.data(), ng * sizeof(int32_t), _ctx.device(), _ctx.queue(), _ctx.command_pool());
          }
        }
        keys_set = true;
      }

      // Create gpu_column for sorted values and upload
      size_t out_col_idx  = num_group_cols + i;
      auto sorted_val_col = allocate_column(_ctx, val_col_rdf.type(), ng);
      sorted_val_col.data.copy_from_host(
        sorted_vals.data(), ng * val_elem_size, _ctx.device(), _ctx.queue(), _ctx.command_pool());
      output.columns[out_col_idx] = std::move(sorted_val_col);
    }
  }
  // Propagate dictionary metadata for GROUP BY key columns
  for (size_t g = 0; g < num_group_cols; g++) {
    auto src_idx = group_col_indices[g];
    if (input.dictionaries.has_dict(src_idx)) {
      output.dictionaries.col_dicts[g] = input.dictionaries.get(src_idx);
    }
  }

  RASTERDB_LOG_DEBUG(
    "GROUP BY result: {} groups, {} output cols", num_groups_result, output.columns.size());
}

}  // namespace gpu
}  // namespace rasterdb
