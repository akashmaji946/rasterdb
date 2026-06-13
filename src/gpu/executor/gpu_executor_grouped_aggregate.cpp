/*
 * Copyright 2026, RasterDB Contributors.
 * Split from src/gpu/gpu_executor.cpp.
*/

#include "gpu/gpu_executor_internal.hpp"

#include <rasterdf/gfx_groupby_engine.hpp>

#include <algorithm>
#include <cstdlib>

namespace rasterdb {
namespace gpu {

// Grouped aggregate routing:
//   false = compute tuple-key groupby. This is the correctness/default path and
//           mirrors the cuDF/Sirius model: keep keys as a table of columns and
//           choose dense/hash internally.
//   true  = legacy graphics/GFXM groupby. This is an optional accelerator for
//           simple single-key shapes only.
static constexpr bool USE_SIMPLE_GFX_AGGR = false;

// ============================================================================
// GROUP BY aggregate.
// Compute mode supports fixed-width tuple keys directly. Graphics mode is kept
// narrow and rejects multi-column keys.
// ============================================================================

namespace {

enum int128_groupby_flags : uint32_t {
  I128_GB_COUNT = 1u,
  I128_GB_SUM = 2u,
  I128_GB_MIN = 4u,
  I128_GB_MAX = 8u,
};

static bool grouped_decimal_aggregate_supported(const duckdb::LogicalType& type,
                                                const std::string& function_name)
{
  if (!is_decimal_type(type)) {
    return true;
  }
  if (function_name == "count" || function_name == "count_star") {
    return true;
  }
  auto rdf_type = to_rdf_type(type).id;
  if (function_name == "min" || function_name == "max" ||
      function_name == "sum" || function_name == "sum_no_overflow") {
    return rdf_type == rasterdf::type_id::INT32 ||
           rdf_type == rasterdf::type_id::INT64 ||
           rdf_type == rasterdf::type_id::INT128;
  }
  if (function_name == "avg" || function_name == "mean") {
    return rdf_type == rasterdf::type_id::INT32 ||
           rdf_type == rasterdf::type_id::INT64;
  }
  return false;
}

static bool int128_groupby_agg_kind(const std::string& function_name,
                                    uint32_t& flag)
{
  if (function_name == "count" || function_name == "count_star") {
    flag = I128_GB_COUNT;
    return true;
  }
  if (function_name == "sum" || function_name == "sum_no_overflow") {
    flag = I128_GB_SUM;
    return true;
  }
  if (function_name == "min") {
    flag = I128_GB_MIN;
    return true;
  }
  if (function_name == "max") {
    flag = I128_GB_MAX;
    return true;
  }
  return false;
}

}  // namespace

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
  if (num_group_cols < 1) {
    throw duckdb::NotImplementedException("RasterDB GPU: GROUP BY requires at least one column");
  }

  for (auto& aggregate : aggregates) {
    auto& expr = aggregate->Cast<duckdb::BoundAggregateExpression>();
    if (!expr.children.empty() &&
        !grouped_decimal_aggregate_supported(expr.children[0]->return_type,
                                             expr.function.name)) {
      throw duckdb::NotImplementedException(
        "RasterDB GPU: grouped decimal aggregate '%s' requires DECIMAL64 min/max or fixed-point accumulator support",
        expr.function.name.c_str());
    }
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

  // Early return for empty input — nothing to group
  if (input.num_rows() == 0) {
    RASTERDB_LOG_DEBUG("GROUP BY: 0 input rows, returning empty result");
    output.set_num_rows(0);
    return;
  }

  if (num_group_cols == 1 &&
      input.col(group_col_indices[0]).type.id == rasterdf::type_id::INT128) {
    uint32_t flags = 0;
    duckdb::idx_t value_col_idx = group_col_indices[0];
    bool has_value_agg = false;
    bool seen_count = false;
    bool seen_sum = false;
    bool seen_min = false;
    bool seen_max = false;

    for (auto& aggregate : aggregates) {
      auto& expr = aggregate->Cast<duckdb::BoundAggregateExpression>();
      uint32_t flag = 0;
      if (!int128_groupby_agg_kind(expr.function.name, flag)) {
        throw duckdb::NotImplementedException(
          "RasterDB GPU: DECIMAL128 GROUP BY supports count, sum, min, and max");
      }
      if ((flag == I128_GB_COUNT && seen_count) ||
          (flag == I128_GB_SUM && seen_sum) ||
          (flag == I128_GB_MIN && seen_min) ||
          (flag == I128_GB_MAX && seen_max)) {
        throw duckdb::NotImplementedException(
          "RasterDB GPU: DECIMAL128 GROUP BY currently supports one output per aggregate kind");
      }
      seen_count = seen_count || flag == I128_GB_COUNT;
      seen_sum = seen_sum || flag == I128_GB_SUM;
      seen_min = seen_min || flag == I128_GB_MIN;
      seen_max = seen_max || flag == I128_GB_MAX;
      flags |= flag;

      if (flag != I128_GB_COUNT) {
        if (expr.children.empty()) {
          throw duckdb::NotImplementedException(
            "RasterDB GPU: DECIMAL128 aggregate '%s' requires a value column",
            expr.function.name.c_str());
        }
        auto& child = unwrap_cast(*expr.children[0]);
        if (child.type != duckdb::ExpressionType::BOUND_REF) {
          throw duckdb::NotImplementedException(
            "RasterDB GPU: DECIMAL128 GROUP BY aggregate child must be a column reference");
        }
        auto child_idx = child.Cast<duckdb::BoundReferenceExpression>().index;
        if (input.col(child_idx).type.id != rasterdf::type_id::INT128) {
          throw duckdb::NotImplementedException(
            "RasterDB GPU: DECIMAL128 GROUP BY aggregate child must be DECIMAL128");
        }
        if (has_value_agg && child_idx != value_col_idx) {
          throw duckdb::NotImplementedException(
            "RasterDB GPU: DECIMAL128 GROUP BY currently supports one aggregate value column");
        }
        value_col_idx = child_idx;
        has_value_agg = true;
      }
    }

    uint64_t rows = static_cast<uint64_t>(input.num_rows());
    if (rows > std::numeric_limits<uint32_t>::max()) {
      throw duckdb::NotImplementedException(
        "RasterDB GPU: DECIMAL128 GROUP BY input is too large for this path");
    }

    RASTERDB_LOG_DEBUG("[RDB_OP] groupby path=int128_hash keys=1 aggs={} rows={}",
                       aggregates.size(),
                       input.num_rows());

    auto n = static_cast<rasterdf::size_type>(input.num_rows());
    uint64_t target_slots = std::max<uint64_t>(1024, rows * 2ull);
    if (target_slots > (1ull << 31)) {
      throw duckdb::NotImplementedException(
        "RasterDB GPU: DECIMAL128 hash GROUP BY input is too large for this table path");
    }
    uint32_t table_size = 1024;
    while (static_cast<uint64_t>(table_size) < target_slots) {
      table_size <<= 1u;
    }

    auto out_keys = allocate_column(_ctx, input.col(group_col_indices[0]).type, n);
    auto out_counts = allocate_column(_ctx, {rasterdf::type_id::INT64}, n);
    auto out_sums = allocate_column(_ctx, input.col(value_col_idx).type, n);
    auto out_mins = allocate_column(_ctx, input.col(value_col_idx).type, n);
    auto out_maxs = allocate_column(_ctx, input.col(value_col_idx).type, n);

    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                               VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    rasterdf::device_buffer slot_state(_ctx.workspace_mr(),
                                       static_cast<size_t>(table_size) * sizeof(uint32_t),
                                       usage);
    rasterdf::device_buffer slot_lock(_ctx.workspace_mr(),
                                      static_cast<size_t>(table_size) * sizeof(uint32_t),
                                      usage);
    rasterdf::device_buffer slot_key(_ctx.workspace_mr(),
                                     static_cast<size_t>(table_size) * 16u,
                                     usage);
    rasterdf::device_buffer count_state(_ctx.workspace_mr(),
                                        static_cast<size_t>(table_size) * sizeof(int64_t),
                                        usage);
    rasterdf::device_buffer sum_state(_ctx.workspace_mr(),
                                      static_cast<size_t>(table_size) * 16u,
                                      usage);
    rasterdf::device_buffer min_state(_ctx.workspace_mr(),
                                      static_cast<size_t>(table_size) * 16u,
                                      usage);
    rasterdf::device_buffer max_state(_ctx.workspace_mr(),
                                      static_cast<size_t>(table_size) * 16u,
                                      usage);
    rasterdf::device_buffer unique_count(_ctx.workspace_mr(), sizeof(uint32_t), usage);
    rasterdf::device_buffer overflow_count(_ctx.workspace_mr(), sizeof(uint32_t), usage);
    rasterdf::device_buffer write_idx(_ctx.workspace_mr(), sizeof(uint32_t), usage);

    auto& disp = _ctx.dispatcher();
    disp.begin_batch();
    disp.fill_buffer(slot_state.buffer(), 0, table_size * sizeof(uint32_t), slot_state.offset());
    disp.fill_buffer(slot_lock.buffer(), 0, table_size * sizeof(uint32_t), slot_lock.offset());
    disp.fill_buffer(count_state.buffer(), 0, table_size * sizeof(int64_t), count_state.offset());
    disp.fill_buffer(sum_state.buffer(), 0, static_cast<VkDeviceSize>(table_size) * 16u, sum_state.offset());
    disp.fill_buffer(unique_count.buffer(), 0, sizeof(uint32_t), unique_count.offset());
    disp.fill_buffer(overflow_count.buffer(), 0, sizeof(uint32_t), overflow_count.offset());
    disp.fill_buffer(write_idx.buffer(), 0, sizeof(uint32_t), write_idx.offset());
    disp.batch_barrier_fill_to_compute();
    rasterdf::execution::int128_hash_groupby_build_pc build_pc{};
    build_pc.keys_ptr = input.col(group_col_indices[0]).address();
    build_pc.values_ptr = input.col(value_col_idx).address();
    build_pc.slot_state_ptr = slot_state.data();
    build_pc.slot_lock_ptr = slot_lock.data();
    build_pc.slot_key_ptr = slot_key.data();
    build_pc.count_state_ptr = count_state.data();
    build_pc.sum_state_ptr = sum_state.data();
    build_pc.min_state_ptr = min_state.data();
    build_pc.max_state_ptr = max_state.data();
    build_pc.unique_count_ptr = unique_count.data();
    build_pc.overflow_count_ptr = overflow_count.data();
    build_pc.numRows = static_cast<uint32_t>(rows);
    build_pc.tableSize = table_size;
    build_pc.flags = flags;
    disp.dispatch_int128_hash_groupby_build(build_pc, (build_pc.numRows + 255) / 256);
    disp.end_batch();

    uint32_t unique_groups = 0;
    uint32_t overflow = 0;
    unique_count.copy_to_host(&unique_groups,
                              sizeof(uint32_t),
                              _ctx.device(),
                              _ctx.queue(),
                              _ctx.command_pool());
    overflow_count.copy_to_host(&overflow,
                                sizeof(uint32_t),
                                _ctx.device(),
                                _ctx.queue(),
                                _ctx.command_pool());
    if (overflow != 0) {
      throw duckdb::NotImplementedException(
        "RasterDB GPU: DECIMAL128 hash GROUP BY table overflow (%u rows, table=%u)",
        static_cast<uint32_t>(rows),
        table_size);
    }
    disp.begin_batch();
    disp.fill_buffer(write_idx.buffer(), 0, sizeof(uint32_t), write_idx.offset());
    disp.batch_barrier_fill_to_compute();
    rasterdf::execution::int128_hash_groupby_extract_pc extract_pc{};
    extract_pc.slot_state_ptr = slot_state.data();
    extract_pc.slot_key_ptr = slot_key.data();
    extract_pc.count_state_ptr = count_state.data();
    extract_pc.sum_state_ptr = sum_state.data();
    extract_pc.min_state_ptr = min_state.data();
    extract_pc.max_state_ptr = max_state.data();
    extract_pc.out_keys_ptr = out_keys.address();
    extract_pc.out_counts_ptr = out_counts.address();
    extract_pc.out_sums_ptr = out_sums.address();
    extract_pc.out_mins_ptr = out_mins.address();
    extract_pc.out_maxs_ptr = out_maxs.address();
    extract_pc.write_idx_ptr = write_idx.data();
    extract_pc.tableSize = table_size;
    extract_pc.flags = flags;
    disp.dispatch_int128_hash_groupby_extract(extract_pc, (table_size + 255) / 256);
    disp.end_batch();

    auto ng = static_cast<rasterdf::size_type>(unique_groups);

    output.columns[0] = std::move(out_keys);
    output.columns[0].num_rows = ng;
    for (size_t a = 0; a < aggregates.size(); a++) {
      auto& expr = aggregates[a]->Cast<duckdb::BoundAggregateExpression>();
      uint32_t flag = 0;
      int128_groupby_agg_kind(expr.function.name, flag);
      size_t out_idx = num_group_cols + a;
      if (flag == I128_GB_COUNT) {
        output.columns[out_idx] = std::move(out_counts);
      } else if (flag == I128_GB_SUM) {
        output.columns[out_idx] = std::move(out_sums);
      } else if (flag == I128_GB_MIN) {
        output.columns[out_idx] = std::move(out_mins);
      } else {
        output.columns[out_idx] = std::move(out_maxs);
      }
      output.columns[out_idx].num_rows = ng;
      if (!result_types.empty() && out_idx < result_types.size()) {
        output.columns[out_idx].type = to_rdf_type(result_types[out_idx]);
      }
    }
    output.set_num_rows(ng);
    return;
  }

  if constexpr (!USE_SIMPLE_GFX_AGGR) {
    if (try_execute_multi_key_aggregate(input,
                                        groups,
                                        aggregates,
                                        result_types,
                                        group_col_indices,
                                        1,
                                        output)) {
      return;
    }
    throw duckdb::NotImplementedException(
      "RasterDB GPU: compute GROUP BY tuple path does not support this shape");
  }

  if (num_group_cols != 1) {
    throw duckdb::NotImplementedException(
      "RasterDB GPU: graphics GROUP BY supports only one key column; use compute tuple-key mode for %zu keys",
      num_group_cols);
  }

  // Graphics path key preparation. Multi-column and mixed-type grouping belongs
  // to the compute tuple-key path.
  auto n_rows = input.num_rows();
  const gpu_column* key_col_ptr = nullptr;

  bool single_col_int32 =
    (num_group_cols == 1 && input.col(group_col_indices[0]).type.id == rasterdf::type_id::INT32);

  // STRING groupby key: hash string keys to INT32, then groupby on hashes
  bool single_col_string =
    (num_group_cols == 1 && input.col(group_col_indices[0]).is_string());
  gpu_column string_hash_key;  // keeps hash column alive

  if (single_col_string) {
    // Hash strings → INT32 for groupby
    auto& str_col = input.col(group_col_indices[0]);
    string_hash_key = allocate_column(_ctx, {rasterdf::type_id::INT32}, n_rows);

    string_hash_pc hpc{};
    hpc.offsets_ptr = str_col.str_offsets.data();
    hpc.chars_ptr = str_col.str_chars.data();
    hpc.output_ptr = string_hash_key.address();
    hpc.num_rows = static_cast<uint32_t>(n_rows);
    _ctx.dispatcher().dispatch_string_hash(hpc);

    key_col_ptr = &string_hash_key;
    RASTERDB_LOG_DEBUG("[RDB_DEBUG] GROUP BY: hashed STRING key -> INT32, {} rows", n_rows);
  } else if (single_col_int32) {
    key_col_ptr = &input.col(group_col_indices[0]);
  } else {
    throw duckdb::NotImplementedException(
      "RasterDB GPU: graphics GROUP BY supports only INT32 or STRING keys; use compute tuple-key mode for type_id %d",
      static_cast<int>(input.col(group_col_indices[0]).type.id));
  }

  if (aggregates.empty()) {
    rasterdf::data_type key_type = key_col_ptr->type;
    if (single_col_string) {
      throw duckdb::NotImplementedException(
        "RasterDB GPU: zero-aggregate GROUP BY on STRING key not supported");
    }

    if (key_type.id == rasterdf::type_id::INT32) {
      rasterdf::gfx_groupby_engine_init(_ctx.vk_context());

      rasterdf::device_buffer out_keys(
        _ctx.workspace_mr(),
        0,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
      rasterdf::device_buffer out_values(
        _ctx.workspace_mr(),
        0,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
      uint32_t out_num_groups = 0;

      rasterdf::gfxm_groupby_aggregate(1,
                                       key_col_ptr->address(),
                                       0,
                                       static_cast<uint32_t>(n_rows),
                                       _ctx.dispatcher(),
                                       _ctx.workspace_mr(),
                                       out_keys,
                                       out_values,
                                       out_num_groups,
                                       0,
                                       rasterdf::type_id::INT32);

      output.columns[0].type = key_type;
      output.columns[0].num_rows = static_cast<rasterdf::size_type>(out_num_groups);
      output.columns[0].data = std::move(out_keys);
      output.set_num_rows(static_cast<rasterdf::size_type>(out_num_groups));
      RASTERDB_LOG_DEBUG("GROUP BY distinct GPU result: {} groups, {} output cols",
                         out_num_groups,
                         output.columns.size());
      return;
    }

    throw duckdb::NotImplementedException(
      "RasterDB GPU: unsupported zero-aggregate GROUP BY key type_id %d",
      static_cast<int>(key_type.id));
  }

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

      // Call gfxm groupby. Graphics mode is intentionally single-key only.
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

      rasterdf::data_type val_type;
      if (gfxm_agg_type == 0) {
        val_type = rasterdf::data_type{value_type_id == rasterdf::type_id::FLOAT32
                                         ? rasterdf::type_id::FLOAT64
                                         : rasterdf::type_id::INT64};
      } else if (gfxm_agg_type == 4) {
        val_type = rasterdf::data_type{value_type_id == rasterdf::type_id::INT64
                                         ? rasterdf::type_id::FLOAT64
                                         : rasterdf::type_id::FLOAT32};
      } else if ((gfxm_agg_type == 2 || gfxm_agg_type == 3) &&
                 value_type_id == rasterdf::type_id::INT64) {
        val_type = rasterdf::data_type{rasterdf::type_id::INT64};
      } else {
        val_type = rasterdf::data_type{rasterdf::type_id::INT32};
      }

      rasterdf::data_type key_type = key_col_ptr->type;
      if (aggregates.size() == 1 && !single_col_string) {
        num_groups_result = out_num_groups;
        output.columns[0].type = key_type;
        output.columns[0].num_rows = static_cast<rasterdf::size_type>(out_num_groups);
        output.columns[0].data = std::move(out_keys);
        output.columns[1].type = val_type;
        output.columns[1].num_rows = static_cast<rasterdf::size_type>(out_num_groups);
        output.columns[1].data = std::move(out_values);
        keys_set = true;
        continue;
      }

      rasterdf::column key_col_rdf(key_type, out_num_groups, std::move(out_keys));
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
        if (single_col_string) {
          // Replace INT32 hash keys with original STRING keys via gather.
          auto& str_col = input.col(group_col_indices[0]);
          auto& hash_keys = sorted_key_col;

          rasterdf::device_buffer first_idx_buf(
              _ctx.workspace_mr(), out_num_groups * sizeof(int32_t),
              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
              VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
          _ctx.dispatcher().fill_buffer(first_idx_buf.buffer(), 0xFFFFFFFFu,
                       out_num_groups * sizeof(int32_t), first_idx_buf.offset());

          find_first_index_pc fpc{};
          fpc.all_keys_ptr = string_hash_key.address();
          fpc.unique_keys_ptr = hash_keys.address();
          fpc.first_idx_ptr = first_idx_buf.data();
          fpc.numElements = static_cast<uint32_t>(n_rows);
          fpc.numUnique = out_num_groups;
          _ctx.dispatcher().dispatch_find_first_index(fpc, (n_rows + 255) / 256);

          string_lengths_pc lpc{};
          lpc.offsets_ptr = str_col.str_offsets.data();
          lpc.indices_ptr = first_idx_buf.data();
          lpc.num_indices = out_num_groups;

          rasterdf::device_buffer out_offsets(
              _ctx.workspace_mr(), (out_num_groups + 1) * sizeof(int32_t),
              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
              VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
          lpc.output_ptr = out_offsets.data();
          _ctx.dispatcher().dispatch_string_lengths(lpc);
          _ctx.dispatcher().fill_buffer(out_offsets.buffer(), 0u, sizeof(int32_t),
                       out_offsets.offset() + out_num_groups * sizeof(int32_t));

          uint32_t scan_elems = out_num_groups + 1;
          uint32_t scan_ngroups = (scan_elems + 255) / 256;
          rasterdf::device_buffer scan_bsums(
              _ctx.workspace_mr(), scan_ngroups * sizeof(uint32_t),
              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
              VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
          rasterdf::device_buffer scan_total(
              _ctx.workspace_mr(), sizeof(uint32_t),
              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
              VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

          prefix_scan_pc opc{};
          opc.data_ptr = out_offsets.data();
          opc.block_sums_ptr = scan_bsums.data();
          opc.total_sum_ptr = scan_total.data();
          opc.numElements = scan_elems;
          opc.blockCount = scan_ngroups;
          _ctx.dispatcher().dispatch_prefix_scan_local(opc, scan_ngroups);
          _ctx.dispatcher().dispatch_prefix_scan_global(opc);
          _ctx.dispatcher().dispatch_prefix_scan_add(opc, scan_ngroups);

          int32_t total_out_chars = 0;
          scan_total.copy_to_host(&total_out_chars, sizeof(int32_t),
                                  _ctx.device(), _ctx.queue(), _ctx.command_pool());

          rasterdf::device_buffer out_chars(
              _ctx.workspace_mr(), std::max(total_out_chars, 1),
              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
              VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

          string_copy_pc cpc{};
          cpc.in_offsets_ptr = str_col.str_offsets.data();
          cpc.in_chars_ptr = str_col.str_chars.data();
          cpc.indices_ptr = first_idx_buf.data();
          cpc.out_offsets_ptr = out_offsets.data();
          cpc.out_chars_ptr = out_chars.data();
          cpc.num_indices = out_num_groups;
          _ctx.dispatcher().dispatch_string_copy(cpc);

          output.columns[0].type = rasterdf::data_type{rasterdf::type_id::STRING};
          output.columns[0].num_rows = static_cast<rasterdf::size_type>(out_num_groups);
          output.columns[0].str_offsets = std::move(out_offsets);
          output.columns[0].str_chars = std::move(out_chars);
          output.columns[0].str_total_chars = total_out_chars;
          RASTERDB_LOG_DEBUG("[GFXM] STRING groupby: {} unique groups, {} total chars",
                             out_num_groups, total_out_chars);
        } else {
          output.columns[0] = std::move(sorted_key_col);
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
  output.set_num_rows(num_groups_result);
  RASTERDB_LOG_DEBUG(
    "GROUP BY result: {} groups, {} output cols", num_groups_result, output.columns.size());
}

}  // namespace gpu
}  // namespace rasterdb
