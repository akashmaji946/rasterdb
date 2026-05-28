/*
 * Copyright 2026, RasterDB Contributors.
 * Split from src/gpu/gpu_executor.cpp.
*/

#include "gpu/gpu_executor_internal.hpp"

#include <rasterdf/gfx_groupby_engine.hpp>

#include <cstdlib>

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

static constexpr bool USE_TUPLE_GROUPBY_INT32 = true;

namespace {

struct tuple_key_desc_host {
  VkDeviceAddress values_i32;
  VkDeviceAddress out_i32;
  uint32_t type_id;
  uint32_t pad0;
  uint32_t pad1;
  uint32_t pad2;
};

struct tuple_agg_desc_host {
  VkDeviceAddress values_i32;
  VkDeviceAddress state_i32;
  VkDeviceAddress state_i64;
  VkDeviceAddress aux_i64;
  VkDeviceAddress out_i32;
  VkDeviceAddress out_i64;
  VkDeviceAddress out_f32_bits;
  uint32_t kind;
  uint32_t value_type_id;
  uint32_t output_type_id;
  uint32_t flags;
};

enum tuple_groupby_agg_kind : uint32_t {
  TUPLE_GB_SUM = 0,
  TUPLE_GB_COUNT = 1,
  TUPLE_GB_MIN = 2,
  TUPLE_GB_MAX = 3,
  TUPLE_GB_MEAN = 4,
};

}  // namespace

static bool tuple_groupby_fixed_width_supported(rasterdf::type_id id)
{
  return id == rasterdf::type_id::INT32 || id == rasterdf::type_id::INT64 ||
         id == rasterdf::type_id::FLOAT32 || id == rasterdf::type_id::FLOAT64;
}

static uint64_t double_bits(double value)
{
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static uint32_t float_bits(float value)
{
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static rasterdf::type_id tuple_groupby_output_type(uint32_t kind, rasterdf::type_id value_type)
{
  if (kind == TUPLE_GB_COUNT) {
    return rasterdf::type_id::INT32;
  }
  if (kind == TUPLE_GB_SUM) {
    if (value_type == rasterdf::type_id::FLOAT32 || value_type == rasterdf::type_id::FLOAT64) {
      return value_type;
    }
    return rasterdf::type_id::INT64;
  }
  if (kind == TUPLE_GB_MEAN) {
    if (value_type == rasterdf::type_id::INT64 || value_type == rasterdf::type_id::FLOAT64) {
      return rasterdf::type_id::FLOAT64;
    }
    return rasterdf::type_id::FLOAT32;
  }
  return value_type;
}

struct dense_key_range_info {
  bool supported = false;
  uint64_t group_count = 0;
  std::vector<int32_t> mins;
  std::vector<uint32_t> ranges;
  std::vector<uint32_t> strides;
};

static bool dense_groupby_agg_supported(uint32_t kind)
{
  return kind == TUPLE_GB_SUM || kind == TUPLE_GB_COUNT || kind == TUPLE_GB_MEAN;
}

static rasterdf::type_id dense_groupby_output_type(uint32_t kind, rasterdf::type_id value_type)
{
  if ((kind == TUPLE_GB_SUM || kind == TUPLE_GB_MEAN) &&
      value_type == rasterdf::type_id::FLOAT32) {
    return rasterdf::type_id::FLOAT64;
  }
  return tuple_groupby_output_type(kind, value_type);
}

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
  if (rdf_type != rasterdf::type_id::INT32 && rdf_type != rasterdf::type_id::INT64) {
    return false;
  }
  if (function_name == "min" || function_name == "max" ||
      function_name == "sum" || function_name == "sum_no_overflow" ||
      function_name == "avg" || function_name == "mean") {
    return true;
  }
  return false;
}

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

  if constexpr (USE_TUPLE_GROUPBY_INT32) {
    bool tuple_low_cardinality_contention = false;
    dense_key_range_info dense_info;
    if (num_group_cols >= 1 && input.num_rows() >= 1024) {
      bool can_estimate = true;
      uint64_t range_product = 1;
      dense_info.mins.reserve(num_group_cols);
      dense_info.ranges.reserve(num_group_cols);
      for (auto idx : group_col_indices) {
        const auto& gcol = input.col(idx);
        if (gcol.type.id != rasterdf::type_id::INT32) {
          can_estimate = false;
          break;
        }
        auto col_view = gcol.view();
        rasterdf::reduce_aggregation min_agg(rasterdf::aggregation_kind::MIN);
        rasterdf::reduce_aggregation max_agg(rasterdf::aggregation_kind::MAX);
        auto min_s = rasterdf::reduce(col_view,
                                      min_agg,
                                      rasterdf::data_type{rasterdf::type_id::INT32},
                                      _ctx.vk_context(),
                                      _ctx.dispatcher(),
                                      _ctx.workspace_mr());
        auto max_s = rasterdf::reduce(col_view,
                                      max_agg,
                                      rasterdf::data_type{rasterdf::type_id::INT32},
                                      _ctx.vk_context(),
                                      _ctx.dispatcher(),
                                      _ctx.workspace_mr());
        int64_t min_v = static_cast<int64_t>(min_s->as<int32_t>());
        int64_t max_v = static_cast<int64_t>(max_s->as<int32_t>());
        if (max_v < min_v) {
          can_estimate = false;
          break;
        }
        uint64_t range = static_cast<uint64_t>(max_v - min_v + 1);
        if (range == 0 || range_product > 65536ull / range) {
          can_estimate = false;
          break;
        }
        dense_info.mins.push_back(static_cast<int32_t>(min_v));
        dense_info.ranges.push_back(static_cast<uint32_t>(range));
        range_product *= range;
      }
      dense_info.supported = can_estimate && range_product <= 65536ull;
      dense_info.group_count = range_product;
      if (dense_info.supported) {
        dense_info.strides.resize(num_group_cols);
        uint64_t stride = 1;
        for (int64_t k = static_cast<int64_t>(num_group_cols) - 1; k >= 0; --k) {
          dense_info.strides[static_cast<size_t>(k)] = static_cast<uint32_t>(stride);
          stride *= dense_info.ranges[static_cast<size_t>(k)];
        }
      }
      tuple_low_cardinality_contention = dense_info.supported;
      if (tuple_low_cardinality_contention) {
        RASTERDB_LOG_DEBUG(
          "[RDB_OP] groupby path=dense_candidate reason=low_cardinality groups_est={} rows={}",
          range_product,
          input.num_rows());
      }
    }

    bool tuple_candidate = num_group_cols >= 2;
    for (auto idx : group_col_indices) {
      tuple_candidate = tuple_candidate && tuple_groupby_fixed_width_supported(input.col(idx).type.id);
    }

    if (tuple_candidate && !aggregates.empty()) {
      std::vector<gpu_column> value_temps;
      value_temps.reserve(aggregates.size());
      std::vector<const gpu_column*> value_cols;
      value_cols.reserve(aggregates.size());
      std::vector<uint32_t> agg_kinds;
      agg_kinds.reserve(aggregates.size());
      std::vector<rasterdf::type_id> value_types;
      value_types.reserve(aggregates.size());

      for (duckdb::idx_t i = 0; i < aggregates.size() && tuple_candidate; i++) {
        auto& expr  = aggregates[i]->Cast<duckdb::BoundAggregateExpression>();
        auto& fname = expr.function.name;

        bool is_count_star = false;
        if (expr.children.empty()) {
          is_count_star = (fname == "count" || fname == "count_star");
          tuple_candidate = tuple_candidate && is_count_star;
        }

        uint32_t kind = TUPLE_GB_COUNT;
        if (fname == "sum" || fname == "sum_no_overflow") {
          kind = TUPLE_GB_SUM;
        } else if (fname == "min") {
          kind = TUPLE_GB_MIN;
        } else if (fname == "max") {
          kind = TUPLE_GB_MAX;
        } else if (fname == "count" || fname == "count_star") {
          kind = TUPLE_GB_COUNT;
        } else if (fname == "avg" || fname == "mean") {
          kind = TUPLE_GB_MEAN;
        } else {
          tuple_candidate = false;
        }

        if (!tuple_candidate) {
          break;
        }

        if (is_count_star) {
          value_cols.push_back(&input.col(group_col_indices[0]));
          value_types.push_back(input.col(group_col_indices[0]).type.id);
        } else {
          value_temps.push_back(evaluate_expression(input, *expr.children[0]));
          if (!tuple_groupby_fixed_width_supported(value_temps.back().type.id)) {
            tuple_candidate = false;
            break;
          }
          value_cols.push_back(&value_temps.back());
          value_types.push_back(value_temps.back().type.id);
        }
        agg_kinds.push_back(kind);
      }

      if (tuple_candidate) {
        bool dense_candidate = tuple_low_cardinality_contention && dense_info.supported &&
                               dense_info.group_count > 0 &&
                               dense_info.group_count <= 65536ull;
        for (auto idx : group_col_indices) {
          dense_candidate = dense_candidate &&
                            input.col(idx).type.id == rasterdf::type_id::INT32;
        }
        for (size_t a = 0; a < agg_kinds.size(); ++a) {
          dense_candidate = dense_candidate &&
                            dense_groupby_agg_supported(agg_kinds[a]);
        }

        auto n_rows = input.num_rows();
        auto n = static_cast<uint32_t>(n_rows);
        uint32_t dense_num_groups = static_cast<uint32_t>(dense_info.group_count);
        uint32_t dense_num_workgroups = (n + 255u) / 256u;
        uint64_t dense_partial_entries =
          static_cast<uint64_t>(dense_num_groups) * dense_num_workgroups;
        uint64_t dense_state_entries = dense_partial_entries + dense_num_groups;
        dense_candidate = dense_candidate && dense_partial_entries <= 64000000ull;

        if (dense_candidate) {
          stage_timer dense_timer("    groupby_dense_local");
          auto dense_total_start = std::chrono::high_resolution_clock::now();
          auto dense_stage_start = dense_total_start;
          auto elapsed_ms = [](auto start, auto end) {
            return std::chrono::duration<double, std::milli>(end - start).count();
          };
          bool profile_dense_phases = std::getenv("RASTERDB_DENSE_GB_PROFILE_PHASES") != nullptr;
          RASTERDB_LOG_DEBUG(
            "[RDB_OP] groupby path=dense_local keys={} aggs={} rows={} groups_est={} workgroups={}",
            num_group_cols,
            aggregates.size(),
            input.num_rows(),
            dense_num_groups,
            dense_num_workgroups);
          RASTERDB_LOG_INFO(
            "[Dense GB Profile] mode={} rows={} groups_est={} partial_entries={} partial_mb={:.2f}",
            profile_dense_phases ? "phase" : "batch",
            input.num_rows(),
            dense_num_groups,
            dense_partial_entries,
            static_cast<double>(dense_state_entries) *
              (sizeof(uint32_t) + aggregates.size() * (sizeof(int32_t) + 2.0 * sizeof(int64_t))) /
              (1024.0 * 1024.0));

          VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

          std::vector<rasterdf::device_buffer> out_key_bufs;
          out_key_bufs.reserve(num_group_cols);
          std::vector<tuple_key_desc_host> key_descs(num_group_cols);
          for (size_t k = 0; k < num_group_cols; k++) {
            out_key_bufs.emplace_back(_ctx.workspace_mr(),
                                      static_cast<size_t>(dense_num_groups) * sizeof(int32_t),
                                      usage);
            key_descs[k] = {
              input.col(group_col_indices[k]).address(),
              out_key_bufs.back().data(),
              static_cast<uint32_t>(input.col(group_col_indices[k]).type.id),
              static_cast<uint32_t>(dense_info.mins[k]),
              dense_info.strides[k],
              dense_info.ranges[k],
            };
          }

          std::vector<rasterdf::device_buffer> state_i32_bufs;
          std::vector<rasterdf::device_buffer> state_i64_bufs;
          std::vector<rasterdf::device_buffer> aux_i64_bufs;
          std::vector<rasterdf::device_buffer> out_i32_bufs;
          std::vector<rasterdf::device_buffer> out_i64_bufs;
          std::vector<rasterdf::device_buffer> out_f32_bufs;
          state_i32_bufs.reserve(aggregates.size());
          state_i64_bufs.reserve(aggregates.size());
          aux_i64_bufs.reserve(aggregates.size());
          out_i32_bufs.reserve(aggregates.size());
          out_i64_bufs.reserve(aggregates.size());
          out_f32_bufs.reserve(aggregates.size());

          std::vector<tuple_agg_desc_host> agg_descs(aggregates.size());
          for (size_t a = 0; a < aggregates.size(); a++) {
            state_i32_bufs.emplace_back(_ctx.workspace_mr(),
                                        static_cast<size_t>(dense_state_entries) * sizeof(int32_t),
                                        usage);
            state_i64_bufs.emplace_back(_ctx.workspace_mr(),
                                        static_cast<size_t>(dense_state_entries) * sizeof(int64_t),
                                        usage);
            aux_i64_bufs.emplace_back(_ctx.workspace_mr(),
                                      static_cast<size_t>(dense_state_entries) * sizeof(int64_t),
                                      usage);
            out_i32_bufs.emplace_back(_ctx.workspace_mr(),
                                      static_cast<size_t>(dense_num_groups) * sizeof(int32_t),
                                      usage);
            out_i64_bufs.emplace_back(_ctx.workspace_mr(),
                                      static_cast<size_t>(dense_num_groups) * sizeof(int64_t),
                                      usage);
            out_f32_bufs.emplace_back(_ctx.workspace_mr(),
                                      static_cast<size_t>(dense_num_groups) * sizeof(float),
                                      usage);

            auto output_type_id = dense_groupby_output_type(agg_kinds[a], value_types[a]);
            agg_descs[a] = {
              value_cols[a]->address(),
              state_i32_bufs.back().data(),
              state_i64_bufs.back().data(),
              aux_i64_bufs.back().data(),
              out_i32_bufs.back().data(),
              out_i64_bufs.back().data(),
              out_f32_bufs.back().data(),
              agg_kinds[a],
              static_cast<uint32_t>(value_types[a]),
              static_cast<uint32_t>(output_type_id),
              0,
            };
          }
          auto dense_alloc_end = std::chrono::high_resolution_clock::now();
          RASTERDB_LOG_INFO("[Dense GB Profile] alloc_desc_build_ms={:.2f}",
                            elapsed_ms(dense_stage_start, dense_alloc_end));
          dense_stage_start = dense_alloc_end;

          rasterdf::device_buffer key_desc_buf(_ctx.workspace_mr(),
                                               key_descs.size() * sizeof(tuple_key_desc_host),
                                               usage);
          rasterdf::device_buffer agg_desc_buf(_ctx.workspace_mr(),
                                               agg_descs.size() * sizeof(tuple_agg_desc_host),
                                               usage);
          key_desc_buf.copy_from_host(key_descs.data(),
                                      key_descs.size() * sizeof(tuple_key_desc_host),
                                      _ctx.device(),
                                      _ctx.queue(),
                                      _ctx.command_pool());
          agg_desc_buf.copy_from_host(agg_descs.data(),
                                      agg_descs.size() * sizeof(tuple_agg_desc_host),
                                      _ctx.device(),
                                      _ctx.queue(),
                                      _ctx.command_pool());
          auto dense_desc_upload_end = std::chrono::high_resolution_clock::now();
          RASTERDB_LOG_INFO("[Dense GB Profile] desc_upload_ms={:.2f} key_desc_bytes={} agg_desc_bytes={}",
                            elapsed_ms(dense_stage_start, dense_desc_upload_end),
                            key_descs.size() * sizeof(tuple_key_desc_host),
                            agg_descs.size() * sizeof(tuple_agg_desc_host));
          dense_stage_start = dense_desc_upload_end;

          rasterdf::device_buffer group_counts(
            _ctx.workspace_mr(), static_cast<size_t>(dense_state_entries) * sizeof(uint32_t), usage);
          rasterdf::device_buffer write_idx(_ctx.workspace_mr(), sizeof(uint32_t), usage);
          rasterdf::device_buffer overflow_count(_ctx.workspace_mr(), sizeof(uint32_t), usage);
          auto dense_counter_alloc_end = std::chrono::high_resolution_clock::now();
          RASTERDB_LOG_INFO("[Dense GB Profile] counter_alloc_ms={:.2f}",
                            elapsed_ms(dense_stage_start, dense_counter_alloc_end));
          dense_stage_start = dense_counter_alloc_end;

          auto& disp = _ctx.dispatcher();
          tuple_groupby_dense_pc dense_pc{};
          dense_pc.key_descs_ptr = key_desc_buf.data();
          dense_pc.agg_descs_ptr = agg_desc_buf.data();
          dense_pc.group_counts_ptr = group_counts.data();
          dense_pc.write_idx_ptr = overflow_count.data();
          dense_pc.numRows = n;
          dense_pc.numKeys = static_cast<uint32_t>(num_group_cols);
          dense_pc.numAggs = static_cast<uint32_t>(aggregates.size());
          dense_pc.numGroups = dense_num_groups;
          dense_pc.numWorkgroups = dense_num_workgroups;

          auto record_fills = [&]() {
            disp.fill_buffer(group_counts.buffer(),
                             0,
                             static_cast<VkDeviceSize>(dense_state_entries * sizeof(uint32_t)),
                             group_counts.offset());
            disp.fill_buffer(write_idx.buffer(), 0, sizeof(uint32_t), write_idx.offset());
            disp.fill_buffer(overflow_count.buffer(), 0, sizeof(uint32_t), overflow_count.offset());
            for (size_t a = 0; a < aggregates.size(); a++) {
              disp.fill_buffer(state_i32_bufs[a].buffer(),
                               0,
                               static_cast<VkDeviceSize>(dense_state_entries * sizeof(int32_t)),
                               state_i32_bufs[a].offset());
              disp.fill_buffer(state_i64_bufs[a].buffer(),
                               0,
                               static_cast<VkDeviceSize>(dense_state_entries * sizeof(int64_t)),
                               state_i64_bufs[a].offset());
              disp.fill_buffer(aux_i64_bufs[a].buffer(),
                               0,
                               static_cast<VkDeviceSize>(dense_state_entries * sizeof(int64_t)),
                               aux_i64_bufs[a].offset());
            }
            disp.batch_barrier_fill_to_compute();
          };

          if (profile_dense_phases) {
            auto phase_start = std::chrono::high_resolution_clock::now();
            disp.begin_batch();
            record_fills();
            disp.end_batch();
            auto phase_end = std::chrono::high_resolution_clock::now();
            RASTERDB_LOG_INFO("[Dense GB Profile] gpu_fill_ms={:.2f}",
                              elapsed_ms(phase_start, phase_end));

            phase_start = std::chrono::high_resolution_clock::now();
            disp.begin_batch();
            disp.dispatch_tuple_groupby_dense_partial(dense_pc, dense_num_workgroups);
            disp.end_batch();
            phase_end = std::chrono::high_resolution_clock::now();
            RASTERDB_LOG_INFO("[Dense GB Profile] gpu_partial_ms={:.2f}",
                              elapsed_ms(phase_start, phase_end));

            uint32_t overflow = 0;
            auto overflow_start = std::chrono::high_resolution_clock::now();
            overflow_count.copy_to_host(&overflow,
                                        sizeof(uint32_t),
                                        _ctx.device(),
                                        _ctx.queue(),
                                        _ctx.command_pool());
            auto overflow_end = std::chrono::high_resolution_clock::now();
            RASTERDB_LOG_INFO("[Dense GB Profile] overflow_readback_ms={:.2f} overflow={}",
                              elapsed_ms(overflow_start, overflow_end),
                              overflow);
            if (overflow != 0) {
              throw duckdb::NotImplementedException(
                "RasterDB GPU: dense GROUP BY saw %u rows outside estimated key ranges", overflow);
            }

            phase_start = std::chrono::high_resolution_clock::now();
            disp.begin_batch();
            dense_pc.write_idx_ptr = write_idx.data();
            disp.dispatch_tuple_groupby_dense_reduce(dense_pc, dense_num_groups);
            disp.end_batch();
            phase_end = std::chrono::high_resolution_clock::now();
            RASTERDB_LOG_INFO("[Dense GB Profile] gpu_reduce_ms={:.2f}",
                              elapsed_ms(phase_start, phase_end));

            phase_start = std::chrono::high_resolution_clock::now();
            disp.begin_batch();
            disp.dispatch_tuple_groupby_dense_extract(dense_pc, (dense_num_groups + 255u) / 256u);
            disp.end_batch();
            phase_end = std::chrono::high_resolution_clock::now();
            RASTERDB_LOG_INFO("[Dense GB Profile] gpu_extract_ms={:.2f}",
                              elapsed_ms(phase_start, phase_end));
          } else {
            auto batch_start = std::chrono::high_resolution_clock::now();
            disp.begin_batch();
            record_fills();
            disp.dispatch_tuple_groupby_dense_partial(dense_pc, dense_num_workgroups);
            disp.batch_barrier();
            dense_pc.write_idx_ptr = write_idx.data();
            disp.dispatch_tuple_groupby_dense_reduce(dense_pc, dense_num_groups);
            disp.batch_barrier();
            disp.dispatch_tuple_groupby_dense_extract(dense_pc, (dense_num_groups + 255u) / 256u);
            disp.end_batch();
            auto batch_end = std::chrono::high_resolution_clock::now();
            RASTERDB_LOG_INFO("[Dense GB Profile] gpu_batch_ms={:.2f}",
                              elapsed_ms(batch_start, batch_end));
          }

          uint32_t num_unique_groups = 0;
          auto readback_start = std::chrono::high_resolution_clock::now();
          write_idx.copy_to_host(&num_unique_groups,
                                 sizeof(uint32_t),
                                 _ctx.device(),
                                 _ctx.queue(),
                                 _ctx.command_pool());
          auto readback_end = std::chrono::high_resolution_clock::now();
          RASTERDB_LOG_INFO("[Dense GB Profile] write_idx_readback_ms={:.2f} unique_groups={}",
                            elapsed_ms(readback_start, readback_end),
                            num_unique_groups);
          auto ng = static_cast<rasterdf::size_type>(num_unique_groups);
          auto finalize_start = std::chrono::high_resolution_clock::now();
          for (size_t k = 0; k < num_group_cols; k++) {
            output.columns[k].type = input.col(group_col_indices[k]).type;
            output.columns[k].num_rows = ng;
            output.columns[k].data = std::move(out_key_bufs[k]);
          }
          for (size_t a = 0; a < aggregates.size(); a++) {
            size_t out_col_idx = num_group_cols + a;
            auto output_type = dense_groupby_output_type(agg_kinds[a], value_types[a]);
            output.columns[out_col_idx].type = {output_type};
            output.columns[out_col_idx].num_rows = ng;
            if (output_type == rasterdf::type_id::INT64 ||
                output_type == rasterdf::type_id::FLOAT64) {
              output.columns[out_col_idx].data = std::move(out_i64_bufs[a]);
            } else if (output_type == rasterdf::type_id::FLOAT32) {
              output.columns[out_col_idx].data = std::move(out_f32_bufs[a]);
            } else {
              output.columns[out_col_idx].data = std::move(out_i32_bufs[a]);
            }
          }
          output.set_num_rows(ng);
          auto dense_total_end = std::chrono::high_resolution_clock::now();
          RASTERDB_LOG_INFO("[Dense GB Profile] finalize_ms={:.2f} total_dense_path_ms={:.2f}",
                            elapsed_ms(finalize_start, dense_total_end),
                            elapsed_ms(dense_total_start, dense_total_end));
          RASTERDB_LOG_DEBUG("[Dense GB] rows={} keys={} aggs={} groups={} partial_entries={}",
                             n,
                             num_group_cols,
                             aggregates.size(),
                             num_unique_groups,
                             dense_partial_entries);
          return;
        }

        if (tuple_low_cardinality_contention) {
          RASTERDB_LOG_DEBUG(
            "[Tuple GB] low-cardinality shape not supported by dense path; falling back");
          tuple_candidate = false;
        }
      }

      if (tuple_candidate) {
        stage_timer tuple_timer("    groupby_tuple_fixed_width");
        RASTERDB_LOG_DEBUG("[RDB_OP] groupby path=tuple_fixed_width keys={} aggs={} rows={}",
                          num_group_cols,
                          aggregates.size(),
                          input.num_rows());
        auto n_rows = input.num_rows();
        auto n = static_cast<uint32_t>(n_rows);

        auto next_pow2 = [](uint64_t value) -> uint32_t {
          uint64_t power = 1;
          while (power < value && power < (1ull << 31)) {
            power <<= 1ull;
          }
          return static_cast<uint32_t>(power);
        };

        uint64_t target_slots = std::max<uint64_t>(1024, static_cast<uint64_t>(n) * 2ull);
        bool can_range_estimate = true;
        uint64_t range_product = 1;
        for (auto idx : group_col_indices) {
          const auto& gcol = input.col(idx);
          if (gcol.type.id != rasterdf::type_id::INT32) {
            can_range_estimate = false;
            break;
          }
          auto col_view = gcol.view();
          rasterdf::reduce_aggregation min_agg(rasterdf::aggregation_kind::MIN);
          rasterdf::reduce_aggregation max_agg(rasterdf::aggregation_kind::MAX);
          auto min_s = rasterdf::reduce(col_view,
                                        min_agg,
                                        rasterdf::data_type{rasterdf::type_id::INT32},
                                        _ctx.vk_context(),
                                        _ctx.dispatcher(),
                                        _ctx.workspace_mr());
          auto max_s = rasterdf::reduce(col_view,
                                        max_agg,
                                        rasterdf::data_type{rasterdf::type_id::INT32},
                                        _ctx.vk_context(),
                                        _ctx.dispatcher(),
                                        _ctx.workspace_mr());
          int64_t min_v = static_cast<int64_t>(min_s->as<int32_t>());
          int64_t max_v = static_cast<int64_t>(max_s->as<int32_t>());
          if (max_v < min_v) {
            can_range_estimate = false;
            break;
          }
          uint64_t range = static_cast<uint64_t>(max_v - min_v + 1);
          if (range == 0 || range_product > (static_cast<uint64_t>(n) * 2ull) / range) {
            can_range_estimate = false;
            break;
          }
          range_product *= range;
        }
        if (can_range_estimate) {
          target_slots = std::max<uint64_t>(1024, range_product * 4ull);
          RASTERDB_LOG_DEBUG("[Tuple GB] range_estimate={} target_slots={}",
                             range_product,
                             target_slots);
        }

        uint32_t table_size = 1024;
        uint32_t target = next_pow2(target_slots);
        while (table_size < target) {
          table_size <<= 1u;
        }

        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

        std::vector<rasterdf::device_buffer> out_key_bufs;
        out_key_bufs.reserve(num_group_cols);
        std::vector<tuple_key_desc_host> key_descs(num_group_cols);
        for (size_t k = 0; k < num_group_cols; k++) {
          out_key_bufs.emplace_back(_ctx.workspace_mr(),
                                    static_cast<size_t>(n) *
                                      rdf_type_size(input.col(group_col_indices[k]).type.id),
                                    usage);
          key_descs[k] = {
            input.col(group_col_indices[k]).address(),
            out_key_bufs.back().data(),
            static_cast<uint32_t>(input.col(group_col_indices[k]).type.id),
            0,
            0,
            0,
          };
        }

        std::vector<rasterdf::device_buffer> state_i32_bufs;
        std::vector<rasterdf::device_buffer> state_i64_bufs;
        std::vector<rasterdf::device_buffer> aux_i64_bufs;
        std::vector<rasterdf::device_buffer> out_i32_bufs;
        std::vector<rasterdf::device_buffer> out_i64_bufs;
        std::vector<rasterdf::device_buffer> out_f32_bufs;
        state_i32_bufs.reserve(aggregates.size());
        state_i64_bufs.reserve(aggregates.size());
        aux_i64_bufs.reserve(aggregates.size());
        out_i32_bufs.reserve(aggregates.size());
        out_i64_bufs.reserve(aggregates.size());
        out_f32_bufs.reserve(aggregates.size());

        std::vector<tuple_agg_desc_host> agg_descs(aggregates.size());
        for (size_t a = 0; a < aggregates.size(); a++) {
          state_i32_bufs.emplace_back(_ctx.workspace_mr(),
                                      static_cast<size_t>(table_size) * sizeof(int32_t),
                                      usage);
          state_i64_bufs.emplace_back(_ctx.workspace_mr(),
                                      static_cast<size_t>(table_size) * sizeof(int64_t),
                                      usage);
          aux_i64_bufs.emplace_back(_ctx.workspace_mr(),
                                    static_cast<size_t>(table_size) * sizeof(int64_t),
                                    usage);
          out_i32_bufs.emplace_back(_ctx.workspace_mr(),
                                    static_cast<size_t>(n) * sizeof(int32_t),
                                    usage);
          out_i64_bufs.emplace_back(_ctx.workspace_mr(),
                                    static_cast<size_t>(n) * sizeof(int64_t),
                                    usage);
          out_f32_bufs.emplace_back(_ctx.workspace_mr(),
                                    static_cast<size_t>(n) * sizeof(float),
                                    usage);

          auto kind = agg_kinds[a];
          auto output_type_id = tuple_groupby_output_type(kind, value_types[a]);
          agg_descs[a] = {
            value_cols[a]->address(),
            state_i32_bufs.back().data(),
            state_i64_bufs.back().data(),
            aux_i64_bufs.back().data(),
            out_i32_bufs.back().data(),
            out_i64_bufs.back().data(),
            out_f32_bufs.back().data(),
            kind,
            static_cast<uint32_t>(value_types[a]),
            static_cast<uint32_t>(output_type_id),
            0,
          };
        }

        rasterdf::device_buffer key_desc_buf(_ctx.workspace_mr(),
                                             key_descs.size() * sizeof(tuple_key_desc_host),
                                             usage);
        rasterdf::device_buffer agg_desc_buf(_ctx.workspace_mr(),
                                             agg_descs.size() * sizeof(tuple_agg_desc_host),
                                             usage);
        key_desc_buf.copy_from_host(key_descs.data(),
                                    key_descs.size() * sizeof(tuple_key_desc_host),
                                    _ctx.device(),
                                    _ctx.queue(),
                                    _ctx.command_pool());
        agg_desc_buf.copy_from_host(agg_descs.data(),
                                    agg_descs.size() * sizeof(tuple_agg_desc_host),
                                    _ctx.device(),
                                    _ctx.queue(),
                                    _ctx.command_pool());

        rasterdf::device_buffer slot_state(_ctx.workspace_mr(),
                                           static_cast<size_t>(table_size) * sizeof(uint32_t),
                                           usage);
        rasterdf::device_buffer slot_hash(_ctx.workspace_mr(),
                                          static_cast<size_t>(table_size) * sizeof(uint64_t),
                                          usage);
        rasterdf::device_buffer slot_row(_ctx.workspace_mr(),
                                         static_cast<size_t>(table_size) * sizeof(uint32_t),
                                         usage);
        rasterdf::device_buffer unique_count(_ctx.workspace_mr(), sizeof(uint32_t), usage);
        rasterdf::device_buffer overflow_count(_ctx.workspace_mr(), sizeof(uint32_t), usage);

        std::vector<bool> state_i64_initialized_from_host(aggregates.size(), false);
        for (size_t a = 0; a < aggregates.size(); a++) {
          if ((value_types[a] == rasterdf::type_id::INT64 ||
               value_types[a] == rasterdf::type_id::FLOAT64) &&
              (agg_kinds[a] == TUPLE_GB_MIN || agg_kinds[a] == TUPLE_GB_MAX)) {
            uint64_t init_i64 = 0;
            if (value_types[a] == rasterdf::type_id::INT64) {
              init_i64 = agg_kinds[a] == TUPLE_GB_MIN
                           ? static_cast<uint64_t>(std::numeric_limits<int64_t>::max())
                           : static_cast<uint64_t>(std::numeric_limits<int64_t>::min());
            } else {
              init_i64 = agg_kinds[a] == TUPLE_GB_MIN
                           ? double_bits(std::numeric_limits<double>::infinity())
                           : double_bits(-std::numeric_limits<double>::infinity());
            }
            std::vector<uint64_t> init64(table_size, init_i64);
            state_i64_bufs[a].copy_from_host(init64.data(),
                                             init64.size() * sizeof(uint64_t),
                                             _ctx.device(),
                                             _ctx.queue(),
                                             _ctx.command_pool());
            state_i64_initialized_from_host[a] = true;
          }
        }

        auto& disp = _ctx.dispatcher();
        disp.begin_batch();
        disp.fill_buffer(slot_state.buffer(), 0, table_size * sizeof(uint32_t), slot_state.offset());
        disp.fill_buffer(slot_hash.buffer(), 0, table_size * sizeof(uint64_t), slot_hash.offset());
        disp.fill_buffer(slot_row.buffer(), 0, table_size * sizeof(uint32_t), slot_row.offset());
        disp.fill_buffer(unique_count.buffer(), 0, sizeof(uint32_t), unique_count.offset());
        disp.fill_buffer(overflow_count.buffer(), 0, sizeof(uint32_t), overflow_count.offset());
        for (size_t a = 0; a < aggregates.size(); a++) {
          uint32_t init_i32 = 0;
          if (value_types[a] == rasterdf::type_id::FLOAT32 &&
              (agg_kinds[a] == TUPLE_GB_MIN || agg_kinds[a] == TUPLE_GB_MAX)) {
            init_i32 = agg_kinds[a] == TUPLE_GB_MIN
                         ? float_bits(std::numeric_limits<float>::infinity())
                         : float_bits(-std::numeric_limits<float>::infinity());
          } else if (value_types[a] == rasterdf::type_id::INT32) {
            if (agg_kinds[a] == TUPLE_GB_MIN) {
              init_i32 = 0x7FFFFFFFu;
            } else if (agg_kinds[a] == TUPLE_GB_MAX) {
              init_i32 = 0x80000000u;
            }
          }
          disp.fill_buffer(state_i32_bufs[a].buffer(),
                           init_i32,
                           table_size * sizeof(int32_t),
                           state_i32_bufs[a].offset());
          if (!state_i64_initialized_from_host[a]) {
            disp.fill_buffer(state_i64_bufs[a].buffer(),
                             0,
                             table_size * sizeof(int64_t),
                             state_i64_bufs[a].offset());
          }
          disp.fill_buffer(aux_i64_bufs[a].buffer(),
                           0,
                           table_size * sizeof(int64_t),
                           aux_i64_bufs[a].offset());
        }
        disp.batch_barrier_fill_to_compute();

        tuple_groupby_build_pc build_pc{};
        build_pc.key_descs_ptr = key_desc_buf.data();
        build_pc.agg_descs_ptr = agg_desc_buf.data();
        build_pc.slot_state_ptr = slot_state.data();
        build_pc.slot_hash_ptr = slot_hash.data();
        build_pc.slot_row_ptr = slot_row.data();
        build_pc.unique_count_ptr = unique_count.data();
        build_pc.overflow_count_ptr = overflow_count.data();
        build_pc.numRows = n;
        build_pc.numKeys = static_cast<uint32_t>(num_group_cols);
        build_pc.numAggs = static_cast<uint32_t>(aggregates.size());
        build_pc.tableSize = table_size;
        disp.dispatch_tuple_groupby_build(build_pc, (n + 255) / 256);
        disp.end_batch();

        uint32_t num_unique_groups = 0;
        uint32_t overflow = 0;
        unique_count.copy_to_host(&num_unique_groups,
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
            "RasterDB GPU: tuple GROUP BY hash table overflow (%u rows, table=%u)",
            n,
            table_size);
        }

        rasterdf::device_buffer write_idx(_ctx.workspace_mr(), sizeof(uint32_t), usage);
        disp.begin_batch();
        disp.fill_buffer(write_idx.buffer(), 0, sizeof(uint32_t), write_idx.offset());
        disp.batch_barrier_fill_to_compute();
        tuple_groupby_extract_pc extract_pc{};
        extract_pc.key_descs_ptr = key_desc_buf.data();
        extract_pc.agg_descs_ptr = agg_desc_buf.data();
        extract_pc.slot_state_ptr = slot_state.data();
        extract_pc.slot_row_ptr = slot_row.data();
        extract_pc.write_idx_ptr = write_idx.data();
        extract_pc.numKeys = static_cast<uint32_t>(num_group_cols);
        extract_pc.numAggs = static_cast<uint32_t>(aggregates.size());
        extract_pc.tableSize = table_size;
        disp.dispatch_tuple_groupby_extract(extract_pc, (table_size + 255) / 256);
        disp.end_batch();

        auto ng = static_cast<rasterdf::size_type>(num_unique_groups);
        for (size_t k = 0; k < num_group_cols; k++) {
          output.columns[k].type = input.col(group_col_indices[k]).type;
          output.columns[k].num_rows = ng;
          output.columns[k].data = std::move(out_key_bufs[k]);
        }
        for (size_t a = 0; a < aggregates.size(); a++) {
          size_t out_col_idx = num_group_cols + a;
          auto output_type = tuple_groupby_output_type(agg_kinds[a], value_types[a]);
          output.columns[out_col_idx].type = {output_type};
          output.columns[out_col_idx].num_rows = ng;
          if (output_type == rasterdf::type_id::INT64 ||
              output_type == rasterdf::type_id::FLOAT64) {
            output.columns[out_col_idx].type = {rasterdf::type_id::INT64};
            output.columns[out_col_idx].data = std::move(out_i64_bufs[a]);
            output.columns[out_col_idx].type = {output_type};
          } else if (output_type == rasterdf::type_id::FLOAT32) {
            output.columns[out_col_idx].data = std::move(out_f32_bufs[a]);
          } else {
            output.columns[out_col_idx].data = std::move(out_i32_bufs[a]);
          }
        }
        output.set_num_rows(ng);
        RASTERDB_LOG_DEBUG("[Tuple GB] rows={} keys={} aggs={} groups={} table={}",
                           n,
                           num_group_cols,
                           aggregates.size(),
                           num_unique_groups,
                           table_size);
        return;
      }

      RASTERDB_LOG_DEBUG("[Tuple GB] unsupported shape; falling back to existing groupby path");
    }
  }

  if (num_group_cols > 3) {
    throw duckdb::NotImplementedException("RasterDB GPU: GROUP BY supports 1-3 columns for fallback path, got %zu",
                                          num_group_cols);
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
    (num_group_cols == 1 && input.col(group_col_indices[0]).type.id == rasterdf::type_id::INT32);
  uint32_t compute_groupby_table_hint = 0;

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
    if (!USE_SIMPLE_GFX_AGGR) {
      stage_timer tc("    groupby_single_key_hint");
      auto& gcol = input.col(group_col_indices[0]);
      auto col_view = gcol.view();
      rasterdf::reduce_aggregation max_agg(rasterdf::aggregation_kind::MAX);
      auto max_s = rasterdf::reduce(col_view,
                                    max_agg,
                                    rasterdf::data_type{rasterdf::type_id::INT32},
                                    _ctx.vk_context(),
                                    _ctx.dispatcher(),
                                    _ctx.workspace_mr());
      int32_t max_key = max_s->as<int32_t>();
      if (max_key >= 0 &&
          static_cast<uint64_t>(max_key) < static_cast<uint64_t>(UINT32_MAX - 1)) {
        compute_groupby_table_hint = static_cast<uint32_t>(max_key) + 1;
      }
      RASTERDB_LOG_DEBUG("[RDB_DEBUG] GROUP BY single INT32 hint: max_key={} hint={}",
                         max_key,
                         compute_groupby_table_hint);
    }
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
    if (!composite_is_int64 && max_composite_estimate >= 0 &&
        static_cast<uint64_t>(max_composite_estimate) < static_cast<uint64_t>(UINT32_MAX - 1)) {
      compute_groupby_table_hint = static_cast<uint32_t>(max_composite_estimate) + 1;
    }
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
      std::vector<std::vector<uint8_t>> h_group_cols(num_group_cols);
      for (size_t g = 0; g < num_group_cols; g++) {
        size_t elem_size = rdf_type_size(input.col(group_col_indices[g]).type.id);
        h_group_cols[g].resize(static_cast<size_t>(n_rows) * elem_size);
        download_column(_ctx,
                        input.col(group_col_indices[g]),
                        h_group_cols[g].data(),
                        h_group_cols[g].size());
      }

      // Helper: convert raw column bytes to uint64 for composite key construction.
      // For FLOAT32 columns, reinterpret as uint32 to get a non-negative value.
      auto to_unsigned = [&](size_t col_idx, rasterdf::size_type row) -> int64_t {
        auto type_id = input.col(group_col_indices[col_idx]).type.id;
        const uint8_t* ptr = h_group_cols[col_idx].data() + static_cast<size_t>(row) * rdf_type_size(type_id);
        if (type_id == rasterdf::type_id::FLOAT32) {
          uint32_t raw;
          std::memcpy(&raw, ptr, sizeof(uint32_t));
          return static_cast<int64_t>(raw);
        }
        if (type_id == rasterdf::type_id::INT64) {
          int64_t raw;
          std::memcpy(&raw, ptr, sizeof(int64_t));
          return raw;
        }
        int32_t raw;
        std::memcpy(&raw, ptr, sizeof(int32_t));
        return static_cast<int64_t>(raw);
      };

      // Compute INT64 composite keys (simple vectorizable loop, no hash map)
      std::vector<int64_t> h_composite(n_rows);
      if (num_group_cols == 1) {
        for (rasterdf::size_type r = 0; r < n_rows; r++) {
          h_composite[r] = to_unsigned(0, r);
        }
      } else if (num_group_cols == 2) {
        for (rasterdf::size_type r = 0; r < n_rows; r++) {
          h_composite[r] = to_unsigned(0, r) * decompose_base1 +
                           to_unsigned(1, r);
        }
      } else {
        for (rasterdf::size_type r = 0; r < n_rows; r++) {
          h_composite[r] = (to_unsigned(0, r) * decompose_base1 +
                            to_unsigned(1, r)) *
                             decompose_base2 +
                           to_unsigned(2, r);
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

  if (aggregates.empty()) {
    rasterdf::data_type key_type = key_col_ptr->type;
    if (single_col_string) {
      throw duckdb::NotImplementedException(
        "RasterDB GPU: zero-aggregate GROUP BY on STRING key not supported");
    }

    if (key_type.id == rasterdf::type_id::INT32) {
      if (num_group_cols == 1 && surrogate_id_to_composite.empty()) {
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

      std::vector<int32_t> h_keys(static_cast<size_t>(n_rows));
      download_column(_ctx, *key_col_ptr, h_keys.data(), h_keys.size() * sizeof(int32_t));
      std::sort(h_keys.begin(), h_keys.end());
      h_keys.erase(std::unique(h_keys.begin(), h_keys.end()), h_keys.end());

      auto ng = static_cast<rasterdf::size_type>(h_keys.size());
      if (num_group_cols == 1 && surrogate_id_to_composite.empty()) {
        output.columns[0] = allocate_column(_ctx, key_type, ng);
        output.columns[0].data.copy_from_host(h_keys.data(),
                                              h_keys.size() * sizeof(int32_t),
                                              _ctx.device(),
                                              _ctx.queue(),
                                              _ctx.command_pool());
      } else {
        std::vector<int64_t> sorted_composite_i64(ng);
        if (!surrogate_id_to_composite.empty()) {
          for (rasterdf::size_type j = 0; j < ng; j++) {
            int32_t id = h_keys[static_cast<size_t>(j)];
            if (id < 0 || static_cast<size_t>(id) >= surrogate_id_to_composite.size()) {
              throw duckdb::NotImplementedException(
                "RasterDB GPU: invalid surrogate GROUP BY key id %d", id);
            }
            sorted_composite_i64[static_cast<size_t>(j)] =
              surrogate_id_to_composite[static_cast<size_t>(id)];
          }
        } else {
          for (rasterdf::size_type j = 0; j < ng; j++) {
            sorted_composite_i64[static_cast<size_t>(j)] =
              static_cast<int64_t>(h_keys[static_cast<size_t>(j)]);
          }
        }

        auto get_col_type = [&](size_t g) -> rasterdf::type_id {
          if (g < _group_col_types.size()) return _group_col_types[g];
          return rasterdf::type_id::INT32;
        };

        if (num_group_cols == 2) {
          std::vector<int32_t> col0(ng), col1(ng);
          for (rasterdf::size_type j = 0; j < ng; j++) {
            col1[static_cast<size_t>(j)] =
              static_cast<int32_t>(sorted_composite_i64[static_cast<size_t>(j)] %
                                   decompose_base1);
            col0[static_cast<size_t>(j)] =
              static_cast<int32_t>(sorted_composite_i64[static_cast<size_t>(j)] /
                                   decompose_base1);
          }
          output.columns[0] = allocate_column(_ctx, {get_col_type(0)}, ng);
          output.columns[0].data.copy_from_host(
            col0.data(), col0.size() * sizeof(int32_t), _ctx.device(), _ctx.queue(), _ctx.command_pool());
          output.columns[1] = allocate_column(_ctx, {get_col_type(1)}, ng);
          output.columns[1].data.copy_from_host(
            col1.data(), col1.size() * sizeof(int32_t), _ctx.device(), _ctx.queue(), _ctx.command_pool());
        } else {
          std::vector<int32_t> col0(ng), col1(ng), col2(ng);
          for (rasterdf::size_type j = 0; j < ng; j++) {
            int64_t c = sorted_composite_i64[static_cast<size_t>(j)];
            col2[static_cast<size_t>(j)] = static_cast<int32_t>(c % decompose_base2);
            c /= decompose_base2;
            col1[static_cast<size_t>(j)] = static_cast<int32_t>(c % decompose_base1);
            col0[static_cast<size_t>(j)] = static_cast<int32_t>(c / decompose_base1);
          }
          output.columns[0] = allocate_column(_ctx, {get_col_type(0)}, ng);
          output.columns[0].data.copy_from_host(
            col0.data(), col0.size() * sizeof(int32_t), _ctx.device(), _ctx.queue(), _ctx.command_pool());
          output.columns[1] = allocate_column(_ctx, {get_col_type(1)}, ng);
          output.columns[1].data.copy_from_host(
            col1.data(), col1.size() * sizeof(int32_t), _ctx.device(), _ctx.queue(), _ctx.command_pool());
          output.columns[2] = allocate_column(_ctx, {get_col_type(2)}, ng);
          output.columns[2].data.copy_from_host(
            col2.data(), col2.size() * sizeof(int32_t), _ctx.device(), _ctx.queue(), _ctx.command_pool());
        }
      }
      output.set_num_rows(ng);
      RASTERDB_LOG_DEBUG("GROUP BY distinct result: {} groups, {} output cols",
                         ng,
                         output.columns.size());
      return;
    }

    if (key_type.id == rasterdf::type_id::INT64) {
      std::vector<int64_t> h_keys(static_cast<size_t>(n_rows));
      download_column(_ctx, *key_col_ptr, h_keys.data(), h_keys.size() * sizeof(int64_t));
      std::sort(h_keys.begin(), h_keys.end());
      h_keys.erase(std::unique(h_keys.begin(), h_keys.end()), h_keys.end());
      auto ng = static_cast<rasterdf::size_type>(h_keys.size());
      output.columns[0] = allocate_column(_ctx, key_type, ng);
      output.columns[0].data.copy_from_host(h_keys.data(),
                                            h_keys.size() * sizeof(int64_t),
                                            _ctx.device(),
                                            _ctx.queue(),
                                            _ctx.command_pool());
      output.set_num_rows(ng);
      RASTERDB_LOG_DEBUG("GROUP BY distinct result: {} groups, {} output cols",
                         ng,
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
      if (aggregates.size() == 1 && num_group_cols == 1 && surrogate_id_to_composite.empty() &&
          !single_col_string) {
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
          if (single_col_string) {
            // Replace INT32 hash keys with original STRING keys via gather
            // Find first occurrence of each unique hash in the original hash array
            auto& str_col = input.col(group_col_indices[0]);
            auto& hash_keys = sorted_key_col; // these are the unique hash INT32 keys

            rasterdf::device_buffer first_idx_buf(
                _ctx.workspace_mr(), out_num_groups * sizeof(int32_t),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
            // Initialize to 0xFFFFFFFF so atomicMin in shader works correctly
            _ctx.dispatcher().fill_buffer(first_idx_buf.buffer(), 0xFFFFFFFFu,
                         out_num_groups * sizeof(int32_t), first_idx_buf.offset());

            find_first_index_pc fpc{};
            fpc.all_keys_ptr = string_hash_key.address(); // full hash array
            fpc.unique_keys_ptr = hash_keys.address();    // unique hash keys
            fpc.first_idx_ptr = first_idx_buf.data();
            fpc.numElements = static_cast<uint32_t>(n_rows);
            fpc.numUnique = out_num_groups;
            _ctx.dispatcher().dispatch_find_first_index(fpc, (n_rows + 255) / 256);

            // Gather original strings using first_idx
            string_lengths_pc lpc{};
            lpc.offsets_ptr = str_col.str_offsets.data();
            lpc.indices_ptr = first_idx_buf.data();
            lpc.num_indices = out_num_groups;

            rasterdf::device_buffer out_offsets(
                _ctx.workspace_mr(), (out_num_groups + 1) * sizeof(int32_t),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
            // Write lengths into out_offsets[0..N-1]
            lpc.output_ptr = out_offsets.data();
            _ctx.dispatcher().dispatch_string_lengths(lpc);
            // Zero element N, then exclusive prefix scan on N+1 elements
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

          std::vector<std::vector<uint8_t>> decomposed_cols(num_group_cols);
          for (size_t g = 0; g < num_group_cols; g++) {
            decomposed_cols[g].resize(static_cast<size_t>(out_num_groups) * rdf_type_size(get_col_type(g)));
          }
          for (uint32_t j = 0; j < out_num_groups; j++) {
            int64_t c = sorted_composite_i64[j];
            std::vector<int64_t> vals(num_group_cols);
            if (num_group_cols == 1) {
              vals[0] = c;
            } else if (num_group_cols == 2) {
              vals[1] = c % decompose_base1;
              vals[0] = c / decompose_base1;
            } else {
              vals[2] = c % decompose_base2;
              c /= decompose_base2;
              vals[1] = c % decompose_base1;
              vals[0] = c / decompose_base1;
            }
            for (size_t g = 0; g < num_group_cols; g++) {
              auto tid = get_col_type(g);
              uint8_t* dst = decomposed_cols[g].data() + static_cast<size_t>(j) * rdf_type_size(tid);
              if (tid == rasterdf::type_id::INT64) {
                int64_t v = vals[g];
                std::memcpy(dst, &v, sizeof(int64_t));
              } else {
                int32_t v = static_cast<int32_t>(vals[g]);
                std::memcpy(dst, &v, sizeof(int32_t));
              }
            }
          }
          for (size_t g = 0; g < num_group_cols; g++) {
            auto tid = get_col_type(g);
            output.columns[g] = allocate_column(_ctx, {tid}, out_num_groups);
            output.columns[g].data.copy_from_host(decomposed_cols[g].data(),
                                                  decomposed_cols[g].size(),
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
    // ── Compute Shader Groupby (rasterdf::groupby) — FUSED single-pass ──
    RASTERDB_LOG_DEBUG("     [COMPUTE] Using compute shader groupby (fused)");

    struct compute_output_plan {
      bool derived_avg = false;
      size_t request_idx = 0;
      size_t sum_request_idx = 0;
      size_t count_request_idx = 0;
      rasterdf::type_id derived_avg_type = rasterdf::type_id::FLOAT32;
    };

    std::vector<gpu_column> val_temps;
    val_temps.reserve(aggregates.size() * 2 + 1);
    std::vector<rasterdf::aggregation_request> requests;
    requests.reserve(aggregates.size() * 2 + 1);
    std::vector<compute_output_plan> output_plans;
    output_plans.reserve(aggregates.size());
    std::unordered_map<std::string, size_t> sum_request_by_expr;
    std::unordered_map<std::string, rasterdf::type_id> sum_value_type_by_expr;
    size_t count_request_idx = static_cast<size_t>(-1);

    auto add_count_request = [&]() -> size_t {
      if (count_request_idx != static_cast<size_t>(-1)) {
        return count_request_idx;
      }
      rasterdf::aggregation_request req;
      req.values = key_col_ptr->view();
      req.aggregations.push_back(
        std::make_unique<rasterdf::groupby_aggregation>(rasterdf::aggregation_kind::COUNT_ALL));
      count_request_idx = requests.size();
      requests.push_back(std::move(req));
      return count_request_idx;
    };

    auto add_sum_request = [&](duckdb::Expression& child) -> size_t {
      std::string expr_key = child.ToString();
      auto it = sum_request_by_expr.find(expr_key);
      if (it != sum_request_by_expr.end()) {
        return it->second;
      }
      val_temps.push_back(evaluate_expression(input, child));
      rasterdf::aggregation_request req;
      req.values = val_temps.back().view();
      req.aggregations.push_back(
        std::make_unique<rasterdf::groupby_aggregation>(rasterdf::aggregation_kind::SUM));
      size_t request_idx = requests.size();
      requests.push_back(std::move(req));
      sum_request_by_expr[expr_key] = request_idx;
      sum_value_type_by_expr[expr_key] = val_temps.back().type.id;
      return request_idx;
    };

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

      if (fname == "sum" || fname == "sum_no_overflow") {
        size_t request_idx = add_sum_request(*expr.children[0]);
        output_plans.push_back({false, request_idx, 0, 0, rasterdf::type_id::FLOAT32});
        continue;
      } else if (fname == "avg" || fname == "mean") {
        size_t sum_request_idx = add_sum_request(*expr.children[0]);
        size_t avg_count_idx = add_count_request();
        std::string expr_key = expr.children[0]->ToString();
        auto value_type_id = sum_value_type_by_expr[expr_key];
        auto avg_type = value_type_id == rasterdf::type_id::FLOAT32
                          ? rasterdf::type_id::FLOAT64
                          : rasterdf::type_id::FLOAT32;
        output_plans.push_back({true, 0, sum_request_idx, avg_count_idx, avg_type});
        continue;
      }

      rasterdf::aggregation_kind kind;
      if (fname == "min") {
        kind = rasterdf::aggregation_kind::MIN;
      } else if (fname == "max") {
        kind = rasterdf::aggregation_kind::MAX;
      } else if (fname == "count" || fname == "count_star") {
        size_t request_idx = add_count_request();
        output_plans.push_back({false, request_idx, 0, 0, rasterdf::type_id::FLOAT32});
        continue;
      } else {
        throw duckdb::NotImplementedException("RasterDB GPU: unsupported grouped aggregate '%s'",
                                              fname.c_str());
      }

      rasterdf::aggregation_request req;
      if (is_count_star) {
        req.values = key_col_ptr->view();
      } else {
        val_temps.push_back(evaluate_expression(input, *expr.children[0]));
        req.values   = val_temps.back().view();
      }
      req.aggregations.push_back(std::make_unique<rasterdf::groupby_aggregation>(kind));
      output_plans.push_back({false, requests.size(), 0, 0, rasterdf::type_id::FLOAT32});
      requests.push_back(std::move(req));
    }

    size_t request_count = requests.size();

    // 2. Single fused groupby call — builds hash table ONCE, scans data ONCE
    rasterdf::groupby gb(keys_tv,
                         _ctx.vk_context(),
                         _ctx.dispatcher(),
                         _ctx.workspace_mr(),
                         compute_groupby_table_hint);
    auto agg_result = gb.aggregate(std::move(requests));

    // 3. Extract keys (produced once) and validate
    auto result_key_cols = agg_result.keys->extract();
    if (result_key_cols.empty()) {
      throw duckdb::NotImplementedException(
        "RasterDB GPU: fused grouped aggregate produced empty keys");
    }
    auto& key_col_rdf    = *result_key_cols[0];
    auto ng              = key_col_rdf.size();
    size_t key_elem_size = rasterdf::size_of(key_col_rdf.type());
    bool keys_are_int64  = (key_col_rdf.type().id == rasterdf::type_id::INT64);

    // 4. Download keys and build sort permutation (ONCE for all aggregates)
    std::vector<uint8_t> h_keys_raw(ng * key_elem_size);
    key_col_rdf.device_data().copy_to_host(
      h_keys_raw.data(), ng * key_elem_size, 0, _ctx.device(), _ctx.queue(), _ctx.command_pool());

    std::vector<size_t> perm(ng);
    std::iota(perm.begin(), perm.end(), 0);
    if (keys_are_int64) {
      auto* kp = reinterpret_cast<const int64_t*>(h_keys_raw.data());
      std::sort(perm.begin(), perm.end(), [kp](size_t a, size_t b) { return kp[a] < kp[b]; });
    } else {
      auto* kp = reinterpret_cast<const int32_t*>(h_keys_raw.data());
      std::sort(perm.begin(), perm.end(), [kp](size_t a, size_t b) { return kp[a] < kp[b]; });
    }

    // 5. Sort keys and store in output (ONCE)
    std::vector<uint8_t> sorted_keys_raw(ng * key_elem_size);
    for (size_t j = 0; j < ng; j++) {
      std::memcpy(sorted_keys_raw.data() + j * key_elem_size,
                  h_keys_raw.data() + perm[j] * key_elem_size,
                  key_elem_size);
    }

    num_groups_result = ng;
    if (num_group_cols == 1) {
      auto sorted_key_col = allocate_column(_ctx, key_col_rdf.type(), ng);
      sorted_key_col.data.copy_from_host(sorted_keys_raw.data(),
                                         ng * key_elem_size,
                                         _ctx.device(),
                                         _ctx.queue(),
                                         _ctx.command_pool());
      if (single_col_string) {
        auto& str_col = input.col(group_col_indices[0]);
        rasterdf::device_buffer first_idx_buf(
            _ctx.workspace_mr(), ng * sizeof(int32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        _ctx.dispatcher().fill_buffer(first_idx_buf.buffer(), 0xFFFFFFFFu,
                     ng * sizeof(int32_t), first_idx_buf.offset());
        find_first_index_pc fpc{};
        fpc.all_keys_ptr = string_hash_key.address();
        fpc.unique_keys_ptr = sorted_key_col.address();
        fpc.first_idx_ptr = first_idx_buf.data();
        fpc.numElements = static_cast<uint32_t>(n_rows);
        fpc.numUnique = static_cast<uint32_t>(ng);
        _ctx.dispatcher().dispatch_find_first_index(fpc, (n_rows + 255) / 256);

        rasterdf::device_buffer out_offsets(
            _ctx.workspace_mr(), (ng + 1) * sizeof(int32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        string_lengths_pc lpc{};
        lpc.offsets_ptr = str_col.str_offsets.data();
        lpc.indices_ptr = first_idx_buf.data();
        lpc.num_indices = static_cast<uint32_t>(ng);
        lpc.output_ptr = out_offsets.data();
        _ctx.dispatcher().dispatch_string_lengths(lpc);
        _ctx.dispatcher().fill_buffer(out_offsets.buffer(), 0u, sizeof(int32_t),
                     out_offsets.offset() + ng * sizeof(int32_t));
        uint32_t scan_elems = static_cast<uint32_t>(ng) + 1;
        uint32_t scan_ngroups = (scan_elems + 255) / 256;
        rasterdf::device_buffer scan_bsums(_ctx.workspace_mr(), scan_ngroups * sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        rasterdf::device_buffer scan_total(_ctx.workspace_mr(), sizeof(uint32_t),
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
        cpc.num_indices = static_cast<uint32_t>(ng);
        _ctx.dispatcher().dispatch_string_copy(cpc);
        output.columns[0].type = rasterdf::data_type{rasterdf::type_id::STRING};
        output.columns[0].num_rows = static_cast<rasterdf::size_type>(ng);
        output.columns[0].str_offsets = std::move(out_offsets);
        output.columns[0].str_chars = std::move(out_chars);
        output.columns[0].str_total_chars = total_out_chars;
      } else {
        output.columns[0] = std::move(sorted_key_col);
      }
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

    std::vector<std::vector<uint8_t>> sorted_request_vals(request_count);
    std::vector<rasterdf::data_type> request_types;
    std::vector<size_t> request_elem_sizes;
    request_types.reserve(request_count);
    request_elem_sizes.reserve(request_count);

    for (size_t request_idx = 0; request_idx < request_count; request_idx++) {
      if (request_idx >= agg_result.results.size() || !agg_result.results[request_idx]) {
        throw duckdb::NotImplementedException(
          "RasterDB GPU: fused grouped aggregate produced empty result for request %zu", request_idx);
      }
      auto& val_col_rdf   = *agg_result.results[request_idx];
      size_t val_elem_size = rasterdf::size_of(val_col_rdf.type());
      request_types.push_back(val_col_rdf.type());
      request_elem_sizes.push_back(val_elem_size);

      std::vector<uint8_t> h_vals(ng * val_elem_size);
      val_col_rdf.device_data().copy_to_host(
        h_vals.data(), ng * val_elem_size, 0, _ctx.device(), _ctx.queue(), _ctx.command_pool());

      sorted_request_vals[request_idx].resize(ng * val_elem_size);
      for (size_t j = 0; j < ng; j++) {
        std::memcpy(sorted_request_vals[request_idx].data() + j * val_elem_size,
                    h_vals.data() + perm[j] * val_elem_size,
                    val_elem_size);
      }
    }

    auto read_sorted_numeric = [&](size_t request_idx, size_t row) -> double {
      const uint8_t* ptr = sorted_request_vals[request_idx].data() + row * request_elem_sizes[request_idx];
      switch (request_types[request_idx].id) {
      case rasterdf::type_id::INT32:
        return static_cast<double>(*reinterpret_cast<const int32_t*>(ptr));
      case rasterdf::type_id::INT64:
        return static_cast<double>(*reinterpret_cast<const int64_t*>(ptr));
      case rasterdf::type_id::FLOAT32:
        return static_cast<double>(*reinterpret_cast<const float*>(ptr));
      case rasterdf::type_id::FLOAT64:
        return *reinterpret_cast<const double*>(ptr);
      default:
        throw duckdb::NotImplementedException(
          "RasterDB GPU: cannot derive AVG from type_id %d",
          static_cast<int>(request_types[request_idx].id));
      }
    };

    for (duckdb::idx_t i = 0; i < aggregates.size(); i++) {
      auto& plan = output_plans[i];
      size_t out_col_idx  = num_group_cols + i;

      if (!plan.derived_avg) {
        auto sorted_val_col = allocate_column(_ctx, request_types[plan.request_idx], ng);
        sorted_val_col.data.copy_from_host(sorted_request_vals[plan.request_idx].data(),
                                           ng * request_elem_sizes[plan.request_idx],
                                           _ctx.device(),
                                           _ctx.queue(),
                                           _ctx.command_pool());
        output.columns[out_col_idx] = std::move(sorted_val_col);
        continue;
      }

      if (plan.derived_avg_type == rasterdf::type_id::FLOAT64) {
        std::vector<double> avg_vals(ng);
        for (size_t j = 0; j < ng; j++) {
          double sum = read_sorted_numeric(plan.sum_request_idx, j);
          double cnt = read_sorted_numeric(plan.count_request_idx, j);
          avg_vals[j] = cnt != 0.0 ? sum / cnt : 0.0;
        }
        auto sorted_val_col = allocate_column(_ctx, {rasterdf::type_id::FLOAT64}, ng);
        sorted_val_col.data.copy_from_host(
          avg_vals.data(), ng * sizeof(double), _ctx.device(), _ctx.queue(), _ctx.command_pool());
        output.columns[out_col_idx] = std::move(sorted_val_col);
      } else {
        std::vector<float> avg_vals(ng);
        for (size_t j = 0; j < ng; j++) {
          double sum = read_sorted_numeric(plan.sum_request_idx, j);
          double cnt = read_sorted_numeric(plan.count_request_idx, j);
          avg_vals[j] = cnt != 0.0 ? static_cast<float>(sum / cnt) : 0.0f;
        }
        auto sorted_val_col = allocate_column(_ctx, {rasterdf::type_id::FLOAT32}, ng);
        sorted_val_col.data.copy_from_host(
          avg_vals.data(), ng * sizeof(float), _ctx.device(), _ctx.queue(), _ctx.command_pool());
        output.columns[out_col_idx] = std::move(sorted_val_col);
      }
    }
  }
  output.set_num_rows(num_groups_result);
  RASTERDB_LOG_DEBUG(
    "GROUP BY result: {} groups, {} output cols", num_groups_result, output.columns.size());
}

}  // namespace gpu
}  // namespace rasterdb
