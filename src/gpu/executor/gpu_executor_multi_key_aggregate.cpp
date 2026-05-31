/*
 * Copyright 2026, RasterDB Contributors.
 * Multi-key GROUP BY aggregate executor.
 */

#include "gpu/gpu_executor_internal.hpp"

#include <algorithm>
#include <cstdlib>

namespace rasterdb {
namespace gpu {

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

struct dense_key_range_info {
  bool supported = false;
  uint64_t group_count = 0;
  std::vector<int32_t> mins;
  std::vector<uint32_t> ranges;
  std::vector<uint32_t> strides;
};

bool tuple_groupby_fixed_width_supported(rasterdf::type_id id)
{
  return id == rasterdf::type_id::INT32 || id == rasterdf::type_id::INT64 ||
         id == rasterdf::type_id::FLOAT32 || id == rasterdf::type_id::FLOAT64;
}

uint64_t double_bits(double value)
{
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

uint32_t float_bits(float value)
{
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

rasterdf::type_id tuple_groupby_output_type(uint32_t kind, rasterdf::type_id value_type)
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

bool dense_groupby_agg_supported(uint32_t kind)
{
  return kind == TUPLE_GB_SUM || kind == TUPLE_GB_COUNT || kind == TUPLE_GB_MEAN;
}

rasterdf::type_id dense_groupby_output_type(uint32_t kind, rasterdf::type_id value_type)
{
  if ((kind == TUPLE_GB_SUM || kind == TUPLE_GB_MEAN) &&
      value_type == rasterdf::type_id::FLOAT32) {
    return rasterdf::type_id::FLOAT64;
  }
  return tuple_groupby_output_type(kind, value_type);
}

}  // namespace

bool gpu_executor::try_execute_multi_key_aggregate(
  const gpu_table& input,
  const duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& groups,
  const duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& aggregates,
  const duckdb::vector<duckdb::LogicalType>& result_types,
  const std::vector<duckdb::idx_t>& group_col_indices,
  int tuple_key_policy,
  gpu_table& output)
{
  (void)groups;
  (void)result_types;
  const size_t num_group_cols = group_col_indices.size();
  if (tuple_key_policy == 0) {
    return false;
  }
  if (tuple_key_policy != -1 && tuple_key_policy != 1) {
    throw duckdb::InvalidInputException(
      "RasterDB GPU: USE_SIMPLE_TUPLE_KEY_AGGR must be -1, 0, or 1");
  }
  const bool force_tuple_key = tuple_key_policy == 1;
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

bool single_float_key =
  num_group_cols == 1 &&
  (input.col(group_col_indices[0]).type.id == rasterdf::type_id::FLOAT32 ||
   input.col(group_col_indices[0]).type.id == rasterdf::type_id::FLOAT64);
bool tuple_candidate = force_tuple_key ? num_group_cols >= 1
                                       : (num_group_cols >= 2 || single_float_key);
for (auto idx : group_col_indices) {
  tuple_candidate = tuple_candidate && tuple_groupby_fixed_width_supported(input.col(idx).type.id);
}

if (!tuple_candidate) {
  if (force_tuple_key) {
    throw duckdb::NotImplementedException(
      "RasterDB GPU: forced tuple-key GROUP BY only supports fixed-width numeric keys");
  }
  return false;
}
if (aggregates.empty()) {
  if (force_tuple_key) {
    throw duckdb::NotImplementedException(
      "RasterDB GPU: forced tuple-key GROUP BY does not yet support distinct-only grouping");
  }
  return false;
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
      return true;
    }

        if (tuple_low_cardinality_contention && !force_tuple_key) {
          RASTERDB_LOG_DEBUG(
            "[Tuple GB] low-cardinality shape not supported by dense path; falling back");
          tuple_candidate = false;
        } else if (tuple_low_cardinality_contention) {
          RASTERDB_LOG_DEBUG(
            "[Tuple GB] forced tuple-key path: dense-ineligible low-cardinality shape will use tuple hash");
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
    return true;
  }

      RASTERDB_LOG_DEBUG("[Tuple GB] unsupported shape; falling back to existing groupby path");
    }

  if (force_tuple_key) {
    throw duckdb::NotImplementedException(
      "RasterDB GPU: forced tuple-key GROUP BY shape is not supported");
  }

  return false;
}

}  // namespace gpu
}  // namespace rasterdb
