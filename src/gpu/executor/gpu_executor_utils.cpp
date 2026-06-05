/*
 * Copyright 2026, RasterDB Contributors.
 * Shared executor implementation helpers.
 */

#include "gpu/gpu_executor_internal.hpp"

namespace rasterdb {
namespace gpu {

void debug_print_plan(duckdb::LogicalOperator& op, int depth) {
  std::string indent(depth * 2, ' ');
  RASTERDB_LOG_DEBUG("[RDB_PLAN] {}{} (types={}, children={})",
                     indent, duckdb::LogicalOperatorToString(op.type),
                     op.types.size(), op.children.size());
  for (auto& child : op.children) {
    debug_print_plan(*child, depth + 1);
  }
}

void append_logical_plan(duckdb::LogicalOperator& op, std::string& out, int depth) {
  out.append(static_cast<size_t>(depth * 2), ' ');
  out += duckdb::LogicalOperatorToString(op.type);
  out += " types=";
  out += std::to_string(op.types.size());
  out += " children=";
  out += std::to_string(op.children.size());
  out += " est_card=";
  out += std::to_string(static_cast<uint64_t>(op.estimated_cardinality));
  out += "\n";
  for (auto& child : op.children) {
    append_logical_plan(*child, out, depth + 1);
  }
}

bool gpu_executor::has_lazy_columns(const gpu_table& input) const {
  for (size_t c = 0; c < input.num_columns(); c++) {
    if (input.col(c).is_lazy()) {
      return true;
    }
  }
  return false;
}

gpu_column gpu_executor::alias_lazy_column(const gpu_column& input) {
  if (!input.is_lazy()) {
    throw duckdb::InternalException("RasterDB GPU: alias_lazy_column called for materialized column");
  }
  gpu_column out;
  out.type = input.type;
  out.num_rows = input.num_rows;
  out.lazy_base_table = input.lazy_base_table;
  out.lazy_base_col_idx = input.lazy_base_col_idx;
  out.lazy_row_indices = input.lazy_row_indices;
  out.has_i32_minmax = input.has_i32_minmax;
  out.i32_min = input.i32_min;
  out.i32_max = input.i32_max;
  return out;
}

gpu_column gpu_executor::materialize_column(const gpu_column& input) {
  if (!input.is_lazy()) {
    throw duckdb::InternalException("RasterDB GPU: materialize_column called for non-lazy column");
  }
  if (!input.lazy_base_table || !input.lazy_row_indices) {
    throw duckdb::InternalException("RasterDB GPU: invalid lazy column metadata");
  }

  const auto& base_col = input.lazy_base_table->col(input.lazy_base_col_idx);
  if (base_col.is_string()) {
    throw duckdb::NotImplementedException(
        "RasterDB GPU: lazy STRING materialization is not yet supported");
  }

  rasterdf::column_view idx_view(
      rasterdf::data_type{rasterdf::type_id::INT32}, input.num_rows,
      input.lazy_row_indices->view().data(), 0, 0, 0);

  auto base_materialized = base_col.is_lazy() ? materialize_column(base_col) : gpu_column{};
  const gpu_column& physical_base = base_col.is_lazy() ? base_materialized : base_col;

  auto gathered = rasterdf::gather(physical_base.view(), idx_view,
                                   _ctx.vk_context(), _ctx.dispatcher(),
                                   _ctx.workspace_mr());
  gpu_column out = gpu_column_from_rdf(std::move(*gathered));
  out.type = input.type;
  out.num_rows = input.num_rows;

  if (physical_base.has_validity) {
    const size_t validity_bytes =
        ((static_cast<size_t>(input.num_rows) + 31u) / 32u) * sizeof(uint32_t);
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    out.validity = rasterdf::device_buffer(
        _ctx.workspace_mr(), std::max<size_t>(validity_bytes, sizeof(uint32_t)), usage);
    out.has_validity = true;
    _ctx.dispatcher().fill_buffer(out.validity.buffer(), 0u, validity_bytes,
                                  out.validity.offset());
    gather_validity_pc vpc{};
    vpc.input_validity_addr = physical_base.validity.data();
    vpc.indices_addr = input.lazy_row_indices->address();
    vpc.output_validity_addr = out.validity.data();
    vpc.size = static_cast<uint32_t>(input.num_rows);
    _ctx.dispatcher().dispatch_gather_validity(vpc, div_ceil(vpc.size, WG_SIZE));
  }

  return out;
}

std::unique_ptr<gpu_table> gpu_executor::materialize_table(const gpu_table& input) {
  if (!has_lazy_columns(input)) {
    throw duckdb::InternalException("RasterDB GPU: materialize_table called without lazy columns");
  }

  auto result = std::make_unique<gpu_table>();
  result->duckdb_types = input.duckdb_types;
  result->columns.resize(input.num_columns());
  result->set_num_rows(input.num_rows());
  auto t0 = std::chrono::high_resolution_clock::now();
  size_t lazy_cols = 0;
  size_t logical_bytes = 0;

  for (size_t c = 0; c < input.num_columns(); c++) {
    const auto& src = input.col(c);
    if (src.is_lazy()) {
      result->columns[c] = materialize_column(src);
      lazy_cols++;
      logical_bytes += static_cast<size_t>(src.num_rows) * rdf_type_size(src.type.id);
    } else if (src.is_host_only) {
      result->columns[c].type = src.type;
      result->columns[c].num_rows = src.num_rows;
      result->columns[c].is_host_only = true;
      result->columns[c].host_data = src.host_data;
    } else if (src.is_string()) {
      result->columns[c].type = src.type;
      result->columns[c].num_rows = src.num_rows;
      result->columns[c].str_offsets = std::move(const_cast<gpu_column&>(src).str_offsets);
      result->columns[c].str_chars = std::move(const_cast<gpu_column&>(src).str_chars);
      result->columns[c].str_total_chars = src.str_total_chars;
    } else if (can_alias_fixed_width_column(src)) {
      result->columns[c] = alias_fixed_width_column(src);
    } else {
      result->columns[c] = allocate_column(_ctx, src.type, src.num_rows);
      size_t bytes = static_cast<size_t>(src.num_rows) * rdf_type_size(src.type.id);
      _ctx.dispatcher().copy_buffer(src.data.buffer(), result->columns[c].data.buffer(),
                                    bytes, src.data.offset(),
                                    result->columns[c].data.offset());
    }
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  RASTERDB_LOG_INFO("[RDB_LAZY_PROFILE] materialize_table_ms={:.2f} lazy_cols={} rows={} logical_mb={:.2f}",
                    ms, lazy_cols, input.num_rows(),
                    static_cast<double>(logical_bytes) / (1024.0 * 1024.0));
  return result;
}

std::unique_ptr<gpu_table> gpu_executor::materialize_output_table(std::unique_ptr<gpu_table> table) {
  if (!table || !has_lazy_columns(*table)) {
    return table;
  }
  return materialize_table(*table);
}

} // namespace gpu
} // namespace rasterdb
