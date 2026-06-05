/*
 * Copyright 2026, RasterDB Contributors.
 * Split from src/gpu/gpu_executor.cpp.
 */

#include "gpu/gpu_executor_internal.hpp"

namespace rasterdb {
namespace gpu {

// ============================================================================
// PROJECTION — select columns + evaluate arithmetic expressions
// ============================================================================

std::unique_ptr<gpu_table> gpu_executor::execute_projection(duckdb::LogicalProjection& op)
{
  RASTERDB_LOG_DEBUG("GPU execute_projection");
  D_ASSERT(op.children.size() == 1);
  auto input = execute_operator(*op.children[0]);

  stage_timer t("  projection");  // Timer starts AFTER child execution

  auto result = std::make_unique<gpu_table>();
  result->duckdb_types = op.types;
  result->columns.resize(op.expressions.size());

  auto copy_validity = [&](const gpu_column& src, gpu_column& dst) {
    if (!src.has_validity) return;
    if (src.is_string()) {
      throw duckdb::NotImplementedException(
        "RasterDB GPU: nullable STRING projection is not yet supported");
    }

    const size_t byte_count = src.validity_byte_size();
    if (byte_count == 0) return;
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    dst.validity = rasterdf::device_buffer(
        _ctx.workspace_mr(), std::max<size_t>(byte_count, sizeof(uint32_t)), usage);
    dst.has_validity = true;
    _ctx.dispatcher().copy_buffer(src.validity.buffer(), dst.validity.buffer(),
                                  byte_count, src.validity.offset(),
                                  dst.validity.offset());
  };

  auto collect_refs = [&](auto& self, duckdb::Expression& e,
                          std::vector<size_t>& refs) -> void {
    if (e.type == duckdb::ExpressionType::BOUND_REF) {
      refs.push_back(e.Cast<duckdb::BoundReferenceExpression>().index);
      return;
    }
    if (e.expression_class == duckdb::ExpressionClass::BOUND_CAST) {
      auto& cast = e.Cast<duckdb::BoundCastExpression>();
      self(self, *cast.child, refs);
      return;
    }
    if (e.expression_class == duckdb::ExpressionClass::BOUND_FUNCTION) {
      auto& func = e.Cast<duckdb::BoundFunctionExpression>();
      for (auto& child : func.children) {
        self(self, *child, refs);
      }
      return;
    }
    if (e.expression_class == duckdb::ExpressionClass::BOUND_OPERATOR) {
      auto& op = e.Cast<duckdb::BoundOperatorExpression>();
      for (auto& child : op.children) {
        self(self, *child, refs);
      }
      return;
    }
    if (e.expression_class == duckdb::ExpressionClass::BOUND_COMPARISON) {
      auto& cmp = e.Cast<duckdb::BoundComparisonExpression>();
      self(self, *cmp.left, refs);
      self(self, *cmp.right, refs);
      return;
    }
    if (e.expression_class == duckdb::ExpressionClass::BOUND_CONJUNCTION) {
      auto& conj = e.Cast<duckdb::BoundConjunctionExpression>();
      for (auto& child : conj.children) {
        self(self, *child, refs);
      }
      return;
    }
    if (e.expression_class == duckdb::ExpressionClass::BOUND_BETWEEN) {
      auto& between = e.Cast<duckdb::BoundBetweenExpression>();
      self(self, *between.input, refs);
      self(self, *between.lower, refs);
      self(self, *between.upper, refs);
    }
  };

  auto materialize_referenced_lazy_columns = [&](duckdb::Expression& e) {
    std::vector<size_t> refs;
    collect_refs(collect_refs, e, refs);
    std::sort(refs.begin(), refs.end());
    refs.erase(std::unique(refs.begin(), refs.end()), refs.end());
    size_t lazy_cols = 0;
    size_t logical_bytes = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (auto idx : refs) {
      if (idx >= input->num_columns()) {
        throw duckdb::InternalException("RasterDB GPU projection: reference index out of range");
      }
      if (input->col(idx).is_lazy()) {
        logical_bytes += static_cast<size_t>(input->col(idx).num_rows) *
                         rdf_type_size(input->col(idx).type.id);
        input->columns[idx] = materialize_column(input->col(idx));
        lazy_cols++;
      }
    }
    if (lazy_cols > 0) {
      auto t1 = std::chrono::high_resolution_clock::now();
      double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      RASTERDB_LOG_INFO("[RDB_LAZY_PROFILE] projection_materialize_refs_ms={:.2f} lazy_cols={} rows={} logical_mb={:.2f}",
                        ms, lazy_cols, input->num_rows(),
                        static_cast<double>(logical_bytes) / (1024.0 * 1024.0));
    }
  };

  for (size_t i = 0; i < op.expressions.size(); i++) {
    auto& expr = *op.expressions[i];

    if (expr.type == duckdb::ExpressionType::BOUND_REF) {
      auto& ref = expr.Cast<duckdb::BoundReferenceExpression>();
      auto& src = input->col(ref.index);

      if (src.is_lazy()) {
        result->columns[i] = alias_lazy_column(src);
        continue;
      }

      // If source is a host-only column (e.g. scalar aggregate), pass through directly
      if (src.is_host_only) {
        result->columns[i].type = src.type;
        result->columns[i].num_rows = src.num_rows;
        result->columns[i].is_host_only = true;
        result->columns[i].host_data = src.host_data;
        copy_validity(src, result->columns[i]);
        continue;
      }

      // STRING columns: move the offsets+chars buffers directly (zero-copy alias)
      if (src.is_string()) {
        result->columns[i].type = src.type;
        result->columns[i].num_rows = src.num_rows;
        result->columns[i].str_offsets = std::move(const_cast<gpu_column&>(src).str_offsets);
        result->columns[i].str_chars = std::move(const_cast<gpu_column&>(src).str_chars);
        result->columns[i].str_total_chars = src.str_total_chars;
        copy_validity(src, result->columns[i]);
        continue;
      }

      // Scan/cache-backed columns can be projected as metadata-only views.
      // Temporary owned columns still take the copy path below because the
      // input table is destroyed when this operator returns.
      if (can_alias_fixed_width_column(src)) {
        result->columns[i] = alias_fixed_width_column(src);
        copy_validity(src, result->columns[i]);
        continue;
      }

      // Copy column to output — use shader for INT32/FLOAT32, buffer copy otherwise
      result->columns[i] = allocate_column(_ctx, src.type, src.num_rows);
      bool has_shader = (src.type.id == rasterdf::type_id::INT32 ||
                         src.type.id == rasterdf::type_id::FLOAT32 ||
                         src.type.id == rasterdf::type_id::TIMESTAMP_DAYS);
      if (has_shader) {
        binary_op_push_constants pc{};
        pc.input_a = src.address();
        pc.input_b = 0;
        pc.output_addr = result->columns[i].address();
        pc.size = static_cast<uint32_t>(src.num_rows);
        pc.op = 0; pc.scalar_val = 0; pc.mode = 1; pc.debug_mode = 0;
        pc.type_id = rdf_shader_type_id(src.type.id);
        _ctx.dispatcher().dispatch_binary_op(pc);
      } else {
        // GPU-side buffer copy for types without shader support (e.g. INT64/FLOAT64)
        size_t byte_count = static_cast<size_t>(src.num_rows) * rdf_type_size(src.type.id);
        VkBuffer src_buf = src.data.buffer() != VK_NULL_HANDLE ? src.data.buffer() : src.cached_buffer;
        VkDeviceSize src_off = src.data.buffer() != VK_NULL_HANDLE ? src.data.offset() : src.cached_offset;
        _ctx.dispatcher().copy_buffer(src_buf, result->columns[i].data.buffer(),
                                      byte_count, src_off,
                                      result->columns[i].data.offset());
      }
      copy_validity(src, result->columns[i]);
    } else if (expr.type == duckdb::ExpressionType::BOUND_FUNCTION) {
      materialize_referenced_lazy_columns(expr);
      result->columns[i] = evaluate_expression(*input, expr);
    } else if (expr.expression_class == duckdb::ExpressionClass::BOUND_CAST) {
      materialize_referenced_lazy_columns(expr);
      result->columns[i] = evaluate_expression(*input, expr);
    } else {
      throw duckdb::NotImplementedException(
        "RasterDB GPU: unsupported projection expression %s",
        duckdb::ExpressionTypeToString(expr.type).c_str());
    }
  }

  return result;
}

} // namespace gpu
} // namespace rasterdb
