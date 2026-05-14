/*
 * Copyright 2025, RasterDB Contributors.
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

  for (size_t i = 0; i < op.expressions.size(); i++) {
    auto& expr = *op.expressions[i];

    if (expr.type == duckdb::ExpressionType::BOUND_REF) {
      auto& ref = expr.Cast<duckdb::BoundReferenceExpression>();
      auto& src = input->col(ref.index);

      // If source is a host-only column (e.g. scalar aggregate), pass through directly
      if (src.is_host_only) {
        result->columns[i].type = src.type;
        result->columns[i].num_rows = src.num_rows;
        result->columns[i].is_host_only = true;
        result->columns[i].host_data = src.host_data;
        continue;
      }

      // STRING columns: move the offsets+chars buffers directly (zero-copy alias)
      if (src.is_string()) {
        result->columns[i].type = src.type;
        result->columns[i].num_rows = src.num_rows;
        result->columns[i].str_offsets = std::move(const_cast<gpu_column&>(src).str_offsets);
        result->columns[i].str_chars = std::move(const_cast<gpu_column&>(src).str_chars);
        result->columns[i].str_total_chars = src.str_total_chars;
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
        _ctx.dispatcher().copy_buffer(src.data.buffer(), result->columns[i].data.buffer(),
                                      byte_count, src.data.offset(),
                                      result->columns[i].data.offset());
      }
    } else if (expr.type == duckdb::ExpressionType::BOUND_FUNCTION) {
      result->columns[i] = evaluate_binary_op(*input, expr);
    } else if (expr.expression_class == duckdb::ExpressionClass::BOUND_CAST) {
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
