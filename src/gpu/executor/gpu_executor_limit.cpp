/*
 * Copyright 2025, RasterDB Contributors.
 * Split from src/gpu/gpu_executor.cpp.
 */

#include "gpu/gpu_executor_internal.hpp"

namespace rasterdb {
namespace gpu {

// ============================================================================
// LIMIT — take first N rows
// ============================================================================

std::unique_ptr<gpu_table> gpu_executor::execute_limit(duckdb::LogicalLimit& op)
{
  RASTERDB_LOG_DEBUG("GPU execute_limit");
  D_ASSERT(op.children.size() == 1);
  auto input = execute_operator(*op.children[0]);

  stage_timer t("  limit");  // Timer starts AFTER child execution

  int64_t limit_val = 0;
  int64_t offset_val = 0;

  if (op.limit_val.Type() == duckdb::LimitNodeType::CONSTANT_VALUE) {
    limit_val = static_cast<int64_t>(op.limit_val.GetConstantValue());
  } else if (op.limit_val.Type() == duckdb::LimitNodeType::UNSET) {
    limit_val = input->num_rows();
  } else {
    throw duckdb::NotImplementedException("RasterDB GPU: expression LIMIT not supported");
  }

  if (op.offset_val.Type() == duckdb::LimitNodeType::CONSTANT_VALUE) {
    offset_val = static_cast<int64_t>(op.offset_val.GetConstantValue());
  }

  rasterdf::size_type start = static_cast<rasterdf::size_type>(
    std::min(offset_val, static_cast<int64_t>(input->num_rows())));
  rasterdf::size_type count = static_cast<rasterdf::size_type>(
    std::min(limit_val, static_cast<int64_t>(input->num_rows() - start)));

  if (count == input->num_rows() && start == 0) return input;

  auto& disp = _ctx.dispatcher();

  // Create index column [start, start+1, ..., start+count-1]
  auto indices = allocate_column(_ctx, {rasterdf::type_id::INT32}, count);
  uint32_t ng = div_ceil(static_cast<uint32_t>(count), WG_SIZE);
  {
    radix_init_indices_pc pc{};
    pc.indices_ptr = indices.address();
    pc.numElements = static_cast<uint32_t>(count);
    disp.dispatch_radix_init_indices(pc, ng);

    if (start > 0) {
      binary_op_push_constants bpc{};
      bpc.input_a = indices.address();
      bpc.output_addr = indices.address();
      bpc.size = static_cast<uint32_t>(count);
      bpc.op = 0; bpc.scalar_val = static_cast<int32_t>(start);
      bpc.mode = 1; bpc.type_id = 0; bpc.input_b = 0; bpc.debug_mode = 0;
      disp.dispatch_binary_op(bpc);
    }
  }

  auto result = std::make_unique<gpu_table>();
  result->duckdb_types = input->duckdb_types;
  result->columns.resize(input->num_columns());

  for (size_t c = 0; c < input->num_columns(); c++) {
    auto& in_col = input->col(c);
    result->columns[c] = allocate_column(_ctx, in_col.type, count);

    gather_indices_pc gc{};
    gc.input_addr = in_col.address();
    gc.indices_addr = indices.address();
    gc.output_addr = result->columns[c].address();
    gc.size = static_cast<uint32_t>(count);
    if (rdf_type_size(in_col.type.id) == 8) {
      disp.dispatch_gather_indices_64(gc, ng);
    } else {
      disp.dispatch_gather_indices(gc, ng);
    }
  }

  return result;
}

} // namespace gpu
} // namespace rasterdb
