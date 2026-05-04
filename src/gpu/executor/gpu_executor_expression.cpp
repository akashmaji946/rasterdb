/*
 * Copyright 2025, RasterDB Contributors.
 * Split from src/gpu/gpu_executor.cpp.
 */

#include "gpu/gpu_executor_internal.hpp"

namespace rasterdb {
namespace gpu {

// ============================================================================
// Evaluate comparison expression -> int32 mask (0/1 per element)
// ============================================================================

gpu_column gpu_executor::evaluate_comparison(const gpu_table& input, duckdb::Expression& expr)
{
  auto& disp = _ctx.dispatcher();
  uint32_t n = static_cast<uint32_t>(input.num_rows());

  // Comparison: column <op> constant or column <op> column
  if (expr.type == duckdb::ExpressionType::COMPARE_LESSTHAN ||
      expr.type == duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO ||
      expr.type == duckdb::ExpressionType::COMPARE_GREATERTHAN ||
      expr.type == duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO ||
      expr.type == duckdb::ExpressionType::COMPARE_EQUAL ||
      expr.type == duckdb::ExpressionType::COMPARE_NOTEQUAL) {

    auto& cmp = expr.Cast<duckdb::BoundComparisonExpression>();

    // Map to shader op code: 0=gt, 1=lt, 2=ge, 3=le, 4=eq, 5=ne
    int32_t cmp_op = 0;
    switch (expr.type) {
      case duckdb::ExpressionType::COMPARE_GREATERTHAN:         cmp_op = 0; break;
      case duckdb::ExpressionType::COMPARE_LESSTHAN:            cmp_op = 1; break;
      case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO: cmp_op = 2; break;
      case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:   cmp_op = 3; break;
      case duckdb::ExpressionType::COMPARE_EQUAL:               cmp_op = 4; break;
      case duckdb::ExpressionType::COMPARE_NOTEQUAL:            cmp_op = 5; break;
      default: break;
    }

    // Unwrap casts inserted by the optimizer
    auto& left = unwrap_cast(*cmp.left);
    auto& right = unwrap_cast(*cmp.right);

    // Column vs constant
    if (left.type == duckdb::ExpressionType::BOUND_REF &&
        right.type == duckdb::ExpressionType::VALUE_CONSTANT) {

      auto& col_ref = left.Cast<duckdb::BoundReferenceExpression>();
      auto& constant = right.Cast<duckdb::BoundConstantExpression>();
      auto& col = input.col(col_ref.index);

      // ── INT64 / FLOAT64 GPU comparison (for HAVING and wide types) ──
      // Uses the compare_int64 compute shader which handles both INT64 and FLOAT64.
      if (col.type.id == rasterdf::type_id::INT64 ||
          col.type.id == rasterdf::type_id::FLOAT64) {
        auto result = allocate_column(_ctx, {rasterdf::type_id::INT32}, n);

        compare_int64_push_constants pc{};
        pc.input_addr = col.address();
        pc.output_addr = result.address();
        pc.size = n;
        pc._pad = 0;
        if (col.type.id == rasterdf::type_id::INT64) {
          pc.threshold = constant.value.DefaultCastAs(duckdb::LogicalType::BIGINT).GetValue<int64_t>();
          pc.type_id = static_cast<int32_t>(rasterdf::ShaderTypeId::INT32); // int64
        } else {
          double dval = constant.value.DefaultCastAs(duckdb::LogicalType::DOUBLE).GetValue<double>();
          int64_t bits;
          std::memcpy(&bits, &dval, sizeof(double));
          pc.threshold = bits;
          pc.type_id = static_cast<int32_t>(rasterdf::ShaderTypeId::FLOAT64); // float64
        }
        pc.op = cmp_op;

        disp.dispatch_compare_int64(pc);
        RASTERDB_LOG_DEBUG("[RDB_DEBUG] HAVING: GPU {} compare on {} rows",
                           col.type.id == rasterdf::type_id::INT64 ? "INT64" : "FLOAT64", n);
        return result;
      }

      int32_t type_id = rdf_shader_type_id(col.type.id);

      auto result = allocate_column(_ctx, {rasterdf::type_id::INT32}, input.num_rows());

      // Cast the constant value to match the column's native type
      int32_t threshold = 0;
      if (type_id == 0) { // int32
        threshold = constant.value.DefaultCastAs(duckdb::LogicalType::INTEGER).GetValue<int32_t>();
      } else { // float32
        float fval = constant.value.DefaultCastAs(duckdb::LogicalType::FLOAT).GetValue<float>();
        std::memcpy(&threshold, &fval, sizeof(float));
      }

      compare_push_constants pc{};
      pc.input_addr = col.address();
      pc.output_addr = result.address();
      pc.size = n;
      pc.threshold = threshold;
      pc.op = cmp_op;
      pc.type_id = type_id;

      disp.dispatch_compare(pc);
      return result;

    } else if (left.type == duckdb::ExpressionType::BOUND_REF &&
               right.type == duckdb::ExpressionType::BOUND_REF) {
      auto& left_ref = left.Cast<duckdb::BoundReferenceExpression>();
      auto& right_ref = right.Cast<duckdb::BoundReferenceExpression>();
      auto& left_col = input.col(left_ref.index);
      auto& right_col = input.col(right_ref.index);
      int32_t type_id = rdf_shader_type_id(left_col.type.id);

      auto result = allocate_column(_ctx, {rasterdf::type_id::INT32}, input.num_rows());

      compare_columns_push_constants pc{};
      pc.input_a = left_col.address();
      pc.input_b = right_col.address();
      pc.output_addr = result.address();
      pc.size = n;
      pc.op = cmp_op;
      pc.type_id = type_id;
      disp.dispatch_compare_columns(pc);
      return result;

    } else {
      throw duckdb::NotImplementedException(
        "RasterDB GPU: unsupported comparison operand types (left=%s, right=%s)",
        duckdb::ExpressionTypeToString(left.type).c_str(),
        duckdb::ExpressionTypeToString(right.type).c_str());
    }
  }

  // Conjunction (AND)
  if (expr.type == duckdb::ExpressionType::CONJUNCTION_AND) {
    auto& conj = expr.Cast<duckdb::BoundConjunctionExpression>();
    gpu_column result = evaluate_comparison(input, *conj.children[0]);
    for (size_t i = 1; i < conj.children.size(); i++) {
      gpu_column child_mask = evaluate_comparison(input, *conj.children[i]);
      auto combined = allocate_column(_ctx, {rasterdf::type_id::INT32}, result.num_rows);
      binary_op_push_constants pc{};
      pc.input_a = result.address();
      pc.input_b = child_mask.address();
      pc.output_addr = combined.address();
      pc.size = static_cast<uint32_t>(result.num_rows);
      pc.op = 2; pc.mode = 0; pc.type_id = static_cast<int32_t>(rasterdf::ShaderTypeId::INT32); pc.debug_mode = 0; pc.scalar_val = 0;
      disp.dispatch_binary_op(pc);
      result = std::move(combined);
    }
    return result;
  }

  throw duckdb::NotImplementedException(
    "RasterDB GPU: unsupported filter expression type %s",
    duckdb::ExpressionTypeToString(expr.type).c_str());
}

// ============================================================================
// Evaluate binary function expression -> gpu_column
// ============================================================================

// ============================================================================
// Evaluate any expression against a gpu_table → returns a gpu_column.
// Handles: BOUND_REF, VALUE_CONSTANT, BOUND_FUNCTION, BOUND_CAST
// ============================================================================

gpu_column gpu_executor::evaluate_expression(const gpu_table& input, duckdb::Expression& raw_expr)
{
  // Strip casts
  auto& expr = unwrap_cast(raw_expr);

  switch (expr.type) {
  case duckdb::ExpressionType::BOUND_REF: {
    auto& ref = expr.Cast<duckdb::BoundReferenceExpression>();
    auto& src = input.col(ref.index);
    // Return a lightweight alias that references the same device memory
    gpu_column col;
    col.type = src.type;
    col.num_rows = src.num_rows;
    // Don't copy device_buffer (deleted copy). Use cached address to alias.
    col.cached_address = src.address();
    col.cached_buffer = src.cached_buffer;
    col.is_host_only = src.is_host_only;
    col.host_data = src.host_data;
    return col;
  }
  case duckdb::ExpressionType::VALUE_CONSTANT: {
    // Broadcast scalar to a full column
    auto& c = expr.Cast<duckdb::BoundConstantExpression>();
    rasterdf::data_type rdf_type = to_rdf_type(c.return_type);
    // Downcast FLOAT64 constants to FLOAT32 (shader only supports INT32/FLOAT32)
    if (rdf_type.id == rasterdf::type_id::FLOAT64) {
      rdf_type = {rasterdf::type_id::FLOAT32};
    }
    auto col = allocate_column(_ctx, rdf_type, input.num_rows());
    // Fill via binary_op: col = 0 + scalar (broadcast)
    binary_op_push_constants pc{};
    pc.input_a = col.address();  // will be overwritten
    pc.output_addr = col.address();
    pc.size = static_cast<uint32_t>(input.num_rows());
    pc.op = 0; // ADD (0 + scalar = scalar)
    pc.mode = 1; // COL_SCALAR — but we need a "fill" op. Use MUL 0 + scalar via ADD trick:
    // Actually, just copy scalar: col = input_a * 0 + scalar? No. Use: output = 0_col + scalar.
    // Simplest: use the identity: set input_a to the same address, op=MUL, scalar=0, then ADD scalar.
    // Actually let's just memset the staging and fill:
    // Easier: binary_op with mode=1 (COL_SCALAR), op=2 (MUL), scalar=1 then scalar=val via ADD
    // Let me just use: output = 0 + scalar via COL_SCALAR ADD with the output as input
    // Hmm, we need a cleaner approach. For now, just allocate and fill on host for small scalar cols.
    // This is used for constant expressions in aggregates, typically rare.
    {
      int32_t type_id_s = rdf_shader_type_id(rdf_type.id);
      // First zero the column, then add scalar to fill
      // Use ADD with scalar, mode=1 means input_a + scalar_val
      // To broadcast, we need input_a to be 0. Zero it first.
      // Actually: simplest approach — allocate, zero with MUL 0, then ADD scalar.
      binary_op_push_constants pz{};
      pz.input_a = col.address();
      pz.output_addr = col.address();
      pz.size = static_cast<uint32_t>(input.num_rows());
      pz.op = 2; // MUL
      pz.mode = 1; // COL_SCALAR
      pz.scalar_val = 0;
      pz.type_id = type_id_s;
      pz.debug_mode = 0;
      _ctx.dispatcher().dispatch_binary_op(pz);

      binary_op_push_constants pa{};
      pa.input_a = col.address();
      pa.output_addr = col.address();
      pa.size = static_cast<uint32_t>(input.num_rows());
      pa.op = 0; // ADD
      pa.mode = 1; // COL_SCALAR
      pa.type_id = type_id_s;
      pa.debug_mode = 0;
      if (type_id_s == 0) {
        pa.scalar_val = c.value.GetValue<int32_t>();
      } else {
        float fval = c.value.GetValue<float>();
        std::memcpy(&pa.scalar_val, &fval, sizeof(float));
      }
      _ctx.dispatcher().dispatch_binary_op(pa);
    }
    return col;
  }
  case duckdb::ExpressionType::BOUND_FUNCTION:
    return evaluate_binary_op(input, expr);
  default:
    throw duckdb::NotImplementedException(
      "RasterDB GPU: unsupported expression type %s in evaluate_expression",
      duckdb::ExpressionTypeToString(expr.type).c_str());
  }
}

// Helper: cast an INT32 gpu_column to FLOAT32 via CPU round-trip.
// Used to align mixed-type operands before dispatching the (single-type) binary_op shader.
static gpu_column cast_int32_to_float32(gpu_context& ctx, const gpu_column& src)
{
  size_t n = static_cast<size_t>(src.num_rows);
  const int32_t* src_int = nullptr;
  std::vector<int32_t> h_int;

  auto& bufMgr = GPUBufferManager::GetInstance();
  if (src.cached_address != 0 && src.cached_buffer == bufMgr.cpuStagingBuffer()) {
    // Zero-copy reBAR: data is directly accessible via mapped CPU staging.
    size_t staging_off = static_cast<size_t>(
        src.cached_address - bufMgr.cpuStagingAddress());
    src_int = reinterpret_cast<const int32_t*>(bufMgr.cpuProcessing + staging_off);
  } else {
    // gpuCache-backed or device_buffer-owned: fall back to generic download.
    h_int.resize(n);
    download_column(ctx, src, h_int.data(), n * sizeof(int32_t));
    src_int = h_int.data();
  }

  std::vector<float> h_flt(n);
  for (size_t i = 0; i < n; i++) h_flt[i] = static_cast<float>(src_int[i]);
  auto out = allocate_column(ctx, {rasterdf::type_id::FLOAT32}, static_cast<rasterdf::size_type>(n));
  out.data.copy_from_host(h_flt.data(), n * sizeof(float),
                          ctx.device(), ctx.queue(), ctx.command_pool());
  return out;
}

gpu_column gpu_executor::evaluate_binary_op(const gpu_table& input, duckdb::Expression& expr)
{
  auto& func = expr.Cast<duckdb::BoundFunctionExpression>();
  auto& fname = func.function.name;

  // Map function names to binary op codes: 0=ADD, 1=SUB, 2=MUL, 3=DIV
  int32_t op_code = -1;
  if (fname == "+" || fname == "add") op_code = 0;
  else if (fname == "-" || fname == "subtract") op_code = 1;
  else if (fname == "*" || fname == "multiply") op_code = 2;
  else if (fname == "/" || fname == "divide") op_code = 3;
  else if (fname == "%" || fname == "modulo") op_code = 4;
  else {
    throw duckdb::NotImplementedException(
      "RasterDB GPU: unsupported function '%s'", fname.c_str());
  }

  D_ASSERT(func.children.size() == 2);

  // Recursively evaluate both operands
  auto& left_expr = unwrap_cast(*func.children[0]);
  auto& right_expr = unwrap_cast(*func.children[1]);

  rasterdf::data_type out_type = to_rdf_type(func.return_type);
  // Downcast FLOAT64 to FLOAT32 for binary op shader (inputs are FLOAT32 from the integer dataset;
  // the aggregation shader accumulates FLOAT32 values in double precision internally)
  if (out_type.id == rasterdf::type_id::FLOAT64) {
    out_type = {rasterdf::type_id::FLOAT32};
  }
  int32_t type_id = rdf_shader_type_id(out_type.id);
  auto result = allocate_column(_ctx, out_type, input.num_rows());

  binary_op_push_constants pc{};
  pc.output_addr = result.address();
  pc.size = static_cast<uint32_t>(input.num_rows());
  pc.op = op_code;
  pc.debug_mode = 0;
  pc.type_id = type_id;

  // Check for simple col-col or col-scalar cases first (avoid temp allocation)
  bool left_is_ref = left_expr.type == duckdb::ExpressionType::BOUND_REF;
  bool right_is_ref = right_expr.type == duckdb::ExpressionType::BOUND_REF;
  bool right_is_const = right_expr.type == duckdb::ExpressionType::VALUE_CONSTANT;
  bool left_is_const = left_expr.type == duckdb::ExpressionType::VALUE_CONSTANT;

  // Resolve addresses — evaluate complex sub-expressions to temp columns
  gpu_column left_temp, right_temp;    // keep alive for address validity
  gpu_column left_cast, right_cast;    // keep cast-to-float temp alive if needed

  // Helper lambda: resolve an operand to (addr, source_type_id). For a BOUND_REF,
  // reads directly from input.col(idx); otherwise evaluates into `temp_out`.
  auto resolve_operand =
      [&](duckdb::Expression& e, bool is_ref, gpu_column& temp_out)
      -> std::pair<VkDeviceAddress, rasterdf::type_id> {
    if (is_ref) {
      auto& ref = e.Cast<duckdb::BoundReferenceExpression>();
      const auto& src = input.col(ref.index);
      return {src.address(), src.type.id};
    }
    temp_out = evaluate_expression(input, e);
    return {temp_out.address(), temp_out.type.id};
  };

  VkDeviceAddress left_addr = 0;
  rasterdf::type_id left_src_type = rasterdf::type_id::INT32;
  if (!left_is_const) {
    auto r = resolve_operand(left_expr, left_is_ref, left_temp);
    left_addr = r.first;
    left_src_type = r.second;
  }

  // If shader will run the float path but this operand is INT32, cast it.
  auto align_if_int_to_float = [&](VkDeviceAddress& addr,
                                   rasterdf::type_id& src_type,
                                   gpu_column& cast_out,
                                   const gpu_column* ref_src) {
    if (out_type.id != rasterdf::type_id::FLOAT32) return;
    if (src_type == rasterdf::type_id::INT32) {
      // Build a gpu_column view of the int32 source so we can cast it.
      gpu_column tmp_view;
      tmp_view.type = {rasterdf::type_id::INT32};
      tmp_view.num_rows = static_cast<rasterdf::size_type>(input.num_rows());
      if (ref_src) {
        tmp_view.cached_address = ref_src->address();
        tmp_view.cached_buffer = ref_src->cached_buffer;
        tmp_view.is_host_only = ref_src->is_host_only;
        tmp_view.host_data = ref_src->host_data;
        cast_out = cast_int32_to_float32(_ctx, *ref_src);
      } else {
        // Can't cheaply view without the gpu_column; fall back to a re-read.
        // Not expected in current flows (non-ref temps are already float).
        throw duckdb::NotImplementedException(
          "RasterDB GPU: unexpected INT32 temp operand requiring float cast");
      }
      addr = cast_out.address();
      src_type = rasterdf::type_id::FLOAT32;
    }
  };

  // Apply cast to left operand if needed (only for BOUND_REF, where we can access the source column)
  if (!left_is_const && left_is_ref) {
    auto& ref = left_expr.Cast<duckdb::BoundReferenceExpression>();
    align_if_int_to_float(left_addr, left_src_type, left_cast, &input.col(ref.index));
  }

  if (left_is_ref && right_is_ref) {
    auto& rref = right_expr.Cast<duckdb::BoundReferenceExpression>();
    VkDeviceAddress right_addr = input.col(rref.index).address();
    rasterdf::type_id right_src_type = input.col(rref.index).type.id;
    align_if_int_to_float(right_addr, right_src_type, right_cast, &input.col(rref.index));
    pc.input_a = left_addr;
    pc.input_b = right_addr;
    pc.mode = 0; // COL_COL
    pc.scalar_val = 0;
  } else if (left_addr != 0 && right_is_const) {
    auto& c = right_expr.Cast<duckdb::BoundConstantExpression>();
    pc.input_a = left_addr;
    pc.input_b = 0;
    pc.mode = 1; // COL_SCALAR
    if (type_id == 0) {
      pc.scalar_val = c.value.GetValue<int32_t>();
    } else {
      float fval = c.value.GetValue<float>();
      std::memcpy(&pc.scalar_val, &fval, sizeof(float));
    }
  } else if (left_is_const && right_is_ref) {
    // SCALAR op COL — swap to COL op SCALAR with adjusted op (only for commutative, else temp)
    auto& c = left_expr.Cast<duckdb::BoundConstantExpression>();
    VkDeviceAddress right_addr = input.col(right_expr.Cast<duckdb::BoundReferenceExpression>().index).address();
    // For subtraction (scalar - col), evaluate scalar as column
    if (op_code == 1 || op_code == 3 || op_code == 4) {
      // Non-commutative: evaluate left as column
      left_temp = evaluate_expression(input, left_expr);
      pc.input_a = left_temp.address();
      pc.input_b = right_addr;
      pc.mode = 0; // COL_COL
    } else {
      // Commutative (ADD, MUL): swap
      pc.input_a = right_addr;
      pc.input_b = 0;
      pc.mode = 1; // COL_SCALAR
      if (type_id == 0) {
        pc.scalar_val = c.value.GetValue<int32_t>();
      } else {
        float fval = c.value.GetValue<float>();
        std::memcpy(&pc.scalar_val, &fval, sizeof(float));
      }
    }
  } else {
    // General case: evaluate both operands to temp columns
    if (!left_addr) {
      left_temp = evaluate_expression(input, left_expr);
      left_addr = left_temp.address();
    }
    right_temp = evaluate_expression(input, right_expr);
    pc.input_a = left_addr;
    pc.input_b = right_temp.address();
    pc.mode = 0; // COL_COL
    pc.scalar_val = 0;
  }

  _ctx.dispatcher().dispatch_binary_op(pc);
  return result;
}

} // namespace gpu
} // namespace rasterdb
