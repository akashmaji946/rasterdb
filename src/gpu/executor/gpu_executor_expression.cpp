/*
 * Copyright 2026, RasterDB Contributors.
 * Split from src/gpu/gpu_executor.cpp.
 */

#include "gpu/gpu_executor_internal.hpp"

namespace rasterdb {
namespace gpu {

static int64_t encode_int64_scalar(const duckdb::Value& value, const duckdb::LogicalType& target_type)
{
  if (target_type.id() == duckdb::LogicalTypeId::DECIMAL) {
    auto scaled = value.DefaultCastAs(target_type);
    return scaled.GetValueUnsafe<int64_t>();
  }
  return value.DefaultCastAs(duckdb::LogicalType::BIGINT).GetValue<int64_t>();
}

static int32_t encode_int32_scalar(const duckdb::Value& value, const duckdb::LogicalType& target_type)
{
  if (target_type.id() == duckdb::LogicalTypeId::DECIMAL) {
    auto scaled = value.DefaultCastAs(target_type);
    if (target_type.InternalType() == duckdb::PhysicalType::INT16) {
      return static_cast<int32_t>(scaled.GetValueUnsafe<int16_t>());
    }
    return scaled.GetValueUnsafe<int32_t>();
  }
  return value.DefaultCastAs(duckdb::LogicalType::INTEGER).GetValue<int32_t>();
}

static duckdb::hugeint_t encode_int128_scalar(const duckdb::Value& value,
                                              const duckdb::LogicalType& target_type)
{
  if (target_type.id() == duckdb::LogicalTypeId::DECIMAL) {
    auto scaled = value.DefaultCastAs(target_type);
    return scaled.GetValueUnsafe<duckdb::hugeint_t>();
  }
  return value.DefaultCastAs(duckdb::LogicalType::HUGEINT).GetValue<duckdb::hugeint_t>();
}

// ============================================================================
// Evaluate comparison expression -> int32 mask (0/1 per element)
// ============================================================================

gpu_column gpu_executor::evaluate_comparison(const gpu_table& input, duckdb::Expression& expr)
{
  auto& disp = _ctx.dispatcher();
  uint32_t n = static_cast<uint32_t>(input.num_rows());

  if (expr.expression_class == duckdb::ExpressionClass::BOUND_BETWEEN) {
    auto& between = expr.Cast<duckdb::BoundBetweenExpression>();
    auto make_cmp_mask = [&](duckdb::ExpressionType cmp_type,
                             duckdb::Expression& bound_expr) -> gpu_column {
      auto& input_expr = unwrap_cast(*between.input);
      if (input_expr.type != duckdb::ExpressionType::BOUND_REF ||
          unwrap_cast(bound_expr).type != duckdb::ExpressionType::VALUE_CONSTANT) {
        throw duckdb::NotImplementedException(
          "RasterDB GPU: BETWEEN currently supports column BETWEEN constants");
      }
      auto left = duckdb::make_uniq<duckdb::BoundReferenceExpression>(
        input_expr.return_type, input_expr.Cast<duckdb::BoundReferenceExpression>().index);
      auto right = duckdb::make_uniq<duckdb::BoundConstantExpression>(
        unwrap_cast(bound_expr).Cast<duckdb::BoundConstantExpression>().value);
      duckdb::BoundComparisonExpression cmp(cmp_type, std::move(left), std::move(right));
      return evaluate_comparison(input, cmp);
    };

    gpu_column lower_mask = make_cmp_mask(between.LowerComparisonType(), *between.lower);
    gpu_column upper_mask = make_cmp_mask(between.UpperComparisonType(), *between.upper);
    auto combined = allocate_column(_ctx, {rasterdf::type_id::INT32}, lower_mask.num_rows);
    mask_op_push_constants pc{};
    pc.input_a = lower_mask.address();
    pc.input_b = upper_mask.address();
    pc.output_addr = combined.address();
    pc.size = static_cast<uint32_t>(lower_mask.num_rows);
    pc.op = 0;
    disp.dispatch_mask_op(pc);
    return combined;
  }

  if (expr.type == duckdb::ExpressionType::COMPARE_IN ||
      expr.type == duckdb::ExpressionType::COMPARE_NOT_IN) {
    auto& op = expr.Cast<duckdb::BoundOperatorExpression>();
    if (op.children.size() < 2) {
      throw duckdb::NotImplementedException(
        "RasterDB GPU: IN requires one input expression and at least one value");
    }

    auto& value_expr = unwrap_cast(*op.children[0]);
    if (value_expr.type != duckdb::ExpressionType::BOUND_REF) {
      throw duckdb::NotImplementedException(
        "RasterDB GPU: IN currently supports only column IN constant-list");
    }

    auto make_eq_mask = [&](duckdb::Expression& constant_expr) -> gpu_column {
      auto left = duckdb::make_uniq<duckdb::BoundReferenceExpression>(
        value_expr.return_type, value_expr.Cast<duckdb::BoundReferenceExpression>().index);
      auto right = duckdb::make_uniq<duckdb::BoundConstantExpression>(
        constant_expr.Cast<duckdb::BoundConstantExpression>().value);
      duckdb::BoundComparisonExpression cmp(
        duckdb::ExpressionType::COMPARE_EQUAL, std::move(left), std::move(right));
      return evaluate_comparison(input, cmp);
    };

    auto& first_const = unwrap_cast(*op.children[1]);
    if (first_const.type != duckdb::ExpressionType::VALUE_CONSTANT) {
      throw duckdb::NotImplementedException(
        "RasterDB GPU: IN currently supports only constant-list values");
    }
    gpu_column result = make_eq_mask(first_const);

    for (size_t i = 2; i < op.children.size(); i++) {
      auto& child = unwrap_cast(*op.children[i]);
      if (child.type != duckdb::ExpressionType::VALUE_CONSTANT) {
        throw duckdb::NotImplementedException(
          "RasterDB GPU: IN currently supports only constant-list values");
      }
      gpu_column child_mask = make_eq_mask(child);
      auto combined = allocate_column(_ctx, {rasterdf::type_id::INT32}, result.num_rows);
      mask_op_push_constants pc{};
      pc.input_a = result.address();
      pc.input_b = child_mask.address();
      pc.output_addr = combined.address();
      pc.size = static_cast<uint32_t>(result.num_rows);
      pc.op = 1;
      disp.dispatch_mask_op(pc);
      result = std::move(combined);
    }

    if (expr.type == duckdb::ExpressionType::COMPARE_NOT_IN) {
      auto inverted = allocate_column(_ctx, {rasterdf::type_id::INT32}, result.num_rows);
      mask_op_push_constants pc{};
      pc.input_a = result.address();
      pc.input_b = result.address();
      pc.output_addr = inverted.address();
      pc.size = static_cast<uint32_t>(result.num_rows);
      pc.op = 2;
      disp.dispatch_mask_op(pc);
      return inverted;
    }

    return result;
  }

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
      case duckdb::ExpressionType::COMPARE_GREATERTHAN:           cmp_op = 0; break;
      case duckdb::ExpressionType::COMPARE_LESSTHAN:              cmp_op = 1; break;
      case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:  cmp_op = 2; break;
      case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:     cmp_op = 3; break;
      case duckdb::ExpressionType::COMPARE_EQUAL:                 cmp_op = 4; break;
      case duckdb::ExpressionType::COMPARE_NOTEQUAL:              cmp_op = 5; break;
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
      const auto& col_logical_type = col_ref.index < input.duckdb_types.size()
                                       ? input.duckdb_types[col_ref.index]
                                       : left.return_type;

      // ── STRING GPU comparison ──
      if (col.is_string()) {
        uint32_t str_n = static_cast<uint32_t>(col.num_rows);
        auto result = allocate_column(_ctx, {rasterdf::type_id::INT32}, str_n);

        // Upload target string to GPU
        auto target_str = constant.value.DefaultCastAs(duckdb::LogicalType::VARCHAR)
                              .GetValue<duckdb::string>();
        auto target_len = static_cast<uint32_t>(target_str.size());
        rasterdf::device_buffer target_buf(
            _ctx.workspace_mr(), std::max(target_len, 1u),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        if (target_len > 0) {
          target_buf.copy_from_host(target_str.data(), target_len,
                                    _ctx.device(), _ctx.queue(), _ctx.command_pool());
        }

        string_compare_pc spc{};
        spc.offsets_ptr = col.str_offsets.data();
        spc.chars_ptr = col.str_chars.data();
        spc.output_ptr = result.address();
        spc.target_ptr = target_buf.data();
        spc.num_rows = str_n;
        spc.target_len = target_len;
        spc.op = cmp_op;
        disp.dispatch_string_compare(spc);
        RASTERDB_LOG_DEBUG("[RDB_DEBUG] STRING compare: {} rows, target='{}', op={}",
                           str_n, target_str, cmp_op);
        return result;
      }

      // ── INT128 / DECIMAL128 GPU comparison ──
      // DECIMAL(19..38) is stored as DuckDB hugeint_t and RasterDF INT128:
      // two little-endian 64-bit limbs (lower unsigned, upper signed). The
      // shader compares upper signed limbs first, then lower unsigned limbs.
      if (col.type.id == rasterdf::type_id::INT128) {
        auto result = allocate_column(_ctx, {rasterdf::type_id::INT32}, n);
        auto wide = encode_int128_scalar(constant.value, col_logical_type);

        compare_int128_push_constants pc{};
        pc.input_addr = col.address();
        pc.output_addr = result.address();
        pc.size = n;
        pc._pad = 0;
        pc.threshold_lo = wide.lower;
        pc.threshold_hi = wide.upper;
        pc.op = cmp_op;

        disp.dispatch_compare_int128(pc);
        RASTERDB_LOG_DEBUG("[RDB_DEBUG] GPU INT128/DECIMAL128 compare on {} rows", n);
        return result;
      }

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
          pc.threshold = encode_int64_scalar(constant.value, col_logical_type);
          pc.type_id = static_cast<int32_t>(rasterdf::ShaderTypeId::INT64);
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
      if (type_id == static_cast<int32_t>(rasterdf::ShaderTypeId::INT32)) {
        threshold = encode_int32_scalar(constant.value, col_logical_type);
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
      const auto& left_type = left_ref.index < input.duckdb_types.size()
                                ? input.duckdb_types[left_ref.index]
                                : left.return_type;
      const auto& right_type = right_ref.index < input.duckdb_types.size()
                                 ? input.duckdb_types[right_ref.index]
                                 : right.return_type;
      if ((is_decimal_type(left_type) || is_decimal_type(right_type)) &&
          !(left_type == right_type)) {
        throw duckdb::NotImplementedException(
          "RasterDB GPU: mixed-scale decimal column comparison requires rescaling support");
      }
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

  // Conjunction (OR)
  if (expr.type == duckdb::ExpressionType::CONJUNCTION_OR) {
    auto& conj = expr.Cast<duckdb::BoundConjunctionExpression>();
    gpu_column result = evaluate_comparison(input, *conj.children[0]);
    for (size_t i = 1; i < conj.children.size(); i++) {
      gpu_column child_mask = evaluate_comparison(input, *conj.children[i]);
      auto combined = allocate_column(_ctx, {rasterdf::type_id::INT32}, result.num_rows);
      mask_op_push_constants pc{};
      pc.input_a = result.address();
      pc.input_b = child_mask.address();
      pc.output_addr = combined.address();
      pc.size = static_cast<uint32_t>(result.num_rows);
      pc.op = 1; // OR
      disp.dispatch_mask_op(pc);
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

static gpu_column cast_int32_to_float32(gpu_context& ctx, const gpu_column& src);
static gpu_column cast_float32_to_int32(gpu_context& ctx, const gpu_column& src);

gpu_column gpu_executor::evaluate_expression(const gpu_table& input, duckdb::Expression& raw_expr)
{
  // CASE WHEN <cond> THEN <const> ELSE <const> END → select_if_int32
  if (raw_expr.expression_class == duckdb::ExpressionClass::BOUND_CASE) {
    auto& case_expr = raw_expr.Cast<duckdb::BoundCaseExpression>();
    if (case_expr.case_checks.size() != 1 || !case_expr.else_expr) {
      throw duckdb::NotImplementedException(
        "RasterDB GPU: CASE currently supports exactly one WHEN branch with ELSE");
    }
    auto& when_expr = *case_expr.case_checks[0].when_expr;
    auto& then_expr = unwrap_cast(*case_expr.case_checks[0].then_expr);
    auto& else_expr = unwrap_cast(*case_expr.else_expr);

    if (then_expr.type != duckdb::ExpressionType::VALUE_CONSTANT ||
        else_expr.type != duckdb::ExpressionType::VALUE_CONSTANT) {
      throw duckdb::NotImplementedException(
        "RasterDB GPU: CASE currently supports only constant THEN/ELSE values");
    }
    int32_t true_val = then_expr.Cast<duckdb::BoundConstantExpression>()
                         .value.DefaultCastAs(duckdb::LogicalType::INTEGER)
                         .GetValue<int32_t>();
    int32_t false_val = else_expr.Cast<duckdb::BoundConstantExpression>()
                          .value.DefaultCastAs(duckdb::LogicalType::INTEGER)
                          .GetValue<int32_t>();

    gpu_column mask = evaluate_comparison(input, when_expr);
    auto out = allocate_column(_ctx, {rasterdf::type_id::INT32}, input.num_rows());
    select_if_int32_push_constants pc{};
    pc.mask_addr = mask.address();
    pc.output_addr = out.address();
    pc.size = static_cast<uint32_t>(input.num_rows());
    pc.true_value = true_val;
    pc.false_value = false_val;
    _ctx.dispatcher().dispatch_select_if_int32(pc);
    return out;
  }

  if (raw_expr.expression_class == duckdb::ExpressionClass::BOUND_CAST) {
    auto& cast = raw_expr.Cast<duckdb::BoundCastExpression>();

    // Constant folding for casts
    if (cast.child->type == duckdb::ExpressionType::VALUE_CONSTANT) {
      auto& c = cast.child->Cast<duckdb::BoundConstantExpression>();
      duckdb::Value cast_val = c.value.DefaultCastAs(raw_expr.return_type);
      auto cast_const = duckdb::make_uniq<duckdb::BoundConstantExpression>(cast_val);
      return evaluate_expression(input, *cast_const);
    }

    if ((raw_expr.return_type.id() == duckdb::LogicalTypeId::DECIMAL ||
         cast.child->return_type.id() == duckdb::LogicalTypeId::DECIMAL) &&
        !(raw_expr.return_type == cast.child->return_type)) {
      throw duckdb::NotImplementedException(
        "RasterDB GPU: decimal cast/rescale from %s to %s requires decimal rescale support",
        cast.child->return_type.ToString().c_str(),
        raw_expr.return_type.ToString().c_str());
    }

    auto child_col = evaluate_expression(input, *cast.child);
    rasterdf::data_type target_type = to_rdf_type(raw_expr.return_type);
    if (target_type.id == rasterdf::type_id::FLOAT64) {
      target_type = {rasterdf::type_id::FLOAT32};
    }
    if (child_col.type.id == target_type.id) {
      return child_col;
    }
    if (target_type.id == rasterdf::type_id::INT32 &&
        child_col.type.id == rasterdf::type_id::FLOAT32) {
      return cast_float32_to_int32(_ctx, child_col);
    }
    if (target_type.id == rasterdf::type_id::FLOAT32 &&
        child_col.type.id == rasterdf::type_id::INT32) {
      return cast_int32_to_float32(_ctx, child_col);
    }
    throw duckdb::NotImplementedException(
      "RasterDB GPU: unsupported cast from type_id %d to %s",
      static_cast<int>(child_col.type.id),
      raw_expr.return_type.ToString().c_str());
  }

  auto& expr = raw_expr;

  switch (expr.type) {
  case duckdb::ExpressionType::BOUND_REF: {
    auto& ref = expr.Cast<duckdb::BoundReferenceExpression>();
    auto& src = input.col(ref.index);
    auto copy_validity = [&](gpu_column& dst) {
      if (!src.has_validity) return;
      if (src.is_string()) {
        throw duckdb::NotImplementedException(
          "RasterDB GPU: nullable STRING expression reference is not yet supported");
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
    gpu_column col;
    col.type = src.type;
    col.num_rows = src.num_rows;
    col.is_host_only = src.is_host_only;
    col.host_data = src.host_data;
    if (src.is_string()) {
      col.str_offsets = std::move(const_cast<gpu_column&>(src).str_offsets);
      col.str_chars = std::move(const_cast<gpu_column&>(src).str_chars);
      col.str_total_chars = src.str_total_chars;
      col.cached_address = 0; // no fixed-width data
    } else if (can_alias_fixed_width_column(src)) {
      col = alias_fixed_width_column(src);
    } else if (!src.is_host_only) {
      col = allocate_column(_ctx, src.type, src.num_rows);
      size_t bytes = src.byte_size();
      if (bytes > 0) {
        if (src.data.buffer() != VK_NULL_HANDLE) {
          _ctx.dispatcher().copy_buffer(src.data.buffer(),
                                        col.data.buffer(),
                                        bytes,
                                        src.data.offset(),
                                        col.data.offset());
        } else if (src.cached_buffer != VK_NULL_HANDLE) {
          _ctx.dispatcher().copy_buffer(src.cached_buffer,
                                        col.data.buffer(),
                                        bytes,
                                        src.cached_offset,
                                        col.data.offset());
        } else {
          std::vector<uint8_t> h(bytes);
          download_column(_ctx, src, h.data(), bytes);
          col.data.copy_from_host(h.data(), bytes, _ctx.device(), _ctx.queue(), _ctx.command_pool());
        }
      }
    } else {
      col.cached_address = 0;
      col.cached_buffer = VK_NULL_HANDLE;
    }
    copy_validity(col);
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
      if (rdf_type.id == rasterdf::type_id::INT64 ||
          rdf_type.id == rasterdf::type_id::FLOAT64) {
        binary_op_int64_push_constants pz64{};
        pz64.input_a = pz.input_a;
        pz64.input_b = pz.input_b;
        pz64.output_addr = pz.output_addr;
        pz64.size = pz.size;
        pz64.op = pz.op;
        pz64.scalar_lo = 0;
        pz64.scalar_hi = 0;
        pz64.mode = pz.mode;
        pz64.type_id = type_id_s;
        _ctx.dispatcher().dispatch_binary_op_int64(pz64);
      } else {
        _ctx.dispatcher().dispatch_binary_op(pz);
      }

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
        _ctx.dispatcher().dispatch_binary_op(pa);
      } else if (rdf_type.id == rasterdf::type_id::INT64 ||
                 rdf_type.id == rasterdf::type_id::FLOAT64) {
        int64_t scalar64 = 0;
        if (rdf_type.id == rasterdf::type_id::FLOAT64) {
          double dval = c.value.DefaultCastAs(duckdb::LogicalType::DOUBLE).GetValue<double>();
          std::memcpy(&scalar64, &dval, sizeof(double));
        } else {
          scalar64 = encode_int64_scalar(c.value, c.return_type);
        }
        binary_op_int64_push_constants pa64{};
        pa64.input_a = pa.input_a;
        pa64.input_b = pa.input_b;
        pa64.output_addr = pa.output_addr;
        pa64.size = pa.size;
        pa64.op = pa.op;
        pa64.scalar_lo = static_cast<int32_t>(scalar64 & 0xFFFFFFFFLL);
        pa64.scalar_hi = static_cast<int32_t>((static_cast<uint64_t>(scalar64) >> 32) & 0xFFFFFFFFULL);
        pa64.mode = pa.mode;
        pa64.type_id = type_id_s;
        _ctx.dispatcher().dispatch_binary_op_int64(pa64);
      } else {
        float fval = c.value.GetValue<float>();
        std::memcpy(&pa.scalar_val, &fval, sizeof(float));
        _ctx.dispatcher().dispatch_binary_op(pa);
      }
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

static gpu_column cast_float32_to_int32(gpu_context& ctx, const gpu_column& src)
{
  size_t n = static_cast<size_t>(src.num_rows);
  std::vector<float> h_flt(n);
  download_column(ctx, src, h_flt.data(), n * sizeof(float));

  std::vector<int32_t> h_int(n);
  for (size_t i = 0; i < n; i++) h_int[i] = static_cast<int32_t>(h_flt[i]);
  auto out = allocate_column(ctx, {rasterdf::type_id::INT32}, static_cast<rasterdf::size_type>(n));
  out.data.copy_from_host(h_int.data(), n * sizeof(int32_t),
                          ctx.device(), ctx.queue(), ctx.command_pool());
  return out;
}

gpu_column gpu_executor::evaluate_binary_op(const gpu_table& input, duckdb::Expression& expr)
{
  auto& func = expr.Cast<duckdb::BoundFunctionExpression>();
  auto& fname = func.function.name;

  if ((fname.find("__internal_compress") != std::string::npos ||
       fname.find("__internal_decompress") != std::string::npos) &&
      !func.children.empty()) {
    return evaluate_expression(input, *func.children[0]);
  }

  bool uses_decimal = is_decimal_type(func.return_type);
  for (auto& child : func.children) {
    uses_decimal = uses_decimal || is_decimal_type(child->return_type);
  }
  if (uses_decimal) {
    throw duckdb::NotImplementedException(
      "RasterDB GPU: decimal arithmetic requires fixed-point rescale and overflow support");
  }

  // Map function names to binary op codes: 0=ADD, 1=SUB, 2=MUL, 3=DIV
  int32_t op_code = -1;
  if (fname == "+" || fname == "add") op_code = 0;
  else if (fname == "-" || fname == "subtract") op_code = 1;
  else if (fname == "*" || fname == "multiply") op_code = 2;
  else if (fname == "/" || fname == "divide") op_code = 3;
  else if (fname == "%" || fname == "modulo") op_code = 4;
  else {
    RASTERDB_LOG_INFO("[RDB_EXPR] unsupported function '{}' children={}", fname, func.children.size());
    throw duckdb::NotImplementedException(
      "RasterDB GPU: unsupported function '%s'", fname.c_str());
  }

  D_ASSERT(func.children.size() == 2);

  // Recursively evaluate both operands
  auto& left_expr = *func.children[0];
  auto& right_expr = *func.children[1];

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

  if (out_type.id == rasterdf::type_id::INT64 ||
      out_type.id == rasterdf::type_id::FLOAT64) {
    binary_op_int64_push_constants pc64{};
    pc64.input_a = pc.input_a;
    pc64.input_b = pc.input_b;
    pc64.output_addr = pc.output_addr;
    pc64.size = pc.size;
    pc64.op = pc.op;
    int64_t scalar64 = static_cast<int64_t>(pc.scalar_val);
    if (pc.mode == 1) {
      if (right_is_const) {
        auto& c = right_expr.Cast<duckdb::BoundConstantExpression>();
        if (out_type.id == rasterdf::type_id::FLOAT64) {
          double dval = c.value.DefaultCastAs(duckdb::LogicalType::DOUBLE).GetValue<double>();
          std::memcpy(&scalar64, &dval, sizeof(double));
        } else {
          scalar64 = encode_int64_scalar(c.value, c.return_type);
        }
      } else if (left_is_const) {
        auto& c = left_expr.Cast<duckdb::BoundConstantExpression>();
        if (out_type.id == rasterdf::type_id::FLOAT64) {
          double dval = c.value.DefaultCastAs(duckdb::LogicalType::DOUBLE).GetValue<double>();
          std::memcpy(&scalar64, &dval, sizeof(double));
        } else {
          scalar64 = encode_int64_scalar(c.value, c.return_type);
        }
      }
    }
    pc64.scalar_lo = static_cast<int32_t>(scalar64 & 0xFFFFFFFFLL);
    pc64.scalar_hi = static_cast<int32_t>((static_cast<uint64_t>(scalar64) >> 32) & 0xFFFFFFFFULL);
    pc64.mode = pc.mode;
    pc64.type_id = type_id;
    _ctx.dispatcher().dispatch_binary_op_int64(pc64);
  } else {
    _ctx.dispatcher().dispatch_binary_op(pc);
  }
  return result;
}

} // namespace gpu
} // namespace rasterdb
