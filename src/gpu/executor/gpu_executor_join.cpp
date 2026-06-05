/*
 * Copyright 2026, RasterDB Contributors.
 * Split from src/gpu/gpu_executor.cpp.
 */

#include "gpu/gpu_executor_internal.hpp"

namespace rasterdb {
namespace gpu {

// ============================================================================
// JOIN — hash join via rasterdf
// ============================================================================

// Toggle between compute-shader hash join and graphics-pipeline simple garuda join
static constexpr bool USE_SIMPLE_GFX_JOIN = false;

// Use the optimized instanced-probe variant (parallel inner loop).
// When true, USE_SIMPLE_GFX_JOIN must also be true.
static constexpr bool USE_SIMPLE_GFX_JOIN_OPT = true;

// Hash bits for Simple Garuda join: num_slots = 1 << k.
// Higher k = more slots = less collisions but more memory.
static constexpr uint32_t USE_SIMPLE_GFX_JOIN_K = 28;

// Delay join payload materialization until after DuckDB projection maps and
// residual join predicates are known. The output remains a normal physical
// gpu_table; this only avoids gathering columns that will be discarded.
static constexpr bool USE_RDB_LATE_JOIN_MATERIALIZATION = true;

static bool decimal_join_keys_compatible(const duckdb::LogicalType& left_type,
                                         const duckdb::LogicalType& right_type)
{
  if (!is_decimal_type(left_type) && !is_decimal_type(right_type)) {
    return true;
  }
  if (!is_decimal_type(left_type) || !is_decimal_type(right_type)) {
    return false;
  }
  if (duckdb::DecimalType::GetScale(left_type) !=
      duckdb::DecimalType::GetScale(right_type)) {
    return false;
  }
  return to_rdf_type(left_type).id == to_rdf_type(right_type).id;
}

static duckdb::Expression& unwrap_join_key_cast(duckdb::Expression& expr)
{
  if (expr.expression_class == duckdb::ExpressionClass::BOUND_CAST) {
    auto& cast = expr.Cast<duckdb::BoundCastExpression>();
    if (is_decimal_type(expr.return_type) &&
        is_decimal_type(cast.child->return_type) &&
        decimal_join_keys_compatible(expr.return_type, cast.child->return_type)) {
      return unwrap_join_key_cast(*cast.child);
    }
  }
  return unwrap_cast(expr);
}

static int32_t shader_type_id_for_compare(const rasterdf::data_type& type)
{
  switch (type.id) {
    case rasterdf::type_id::INT64:
      return static_cast<int32_t>(rasterdf::ShaderTypeId::INT64);
    case rasterdf::type_id::FLOAT64:
      return static_cast<int32_t>(rasterdf::ShaderTypeId::FLOAT64);
    case rasterdf::type_id::FLOAT32:
      return static_cast<int32_t>(rasterdf::ShaderTypeId::FLOAT32);
    default:
      return static_cast<int32_t>(rasterdf::ShaderTypeId::INT32);
  }
}

std::unique_ptr<gpu_table> gpu_executor::execute_join(duckdb::LogicalComparisonJoin& op)
{
  RASTERDB_LOG_DEBUG("GPU execute_join (simple_garuda={})", USE_SIMPLE_GFX_JOIN ? "true" : "false");
  D_ASSERT(op.children.size() == 2);

  const bool is_left_join = op.join_type == duckdb::JoinType::LEFT;
  const bool is_right_join = op.join_type == duckdb::JoinType::RIGHT;
  const bool is_full_join = op.join_type == duckdb::JoinType::OUTER;
  const bool is_outer_join = is_left_join || is_right_join || is_full_join;
  const bool is_inner_join = op.join_type == duckdb::JoinType::INNER;
  if (!is_inner_join && !is_outer_join) {
    throw duckdb::NotImplementedException(
        "RasterDB GPU: supported comparison joins are INNER, LEFT, RIGHT and FULL OUTER, got %s",
        duckdb::JoinTypeToString(op.join_type).c_str());
  }

  // Validate all conditions are equi-joins on column references
  RASTERDB_LOG_DEBUG("[RDB_DEBUG] JOIN: {} conditions", op.conditions.size());
  for (size_t ci = 0; ci < op.conditions.size(); ci++) {
    auto& c = op.conditions[ci];
    RASTERDB_LOG_DEBUG("[RDB_DEBUG]   cond[{}]: cmp={} left={} right={}",
                       ci,
                       duckdb::ExpressionTypeToString(c.comparison),
                       c.left->ToString(),
                       c.right->ToString());
  }
  int equi_condition_idx = -1;
  for (size_t ci = 0; ci < op.conditions.size(); ci++) {
    auto& cond = op.conditions[ci];
    auto& le = unwrap_join_key_cast(*cond.left);
    auto& re = unwrap_join_key_cast(*cond.right);
    if (le.type != duckdb::ExpressionType::BOUND_REF ||
        re.type != duckdb::ExpressionType::BOUND_REF) {
      throw duckdb::NotImplementedException(
        "RasterDB GPU: join conditions must be column references");
    }
    if (!decimal_join_keys_compatible(cond.left->return_type,
                                      cond.right->return_type)) {
      throw duckdb::NotImplementedException(
        "RasterDB GPU: decimal join keys require the same scale and GPU physical width");
    }
    if (cond.comparison == duckdb::ExpressionType::COMPARE_EQUAL && equi_condition_idx < 0) {
      equi_condition_idx = static_cast<int>(ci);
    }
  }

  if (is_outer_join) {
    if (equi_condition_idx < 0 || op.conditions.size() != 1) {
      throw duckdb::NotImplementedException(
          "RasterDB GPU: outer joins currently require exactly one equality condition");
    }
    if constexpr (USE_SIMPLE_GFX_JOIN) {
      throw duckdb::NotImplementedException(
          "RasterDB GPU: outer joins require the compute hash join path");
    }
  }

  // Execute both children
  auto left_table_unique  = execute_operator(*op.children[0]);
  auto right_table_unique = execute_operator(*op.children[1]);
  std::shared_ptr<gpu_table> left_table(std::move(left_table_unique));
  std::shared_ptr<gpu_table> right_table(std::move(right_table_unique));

  stage_timer t("  join");  // Timer starts AFTER child execution

  RASTERDB_LOG_DEBUG("JOIN: left {} rows x {} cols, right {} rows x {} cols",
                     left_table->num_rows(),
                     left_table->num_columns(),
                     right_table->num_rows(),
                     right_table->num_columns());

  std::unique_ptr<rasterdf::column> left_indices;
  std::unique_ptr<rasterdf::column> right_indices;
  rasterdf::size_type match_count = 0;
  auto join_index_start = std::chrono::high_resolution_clock::now();

  if (equi_condition_idx >= 0) {
  // Join on FIRST condition
  auto& cond0        = op.conditions[static_cast<size_t>(equi_condition_idx)];
  auto left_key_idx  = unwrap_join_key_cast(*cond0.left).Cast<duckdb::BoundReferenceExpression>().index;
  auto right_key_idx = unwrap_join_key_cast(*cond0.right).Cast<duckdb::BoundReferenceExpression>().index;
  const auto& left_key_type = cond0.left->return_type;
  const auto& right_key_type = cond0.right->return_type;
  bool decimal_join = is_decimal_type(left_key_type) || is_decimal_type(right_key_type);

  // If join keys are STRING, hash them to INT32 first
  gpu_column left_hash_col, right_hash_col;
  bool string_join = left_table->col(left_key_idx).is_string();
  if (is_outer_join && string_join) {
    throw duckdb::NotImplementedException(
        "RasterDB GPU: outer join on STRING keys is not yet supported");
  }

  if (string_join) {
    auto& lk = left_table->col(left_key_idx);
    auto& rk = right_table->col(right_key_idx);
    uint32_t ln = static_cast<uint32_t>(lk.num_rows);
    uint32_t rn = static_cast<uint32_t>(rk.num_rows);

    left_hash_col = allocate_column(_ctx, {rasterdf::type_id::INT32}, ln);
    right_hash_col = allocate_column(_ctx, {rasterdf::type_id::INT32}, rn);

    string_hash_pc lhpc{};
    lhpc.offsets_ptr = lk.str_offsets.data();
    lhpc.chars_ptr = lk.str_chars.data();
    lhpc.output_ptr = left_hash_col.address();
    lhpc.num_rows = ln;
    _ctx.dispatcher().dispatch_string_hash(lhpc);

    string_hash_pc rhpc{};
    rhpc.offsets_ptr = rk.str_offsets.data();
    rhpc.chars_ptr = rk.str_chars.data();
    rhpc.output_ptr = right_hash_col.address();
    rhpc.num_rows = rn;
    _ctx.dispatcher().dispatch_string_hash(rhpc);

    RASTERDB_LOG_DEBUG("[RDB_DEBUG] STRING JOIN: hashed L={} R={} rows", ln, rn);
  }

  gpu_column left_key_materialized;
  gpu_column right_key_materialized;
  if (!string_join && left_table->col(left_key_idx).is_lazy()) {
    left_key_materialized = materialize_column(left_table->col(left_key_idx));
  }
  if (!string_join && right_table->col(right_key_idx).is_lazy()) {
    right_key_materialized = materialize_column(right_table->col(right_key_idx));
  }

  auto left_key_view  = string_join ? left_hash_col.view()
                                     : (left_table->col(left_key_idx).is_lazy()
                                            ? left_key_materialized.view()
                                            : left_table->col(left_key_idx).view());
  auto right_key_view = string_join ? right_hash_col.view()
                                     : (right_table->col(right_key_idx).is_lazy()
                                            ? right_key_materialized.view()
                                            : right_table->col(right_key_idx).view());
  RASTERDB_LOG_DEBUG(
    "[RDB_DEBUG] JOIN keys: L col[{}] addr=0x{:x} size={}, R col[{}] addr=0x{:x} size={}, decimal={}",
    static_cast<size_t>(left_key_idx),
    static_cast<uint64_t>(left_key_view.data()),
    left_key_view.size(),
    static_cast<size_t>(right_key_idx),
    static_cast<uint64_t>(right_key_view.data()),
    right_key_view.size(),
    decimal_join ? "true" : "false");

  if constexpr (USE_SIMPLE_GFX_JOIN) {
    // ── Simple Garuda Join (graphics-pipeline, vertex shader hash join) ──
    uint32_t left_n  = static_cast<uint32_t>(left_key_view.size());
    uint32_t right_n = static_cast<uint32_t>(right_key_view.size());

    const bool int64_join_keys =
        !string_join &&
        left_key_view.type().id == rasterdf::type_id::INT64 &&
        right_key_view.type().id == rasterdf::type_id::INT64;

    auto sg_result = USE_SIMPLE_GFX_JOIN_OPT
        ? (int64_join_keys
               ? rasterdf::simple_garuda_inner_join_opt_int64(left_key_view.data(),
                                                               left_n,
                                                               right_key_view.data(),
                                                               right_n,
                                                               _ctx.vk_context(),
                                                               _ctx.dispatcher(),
                                                               _ctx.workspace_mr(),
                                                               USE_SIMPLE_GFX_JOIN_K)
               : rasterdf::simple_garuda_inner_join_opt(left_key_view.data(),
                                                         left_n,
                                                         right_key_view.data(),
                                                         right_n,
                                                         _ctx.vk_context(),
                                                         _ctx.dispatcher(),
                                                         _ctx.workspace_mr(),
                                                         USE_SIMPLE_GFX_JOIN_K))
        : (int64_join_keys
               ? rasterdf::simple_garuda_inner_join_int64(left_key_view.data(),
                                                           left_n,
                                                           right_key_view.data(),
                                                           right_n,
                                                           _ctx.vk_context(),
                                                           _ctx.dispatcher(),
                                                           _ctx.workspace_mr(),
                                                           USE_SIMPLE_GFX_JOIN_K)
               : rasterdf::simple_garuda_inner_join(left_key_view.data(),
                                                     left_n,
                                                     right_key_view.data(),
                                                     right_n,
                                                     _ctx.vk_context(),
                                                     _ctx.dispatcher(),
                                                     _ctx.workspace_mr(),
                                                     USE_SIMPLE_GFX_JOIN_K));

    match_count = static_cast<rasterdf::size_type>(sg_result.num_matches);

    if (match_count > 0) {
      left_indices =
        std::make_unique<rasterdf::column>(rasterdf::data_type{rasterdf::type_id::INT32},
                                           match_count,
                                           std::move(*sg_result.left_indices));
      right_indices =
        std::make_unique<rasterdf::column>(rasterdf::data_type{rasterdf::type_id::INT32},
                                           match_count,
                                           std::move(*sg_result.right_indices));
    }
  } else {
    // ── Compute-shader hash join ──
    const bool int128_join_keys =
        !string_join &&
        left_key_view.type().id == rasterdf::type_id::INT128 &&
        right_key_view.type().id == rasterdf::type_id::INT128;
    const bool int64_join_keys =
        !string_join &&
        left_key_view.type().id == rasterdf::type_id::INT64 &&
        right_key_view.type().id == rasterdf::type_id::INT64;

    rasterdf::join_result join_result;
    if (int128_join_keys) {
      RASTERDB_LOG_DEBUG("[RDB_DEBUG] JOIN path=int128_hash");
      if (is_left_join) {
        join_result = rasterdf::left_join_int128_hash(
          left_key_view, right_key_view, _ctx.vk_context(), _ctx.dispatcher(), _ctx.workspace_mr());
      } else if (is_right_join) {
        join_result = rasterdf::right_join_int128_hash(
          left_key_view, right_key_view, _ctx.vk_context(), _ctx.dispatcher(), _ctx.workspace_mr());
      } else if (is_full_join) {
        join_result = rasterdf::full_join_int128_hash(
          left_key_view, right_key_view, _ctx.vk_context(), _ctx.dispatcher(), _ctx.workspace_mr());
      } else {
        join_result = rasterdf::inner_join_int128_hash(
          left_key_view, right_key_view, _ctx.vk_context(), _ctx.dispatcher(), _ctx.workspace_mr());
      }
    } else if (int64_join_keys) {
      RASTERDB_LOG_DEBUG("[RDB_DEBUG] JOIN path=int64_hash");
      if (is_left_join) {
        join_result = rasterdf::left_join_int64_hash(
          left_key_view, right_key_view, _ctx.vk_context(), _ctx.dispatcher(), _ctx.workspace_mr());
      } else if (is_right_join) {
        join_result = rasterdf::right_join_int64_hash(
          left_key_view, right_key_view, _ctx.vk_context(), _ctx.dispatcher(), _ctx.workspace_mr());
      } else if (is_full_join) {
        join_result = rasterdf::full_join_int64_hash(
          left_key_view, right_key_view, _ctx.vk_context(), _ctx.dispatcher(), _ctx.workspace_mr());
      } else {
        join_result = rasterdf::inner_join_int64_hash(
          left_key_view, right_key_view, _ctx.vk_context(), _ctx.dispatcher(), _ctx.workspace_mr());
      }
    } else {
      RASTERDB_LOG_DEBUG("[RDB_DEBUG] JOIN path=int32_hash");
      std::vector<rasterdf::column_view> lk = {left_key_view};
      std::vector<rasterdf::column_view> rk = {right_key_view};
      rasterdf::table_view left_keys_tv(lk);
      rasterdf::table_view right_keys_tv(rk);
      if (is_left_join) {
        join_result = rasterdf::left_join_hash(
          left_keys_tv, right_keys_tv, _ctx.vk_context(), _ctx.dispatcher(), _ctx.workspace_mr());
      } else if (is_right_join) {
        join_result = rasterdf::right_join_hash(
          left_keys_tv, right_keys_tv, _ctx.vk_context(), _ctx.dispatcher(), _ctx.workspace_mr());
      } else if (is_full_join) {
        join_result = rasterdf::full_join_hash(
          left_keys_tv, right_keys_tv, _ctx.vk_context(), _ctx.dispatcher(), _ctx.workspace_mr());
      } else {
        join_result = rasterdf::inner_join(
          left_keys_tv, right_keys_tv, _ctx.vk_context(), _ctx.dispatcher(), _ctx.workspace_mr());
      }
    }

    left_indices  = std::move(join_result.first);
    right_indices = std::move(join_result.second);
    match_count   = left_indices ? left_indices->size() : 0;
  }
  RASTERDB_LOG_DEBUG("JOIN: {} matches after first condition", match_count);
  } else {
    if (op.conditions.size() != 1) {
      throw duckdb::NotImplementedException(
        "RasterDB GPU: non-equi join without equality supports one condition");
    }
    auto& cond0 = op.conditions[0];
    auto left_key_idx = unwrap_join_key_cast(*cond0.left).Cast<duckdb::BoundReferenceExpression>().index;
    auto right_key_idx = unwrap_join_key_cast(*cond0.right).Cast<duckdb::BoundReferenceExpression>().index;
    auto& left_key_col = left_table->col(left_key_idx);
    auto& right_key_col = right_table->col(right_key_idx);
    if (left_key_col.is_string() || right_key_col.is_string() ||
        left_key_col.type.id != rasterdf::type_id::INT32 ||
        right_key_col.type.id != rasterdf::type_id::INT32) {
      throw duckdb::NotImplementedException(
        "RasterDB GPU: non-equi join currently supports INT32 fixed-width keys");
    }
    uint64_t product = static_cast<uint64_t>(left_table->num_rows()) *
                       static_cast<uint64_t>(right_table->num_rows());
    if (product > 100'000'000) { // 100 Million
      throw duckdb::NotImplementedException(
        "RasterDB GPU: non-equi join without equality condition too large");
    }
    int32_t cmp_op = 4;
    switch (cond0.comparison) {
      case duckdb::ExpressionType::COMPARE_GREATERTHAN:          cmp_op = 0; break;
      case duckdb::ExpressionType::COMPARE_LESSTHAN:             cmp_op = 1; break;
      case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO: cmp_op = 2; break;
      case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:    cmp_op = 3; break;
      case duckdb::ExpressionType::COMPARE_EQUAL:                cmp_op = 4; break;
      case duckdb::ExpressionType::COMPARE_NOTEQUAL:             cmp_op = 5; break;
      default:
        throw duckdb::NotImplementedException("RasterDB GPU: unsupported non-equi join comparison");
    }
    RASTERDB_LOG_DEBUG("[RDB_DEBUG] NON_EQUI_JOIN: L={} rows R={} rows product={} cmp={}",
                       left_table->num_rows(), right_table->num_rows(), product,
                       duckdb::ExpressionTypeToString(cond0.comparison));
    auto join_result = rasterdf::non_equi_join_int32(left_key_col.address(),
                                                     static_cast<uint32_t>(left_table->num_rows()),
                                                     right_key_col.address(),
                                                     static_cast<uint32_t>(right_table->num_rows()),
                                                     cmp_op,
                                                     _ctx.vk_context(),
                                                     _ctx.dispatcher(),
                                                     _ctx.workspace_mr());
    match_count = static_cast<rasterdf::size_type>(join_result.num_matches);
    if (match_count > 0) {
      left_indices =
        std::make_unique<rasterdf::column>(rasterdf::data_type{rasterdf::type_id::INT32},
                                           match_count,
                                           std::move(*join_result.left_indices));
      right_indices =
        std::make_unique<rasterdf::column>(rasterdf::data_type{rasterdf::type_id::INT32},
                                           match_count,
                                           std::move(*join_result.right_indices));
    }
    RASTERDB_LOG_DEBUG("JOIN: {} matches after non-equi condition", match_count);
  }

  if (match_count == 0) {
    auto result          = std::make_unique<gpu_table>();
    result->duckdb_types = op.types;
    result->columns.resize(op.types.size());
    for (size_t i = 0; i < op.types.size(); i++) {
      result->columns[i].type     = to_rdf_type(op.types[i]);
      result->columns[i].num_rows = 0;
    }
    return result;
  }

  {
    auto join_index_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(join_index_end - join_index_start).count();
    RASTERDB_LOG_INFO("[RDB_JOIN_PROFILE] index_probe_ms={:.2f} rows_left={} rows_right={} rows_out={}",
                      ms, left_table->num_rows(), right_table->num_rows(), match_count);
  }

  if (_join_limit >= 0 && match_count > static_cast<rasterdf::size_type>(_join_limit)) {
    RASTERDB_LOG_DEBUG("JOIN: limiting materialized output from {} to {} rows",
                       match_count, _join_limit);
    match_count = static_cast<rasterdf::size_type>(_join_limit);
  }

  gpu_column left_idx_col = gpu_column_from_rdf(std::move(*left_indices));
  gpu_column right_idx_col = gpu_column_from_rdf(std::move(*right_indices));
  left_indices.reset();
  right_indices.reset();

  auto make_index_view = [](const gpu_column& idx_col, rasterdf::size_type count) {
    return rasterdf::column_view(
        rasterdf::data_type{rasterdf::type_id::INT32}, count,
        idx_col.view().data(), 0, 0, 0);
  };

  rasterdf::column_view left_idx_view = make_index_view(left_idx_col, match_count);
  rasterdf::column_view right_idx_view = make_index_view(right_idx_col, match_count);

  bool residual_prefilter_applied = false;
  if constexpr (USE_RDB_LATE_JOIN_MATERIALIZATION) {
    if (equi_condition_idx >= 0 && op.conditions.size() > 1 && match_count > 0) {
      auto residual_start = std::chrono::high_resolution_clock::now();
      gpu_column combined_mask;
      bool has_mask = false;
      bool can_prefilter_all = true;

      for (size_t ci = 0; ci < op.conditions.size(); ci++) {
        if (static_cast<int>(ci) == equi_condition_idx) {
          continue;
        }
        auto& cond = op.conditions[ci];
        auto lk_idx = unwrap_join_key_cast(*cond.left).Cast<duckdb::BoundReferenceExpression>().index;
        auto rk_idx = unwrap_join_key_cast(*cond.right).Cast<duckdb::BoundReferenceExpression>().index;
        const auto& left_src_orig = left_table->col(lk_idx);
        const auto& right_src_orig = right_table->col(rk_idx);
        auto left_src_mat = left_src_orig.is_lazy() ? materialize_column(left_src_orig) : gpu_column{};
        auto right_src_mat = right_src_orig.is_lazy() ? materialize_column(right_src_orig) : gpu_column{};
        const auto& left_src = left_src_orig.is_lazy() ? left_src_mat : left_src_orig;
        const auto& right_src = right_src_orig.is_lazy() ? right_src_mat : right_src_orig;
        if (left_src.is_string() || right_src.is_string()) {
          can_prefilter_all = false;
          break;
        }

        int32_t cmp_op = 4;
        switch (cond.comparison) {
          case duckdb::ExpressionType::COMPARE_GREATERTHAN:          cmp_op = 0; break;
          case duckdb::ExpressionType::COMPARE_LESSTHAN:             cmp_op = 1; break;
          case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO: cmp_op = 2; break;
          case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:    cmp_op = 3; break;
          case duckdb::ExpressionType::COMPARE_EQUAL:                cmp_op = 4; break;
          case duckdb::ExpressionType::COMPARE_NOTEQUAL:             cmp_op = 5; break;
          default:
            can_prefilter_all = false;
            break;
        }
        if (!can_prefilter_all) {
          break;
        }

        auto gathered_left = rasterdf::gather(left_src.view(), left_idx_view,
                                              _ctx.vk_context(), _ctx.dispatcher(),
                                              _ctx.workspace_mr());
        auto gathered_right = rasterdf::gather(right_src.view(), right_idx_view,
                                               _ctx.vk_context(), _ctx.dispatcher(),
                                               _ctx.workspace_mr());
        auto mask = allocate_column(_ctx, {rasterdf::type_id::INT32}, match_count);

        compare_columns_push_constants cpc{};
        cpc.input_a     = gathered_left->view().data();
        cpc.input_b     = gathered_right->view().data();
        cpc.output_addr = mask.address();
        cpc.size        = static_cast<uint32_t>(match_count);
        cpc.op          = cmp_op;
        cpc.type_id     = shader_type_id_for_compare(left_src.type);
        _ctx.dispatcher().dispatch_compare_columns(cpc);

        if (!has_mask) {
          combined_mask = std::move(mask);
          has_mask = true;
        } else {
          auto next_mask = allocate_column(_ctx, {rasterdf::type_id::INT32}, match_count);
          mask_op_push_constants mpc{};
          mpc.input_a = combined_mask.address();
          mpc.input_b = mask.address();
          mpc.output_addr = next_mask.address();
          mpc.size = static_cast<uint32_t>(match_count);
          mpc.op = 0;
          _ctx.dispatcher().dispatch_mask_op(mpc);
          combined_mask = std::move(next_mask);
        }
      }

      if (can_prefilter_all && has_mask) {
        gpu_table idx_table;
        idx_table.duckdb_types = {duckdb::LogicalType::INTEGER, duckdb::LogicalType::INTEGER};
        idx_table.columns.resize(2);
        idx_table.columns[0] = std::move(left_idx_col);
        idx_table.columns[1] = std::move(right_idx_col);
        idx_table.set_num_rows(match_count);

        auto filtered_indices = apply_filter_mask(idx_table, combined_mask);
        match_count = filtered_indices->num_rows();
        if (match_count == 0) {
          auto empty = std::make_unique<gpu_table>();
          empty->duckdb_types = op.types;
          empty->columns.resize(op.types.size());
          for (size_t i = 0; i < op.types.size(); i++) {
            empty->columns[i].type = to_rdf_type(op.types[i]);
            empty->columns[i].num_rows = 0;
          }
          return empty;
        }
        left_idx_col = std::move(filtered_indices->columns[0]);
        right_idx_col = std::move(filtered_indices->columns[1]);
        left_idx_view = make_index_view(left_idx_col, match_count);
        right_idx_view = make_index_view(right_idx_col, match_count);
        residual_prefilter_applied = true;

        auto residual_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(residual_end - residual_start).count();
        RASTERDB_LOG_INFO("[RDB_JOIN_PROFILE] residual_prefilter_ms={:.2f} rows_after={}",
                          ms, match_count);
      }
    }
  }

  auto left_idx_shared = std::make_shared<gpu_column>(std::move(left_idx_col));
  auto right_idx_shared = std::make_shared<gpu_column>(std::move(right_idx_col));
  left_idx_view = make_index_view(*left_idx_shared, match_count);
  right_idx_view = make_index_view(*right_idx_shared, match_count);

  // Helper: gather a single string column using index array
  auto gather_string_col = [&](const gpu_column& in_col, const rasterdf::column_view& idx_view,
                                rasterdf::size_type count) -> gpu_column {
    auto& disp = _ctx.dispatcher();
    auto* mr = _ctx.workspace_mr();
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    uint32_t nc = static_cast<uint32_t>(count);

    rasterdf::device_buffer out_offsets(mr, (nc + 1) * sizeof(int32_t), usage);
    // Write lengths into out_offsets[0..N-1]
    string_lengths_pc lpc{};
    lpc.offsets_ptr = in_col.str_offsets.data();
    lpc.indices_ptr = idx_view.data();
    lpc.num_indices = nc;
    lpc.output_ptr = out_offsets.data();
    disp.dispatch_string_lengths(lpc);
    // Zero element N, then exclusive prefix scan on N+1 elements
    disp.fill_buffer(out_offsets.buffer(), 0u, sizeof(int32_t), out_offsets.offset() + nc * sizeof(int32_t));

    uint32_t scan_elems = nc + 1;
    uint32_t scan_ngroups = (scan_elems + 255) / 256;
    rasterdf::device_buffer scan_bsums(mr, scan_ngroups * sizeof(uint32_t), usage);
    rasterdf::device_buffer scan_total(mr, sizeof(uint32_t), usage);
    prefix_scan_pc opc{};
    opc.data_ptr = out_offsets.data();
    opc.block_sums_ptr = scan_bsums.data();
    opc.total_sum_ptr = scan_total.data();
    opc.numElements = scan_elems;
    opc.blockCount = scan_ngroups;
    disp.dispatch_prefix_scan_local(opc, scan_ngroups);
    disp.dispatch_prefix_scan_global(opc);
    disp.dispatch_prefix_scan_add(opc, scan_ngroups);

    int32_t total_chars = 0;
    out_offsets.copy_to_host(&total_chars, sizeof(int32_t),
                             static_cast<size_t>(nc) * sizeof(int32_t),
                             _ctx.device(), _ctx.queue(), _ctx.command_pool());

    rasterdf::device_buffer out_chars(mr, std::max(total_chars, 1), usage);
    string_copy_pc cpc{};
    cpc.in_offsets_ptr = in_col.str_offsets.data();
    cpc.in_chars_ptr = in_col.str_chars.data();
    cpc.indices_ptr = idx_view.data();
    cpc.out_offsets_ptr = out_offsets.data();
    cpc.out_chars_ptr = out_chars.data();
    cpc.num_indices = nc;
    disp.dispatch_string_copy(cpc);

    gpu_column out;
    out.type = rasterdf::data_type{rasterdf::type_id::STRING};
    out.num_rows = count;
    out.str_offsets = std::move(out_offsets);
    out.str_chars = std::move(out_chars);
    out.str_total_chars = total_chars;
    return out;
  };

  auto gather_numeric_outer_col = [&](const gpu_column& in_col,
                                      const rasterdf::column_view& idx_view,
                                      rasterdf::size_type count) -> gpu_column {
    if (in_col.is_string()) {
      throw duckdb::NotImplementedException(
          "RasterDB GPU: nullable STRING payloads from outer joins are not yet supported");
    }
    auto out = allocate_column(_ctx, in_col.type, count);
    size_t validity_bytes = ((static_cast<size_t>(count) + 31u) / 32u) * sizeof(uint32_t);
    out.validity = rasterdf::device_buffer(_ctx.workspace_mr(), std::max<size_t>(validity_bytes, sizeof(uint32_t)));
    out.has_validity = true;
    _ctx.dispatcher().fill_buffer(out.validity.buffer(), 0u, validity_bytes, out.validity.offset());

    rasterdf::execution::outer_gather_indices_pc pc{};
    pc.input_addr = in_col.address();
    pc.indices_addr = idx_view.data();
    pc.output_addr = out.address();
    pc.validity_addr = out.validity.data();
    pc.size = static_cast<uint32_t>(count);
    uint32_t groups = (static_cast<uint32_t>(count) + 255u) / 256u;
    size_t elem_size = rdf_type_size(in_col.type.id);
    if (elem_size == 16) {
      _ctx.dispatcher().dispatch_outer_gather_indices_128(pc, groups);
    } else if (elem_size == 8) {
      _ctx.dispatcher().dispatch_outer_gather_indices_64(pc, groups);
    } else {
      _ctx.dispatcher().dispatch_outer_gather_indices(pc, groups);
    }
    return out;
  };

  // Check if any columns are STRING — if so, we can't use rasterdf::gather for them
  bool any_left_string = false, any_right_string = false;
  for (size_t i = 0; i < left_table->num_columns(); i++)
    if (left_table->col(i).is_string()) { any_left_string = true; break; }
  for (size_t i = 0; i < right_table->num_columns(); i++)
    if (right_table->col(i).is_string()) { any_right_string = true; break; }

  const bool left_side_nullable = is_right_join || is_full_join;
  const bool right_side_nullable = is_left_join || is_full_join;
  if (is_outer_join &&
      ((left_side_nullable && any_left_string) ||
       (right_side_nullable && any_right_string))) {
    throw duckdb::NotImplementedException(
        "RasterDB GPU: outer joins with nullable STRING payload columns are not yet supported");
  }

  auto add_unique_col = [](std::vector<size_t>& cols, size_t col_idx) {
    if (std::find(cols.begin(), cols.end(), col_idx) == cols.end()) {
      cols.push_back(col_idx);
    }
  };

  auto estimate_fixed_width_bytes = [&](const gpu_table& tbl,
                                        const std::vector<size_t>& cols,
                                        rasterdf::size_type rows) -> size_t {
    size_t bytes = 0;
    for (auto c : cols) {
      const auto& col = tbl.col(c);
      if (col.is_string()) {
        bytes += static_cast<size_t>(rows + 1) * sizeof(int32_t);
        bytes += static_cast<size_t>(col.str_total_chars);
      } else {
        bytes += static_cast<size_t>(rows) * rdf_type_size(col.type.id);
      }
      if (col.has_validity) {
        bytes += ((static_cast<size_t>(rows) + 31u) / 32u) * sizeof(uint32_t);
      }
    }
    return bytes;
  };

  auto result = std::make_unique<gpu_table>();
  result->duckdb_types = op.types;
  size_t num_left_cols = left_table->num_columns();
  size_t num_right_cols = right_table->num_columns();
  size_t total_cols = num_left_cols + num_right_cols;

  std::vector<size_t> left_result_index(num_left_cols, std::numeric_limits<size_t>::max());
  std::vector<size_t> right_result_index(num_right_cols, std::numeric_limits<size_t>::max());

  if constexpr (USE_RDB_LATE_JOIN_MATERIALIZATION) {
    bool has_left_map = !op.left_projection_map.empty();
    bool has_right_map = !op.right_projection_map.empty();

    std::vector<size_t> left_cols_to_gather;
    std::vector<size_t> right_cols_to_gather;
    if (has_left_map) {
      for (auto src_idx : op.left_projection_map) add_unique_col(left_cols_to_gather, src_idx);
    } else {
      for (size_t i = 0; i < num_left_cols; i++) add_unique_col(left_cols_to_gather, i);
    }
    if (has_right_map) {
      for (auto src_idx : op.right_projection_map) add_unique_col(right_cols_to_gather, src_idx);
    } else {
      for (size_t i = 0; i < num_right_cols; i++) add_unique_col(right_cols_to_gather, i);
    }

    for (size_t ci = 0; ci < op.conditions.size(); ci++) {
      if (static_cast<int>(ci) == equi_condition_idx || equi_condition_idx < 0) {
        continue;
      }
      auto& cond = op.conditions[ci];
      auto lk_idx = unwrap_join_key_cast(*cond.left).Cast<duckdb::BoundReferenceExpression>().index;
      auto rk_idx = unwrap_join_key_cast(*cond.right).Cast<duckdb::BoundReferenceExpression>().index;
      add_unique_col(left_cols_to_gather, lk_idx);
      add_unique_col(right_cols_to_gather, rk_idx);
    }

    auto make_lazy_output_col = [&](std::shared_ptr<gpu_table> owner,
                                    size_t src_idx,
                                    const std::shared_ptr<gpu_column>& idx_shared,
                                    const rasterdf::column_view& idx_view) -> gpu_column {
      const auto& src = owner->col(src_idx);
      if (src.is_host_only || src.is_string()) {
        throw duckdb::InternalException("RasterDB GPU: unsupported lazy join payload column");
      }

      gpu_column out;
      out.type = src.type;
      out.num_rows = match_count;
      out.has_i32_minmax = src.has_i32_minmax;
      out.i32_min = src.i32_min;
      out.i32_max = src.i32_max;

      if (src.is_lazy()) {
        auto composed = rasterdf::gather(src.lazy_row_indices->view(), idx_view,
                                         _ctx.vk_context(), _ctx.dispatcher(),
                                         _ctx.workspace_mr());
        auto composed_idx = gpu_column_from_rdf(std::move(*composed));
        out.lazy_base_table = src.lazy_base_table;
        out.lazy_base_col_idx = src.lazy_base_col_idx;
        out.lazy_row_indices = std::make_shared<gpu_column>(std::move(composed_idx));
      } else {
        out.lazy_base_table = owner;
        out.lazy_base_col_idx = src_idx;
        out.lazy_row_indices = idx_shared;
      }
      return out;
    };

    auto gather_selected_side = [&](std::shared_ptr<gpu_table> owner,
                                    const rasterdf::column_view& idx_view,
                                    const std::shared_ptr<gpu_column>& idx_shared,
                                    const std::vector<size_t>& selected_cols,
                                    bool nullable_side,
                                    std::vector<size_t>& result_index,
                                    const char* side_name) {
      const gpu_table& input = *owner;
      auto side_start = std::chrono::high_resolution_clock::now();
      size_t start_out = result->columns.size();
      bool selected_has_string = false;
      for (auto c : selected_cols) {
        if (input.col(c).is_string()) {
          selected_has_string = true;
          break;
        }
      }

      const bool can_return_lazy_side = is_inner_join && !nullable_side && !selected_has_string;
      if (can_return_lazy_side) {
        for (auto src_idx : selected_cols) {
          result_index[src_idx] = result->columns.size();
          result->columns.push_back(make_lazy_output_col(owner, src_idx, idx_shared, idx_view));
        }
      } else if (nullable_side || selected_has_string) {
        for (auto src_idx : selected_cols) {
          result_index[src_idx] = result->columns.size();
          if (nullable_side) {
            result->columns.push_back(gather_numeric_outer_col(input.col(src_idx), idx_view, match_count));
          } else if (input.col(src_idx).is_string()) {
            result->columns.push_back(gather_string_col(input.col(src_idx), idx_view, match_count));
          } else {
            auto col_view = input.col(src_idx).view();
            std::vector<rasterdf::column_view> cv = {col_view};
            rasterdf::table_view tv(cv);
            auto gathered = rasterdf::gather(tv, idx_view, _ctx.vk_context(),
                                             _ctx.dispatcher(), _ctx.workspace_mr());
            auto cols = gathered->extract();
            result->columns.push_back(gpu_column_from_rdf(std::move(*cols[0])));
          }
        }
      } else if (!selected_cols.empty()) {
        std::vector<rasterdf::size_type> projected_cols;
        projected_cols.reserve(selected_cols.size());
        for (auto src_idx : selected_cols) {
          projected_cols.push_back(static_cast<rasterdf::size_type>(src_idx));
        }
        auto gathered = rasterdf::gather(input.view(), idx_view, projected_cols,
                                         _ctx.vk_context(), _ctx.dispatcher(),
                                         _ctx.workspace_mr());
        auto cols = gathered->extract();
        for (size_t i = 0; i < selected_cols.size(); i++) {
          result_index[selected_cols[i]] = result->columns.size();
          result->columns.push_back(gpu_column_from_rdf(std::move(*cols[i])));
        }
      }

      auto side_end = std::chrono::high_resolution_clock::now();
      double ms = std::chrono::duration<double, std::milli>(side_end - side_start).count();
      size_t logical_bytes = estimate_fixed_width_bytes(input, selected_cols, match_count);
      RASTERDB_LOG_INFO("[RDB_JOIN_PROFILE] materialize_{}_ms={:.2f} cols={} of {} rows={} logical_mb={:.2f}",
                        side_name, ms, result->columns.size() - start_out,
                        input.num_columns(), match_count,
                        static_cast<double>(logical_bytes) / (1024.0 * 1024.0));
    };

    result->columns.reserve(left_cols_to_gather.size() + right_cols_to_gather.size());
    gather_selected_side(left_table, left_idx_view, left_idx_shared, left_cols_to_gather,
                         left_side_nullable, left_result_index, "left");
    gather_selected_side(right_table, right_idx_view, right_idx_shared, right_cols_to_gather,
                         right_side_nullable, right_result_index, "right");
    result->set_num_rows(match_count);
    RASTERDB_LOG_INFO("[RDB_JOIN_PROFILE] late_materialization=1 gathered_cols={} of {} rows={}",
                      result->num_columns(), total_cols, result->num_rows());
  } else {
    result->columns.resize(total_cols);

    auto left_gather_start = std::chrono::high_resolution_clock::now();
    if (left_side_nullable) {
      for (size_t i = 0; i < num_left_cols; i++) {
        result->columns[i] = gather_numeric_outer_col(left_table->col(i), left_idx_view, match_count);
        left_result_index[i] = i;
      }
    } else if (!any_left_string) {
      auto left_gathered = rasterdf::gather(
        left_table->view(), left_idx_view, _ctx.vk_context(), _ctx.dispatcher(), _ctx.workspace_mr());
      auto left_cols = left_gathered->extract();
      for (size_t i = 0; i < left_cols.size(); i++) {
        result->columns[i] = gpu_column_from_rdf(std::move(*left_cols[i]));
        left_result_index[i] = i;
      }
    } else {
      for (size_t i = 0; i < num_left_cols; i++) {
        left_result_index[i] = i;
        if (left_table->col(i).is_string()) {
          result->columns[i] = gather_string_col(left_table->col(i), left_idx_view, match_count);
        } else {
          auto col_view = left_table->col(i).view();
          std::vector<rasterdf::column_view> cv = {col_view};
          rasterdf::table_view tv(cv);
          auto gathered = rasterdf::gather(tv, left_idx_view, _ctx.vk_context(), _ctx.dispatcher(), _ctx.workspace_mr());
          auto cols = gathered->extract();
          result->columns[i] = gpu_column_from_rdf(std::move(*cols[0]));
        }
      }
    }
    RASTERDB_LOG_INFO("[RDB_JOIN_PROFILE] materialize_left_ms={:.2f} cols={} of {} rows={} logical_mb={:.2f}",
                      std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - left_gather_start).count(),
                      num_left_cols, num_left_cols, match_count,
                      static_cast<double>(estimate_fixed_width_bytes(*left_table,
                        [&](){ std::vector<size_t> v; for (size_t i = 0; i < num_left_cols; i++) v.push_back(i); return v; }(),
                        match_count)) / (1024.0 * 1024.0));

    auto right_gather_start = std::chrono::high_resolution_clock::now();
    if (right_side_nullable) {
      for (size_t i = 0; i < num_right_cols; i++) {
        result->columns[num_left_cols + i] =
            gather_numeric_outer_col(right_table->col(i), right_idx_view, match_count);
        right_result_index[i] = num_left_cols + i;
      }
    } else if (!any_right_string) {
      auto right_gathered = rasterdf::gather(
        right_table->view(), right_idx_view, _ctx.vk_context(), _ctx.dispatcher(), _ctx.workspace_mr());
      auto right_cols = right_gathered->extract();
      for (size_t i = 0; i < right_cols.size(); i++) {
        result->columns[num_left_cols + i] = gpu_column_from_rdf(std::move(*right_cols[i]));
        right_result_index[i] = num_left_cols + i;
      }
    } else {
      for (size_t i = 0; i < num_right_cols; i++) {
        right_result_index[i] = num_left_cols + i;
        if (right_table->col(i).is_string()) {
          result->columns[num_left_cols + i] =
              gather_string_col(right_table->col(i), right_idx_view, match_count);
        } else {
          auto col_view = right_table->col(i).view();
          std::vector<rasterdf::column_view> cv = {col_view};
          rasterdf::table_view tv(cv);
          auto gathered = rasterdf::gather(tv, right_idx_view, _ctx.vk_context(), _ctx.dispatcher(), _ctx.workspace_mr());
          auto cols = gathered->extract();
          result->columns[num_left_cols + i] = gpu_column_from_rdf(std::move(*cols[0]));
        }
      }
    }
    RASTERDB_LOG_INFO("[RDB_JOIN_PROFILE] materialize_right_ms={:.2f} cols={} of {} rows={} logical_mb={:.2f}",
                      std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - right_gather_start).count(),
                      num_right_cols, num_right_cols, match_count,
                      static_cast<double>(estimate_fixed_width_bytes(*right_table,
                        [&](){ std::vector<size_t> v; for (size_t i = 0; i < num_right_cols; i++) v.push_back(i); return v; }(),
                        match_count)) / (1024.0 * 1024.0));
    result->set_num_rows(match_count);
  }

  RASTERDB_LOG_DEBUG("JOIN result: {} rows x {} materialized cols (logical join cols={})",
                     match_count, result->num_columns(), total_cols);

  // Post-filter on remaining conditions (multi-condition join)
  for (size_t ci = 0; ci < op.conditions.size(); ci++) {
    if (static_cast<int>(ci) == equi_condition_idx || equi_condition_idx < 0) {
      continue;
    }
    if (residual_prefilter_applied) {
      continue;
    }
    auto& cond  = op.conditions[ci];
    auto lk_idx = unwrap_join_key_cast(*cond.left).Cast<duckdb::BoundReferenceExpression>().index;
    auto rk_idx = unwrap_join_key_cast(*cond.right).Cast<duckdb::BoundReferenceExpression>().index;

    // In the merged table: left cols at [0..num_left-1], right cols at [num_left..]
    size_t left_mat_idx = left_result_index[lk_idx];
    size_t right_mat_idx = right_result_index[rk_idx];
    if (left_mat_idx == std::numeric_limits<size_t>::max() ||
        right_mat_idx == std::numeric_limits<size_t>::max()) {
      throw duckdb::InternalException("RasterDB GPU join: residual predicate column was not materialized");
    }
    if (result->col(left_mat_idx).is_lazy() || result->col(right_mat_idx).is_lazy()) {
      result = materialize_table(*result);
    }
    auto& left_key_col  = result->col(left_mat_idx);
    auto& right_key_col = result->col(right_mat_idx);

    uint32_t n = static_cast<uint32_t>(result->num_rows());
    auto mask  = allocate_column(_ctx, {rasterdf::type_id::INT32}, n);

    int32_t cmp_op = 4;
    switch (cond.comparison) {
      case duckdb::ExpressionType::COMPARE_GREATERTHAN:          cmp_op = 0; break;
      case duckdb::ExpressionType::COMPARE_LESSTHAN:             cmp_op = 1; break;
      case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO: cmp_op = 2; break;
      case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:    cmp_op = 3; break;
      case duckdb::ExpressionType::COMPARE_EQUAL:                cmp_op = 4; break;
      case duckdb::ExpressionType::COMPARE_NOTEQUAL:             cmp_op = 5; break;
      default:
        throw duckdb::NotImplementedException("RasterDB GPU: unsupported post-join comparison");
    }

    compare_columns_push_constants cpc{};
    cpc.input_a     = left_key_col.address();
    cpc.input_b     = right_key_col.address();
    cpc.output_addr = mask.address();
    cpc.size        = n;
    cpc.op          = cmp_op;
    cpc.type_id     = shader_type_id_for_compare(left_key_col.type);
    _ctx.dispatcher().dispatch_compare_columns(cpc);

    result = apply_filter_mask(*result, mask);
    RASTERDB_LOG_DEBUG("JOIN: {} rows after condition {}", result->num_rows(), ci);
  }

  // Apply join projection maps: empty map = "all columns from that side"
  {
      size_t num_right_cols = right_table->num_columns();
    bool has_left_map = !op.left_projection_map.empty();
    bool has_right_map = !op.right_projection_map.empty();

    if (has_left_map || has_right_map) {
      auto projected = std::make_unique<gpu_table>();
      projected->duckdb_types = op.types;

      size_t out_left  = has_left_map  ? op.left_projection_map.size()  : num_left_cols;
      size_t out_right = has_right_map ? op.right_projection_map.size() : num_right_cols;
      projected->columns.resize(out_left + out_right);

      size_t out_idx = 0;
      if (has_left_map) {
        for (auto src_idx : op.left_projection_map) {
          size_t mat_idx = left_result_index[src_idx];
          if (mat_idx == std::numeric_limits<size_t>::max()) {
            throw duckdb::InternalException("RasterDB GPU join: left projection column was not materialized");
          }
          projected->columns[out_idx++] = std::move(result->columns[mat_idx]);
        }
      } else {
        for (size_t i = 0; i < num_left_cols; i++) {
          size_t mat_idx = left_result_index[i];
          if (mat_idx == std::numeric_limits<size_t>::max()) {
            throw duckdb::InternalException("RasterDB GPU join: left output column was not materialized");
          }
          projected->columns[out_idx++] = std::move(result->columns[mat_idx]);
        }
      }
      if (has_right_map) {
        for (auto src_idx : op.right_projection_map) {
          size_t mat_idx = right_result_index[src_idx];
          if (mat_idx == std::numeric_limits<size_t>::max()) {
            throw duckdb::InternalException("RasterDB GPU join: right projection column was not materialized");
          }
          projected->columns[out_idx++] = std::move(result->columns[mat_idx]);
        }
      } else {
        for (size_t i = 0; i < num_right_cols; i++) {
          size_t mat_idx = right_result_index[i];
          if (mat_idx == std::numeric_limits<size_t>::max()) {
            throw duckdb::InternalException("RasterDB GPU join: right output column was not materialized");
          }
          projected->columns[out_idx++] = std::move(result->columns[mat_idx]);
        }
      }
      projected->set_num_rows(result->num_rows());
      RASTERDB_LOG_DEBUG("JOIN projection_map: {} cols => {} cols (left_map={} right_map={})",
                         result->num_columns(), projected->num_columns(),
                         has_left_map, has_right_map);
      result = std::move(projected);
    }
  }

  return result;
}

}  // namespace gpu
}  // namespace rasterdb
