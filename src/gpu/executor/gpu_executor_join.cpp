/*
 * Copyright 2025, RasterDB Contributors.
 * Split from src/gpu/gpu_executor.cpp.
 */

#include "gpu/gpu_executor_internal.hpp"

namespace rasterdb {
namespace gpu {

// ============================================================================
// JOIN — hash join via rasterdf
// ============================================================================

// Toggle between compute-shader hash join and graphics-pipeline simple garuda join
static constexpr bool USE_SIMPLE_GFX_JOIN = true;

// Use the optimized instanced-probe variant (parallel inner loop).
// When true, USE_SIMPLE_GFX_JOIN must also be true.
static constexpr bool USE_SIMPLE_GFX_JOIN_OPT = false;

// Hash bits for Simple Garuda join: num_slots = 1 << k.
// Higher k = more slots = less collisions but more memory.
static constexpr uint32_t USE_SIMPLE_GFX_JOIN_K = 28;

std::unique_ptr<gpu_table> gpu_executor::execute_join(duckdb::LogicalComparisonJoin& op)
{
  RASTERDB_LOG_DEBUG("GPU execute_join (simple_garuda={})", USE_SIMPLE_GFX_JOIN ? "true" : "false");
  D_ASSERT(op.children.size() == 2);

  if (op.join_type != duckdb::JoinType::INNER) {
    throw duckdb::NotImplementedException("RasterDB GPU: only INNER JOIN supported, got %s",
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
  for (auto& cond : op.conditions) {
    if (cond.comparison != duckdb::ExpressionType::COMPARE_EQUAL) {
      throw duckdb::NotImplementedException("RasterDB GPU: only equi-join supported");
    }
    auto& le = unwrap_cast(*cond.left);
    auto& re = unwrap_cast(*cond.right);
    if (le.type != duckdb::ExpressionType::BOUND_REF ||
        re.type != duckdb::ExpressionType::BOUND_REF) {
      throw duckdb::NotImplementedException(
        "RasterDB GPU: join conditions must be column references");
    }
  }

  // Execute both children
  auto left_table  = execute_operator(*op.children[0]);
  auto right_table = execute_operator(*op.children[1]);

  stage_timer t("  join");  // Timer starts AFTER child execution

  RASTERDB_LOG_DEBUG("JOIN: left {} rows x {} cols, right {} rows x {} cols",
                     left_table->num_rows(),
                     left_table->num_columns(),
                     right_table->num_rows(),
                     right_table->num_columns());

  // Join on FIRST condition
  auto& cond0        = op.conditions[0];
  auto left_key_idx  = unwrap_cast(*cond0.left).Cast<duckdb::BoundReferenceExpression>().index;
  auto right_key_idx = unwrap_cast(*cond0.right).Cast<duckdb::BoundReferenceExpression>().index;

  // If join keys are STRING, hash them to INT32 first
  gpu_column left_hash_col, right_hash_col;
  bool string_join = left_table->col(left_key_idx).is_string();

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

  auto left_key_view  = string_join ? left_hash_col.view()
                                     : left_table->col(left_key_idx).view();
  auto right_key_view = string_join ? right_hash_col.view()
                                     : right_table->col(right_key_idx).view();
  RASTERDB_LOG_DEBUG(
    "[RDB_DEBUG] JOIN keys: L col[{}] addr=0x{:x} size={}, R col[{}] addr=0x{:x} size={}",
    static_cast<size_t>(left_key_idx),
    static_cast<uint64_t>(left_key_view.data()),
    left_key_view.size(),
    static_cast<size_t>(right_key_idx),
    static_cast<uint64_t>(right_key_view.data()),
    right_key_view.size());

  std::unique_ptr<rasterdf::column> left_indices;
  std::unique_ptr<rasterdf::column> right_indices;
  rasterdf::size_type match_count = 0;

  if constexpr (USE_SIMPLE_GFX_JOIN) {
    // ── Simple Garuda Join (graphics-pipeline, vertex shader hash join) ──
    uint32_t left_n  = static_cast<uint32_t>(left_key_view.size());
    uint32_t right_n = static_cast<uint32_t>(right_key_view.size());

    auto sg_result = USE_SIMPLE_GFX_JOIN_OPT
        ? rasterdf::simple_garuda_inner_join_opt(left_key_view.data(),
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
                                              USE_SIMPLE_GFX_JOIN_K);

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
    std::vector<rasterdf::column_view> lk = {left_key_view};
    std::vector<rasterdf::column_view> rk = {right_key_view};
    rasterdf::table_view left_keys_tv(lk);
    rasterdf::table_view right_keys_tv(rk);

    auto join_result = rasterdf::inner_join(
      left_keys_tv, right_keys_tv, _ctx.vk_context(), _ctx.dispatcher(), _ctx.workspace_mr());

    left_indices  = std::move(join_result.first);
    right_indices = std::move(join_result.second);
    match_count   = left_indices ? left_indices->size() : 0;
  }
  RASTERDB_LOG_DEBUG("JOIN: {} matches after first condition", match_count);

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

  auto left_idx_view  = left_indices->view();
  auto right_idx_view = right_indices->view();

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
    scan_total.copy_to_host(&total_chars, sizeof(int32_t), _ctx.device(), _ctx.queue(), _ctx.command_pool());

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

  // Check if any columns are STRING — if so, we can't use rasterdf::gather for them
  bool any_left_string = false, any_right_string = false;
  for (size_t i = 0; i < left_table->num_columns(); i++)
    if (left_table->col(i).is_string()) { any_left_string = true; break; }
  for (size_t i = 0; i < right_table->num_columns(); i++)
    if (right_table->col(i).is_string()) { any_right_string = true; break; }

  // Build result table
  auto result = std::make_unique<gpu_table>();
  result->duckdb_types = op.types;
  size_t total_cols = left_table->num_columns() + right_table->num_columns();
  result->columns.resize(total_cols);

  if (!any_left_string) {
    auto left_gathered = rasterdf::gather(
      left_table->view(), left_idx_view, _ctx.vk_context(), _ctx.dispatcher(), _ctx.workspace_mr());
    auto left_cols = left_gathered->extract();
    for (size_t i = 0; i < left_cols.size(); i++)
      result->columns[i] = gpu_column_from_rdf(std::move(*left_cols[i]));
  } else {
    for (size_t i = 0; i < left_table->num_columns(); i++) {
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

  if (!any_right_string) {
    auto right_gathered = rasterdf::gather(
      right_table->view(), right_idx_view, _ctx.vk_context(), _ctx.dispatcher(), _ctx.workspace_mr());
    auto right_cols = right_gathered->extract();
    for (size_t i = 0; i < right_cols.size(); i++)
      result->columns[left_table->num_columns() + i] = gpu_column_from_rdf(std::move(*right_cols[i]));
  } else {
    for (size_t i = 0; i < right_table->num_columns(); i++) {
      if (right_table->col(i).is_string()) {
        result->columns[left_table->num_columns() + i] =
            gather_string_col(right_table->col(i), right_idx_view, match_count);
      } else {
        auto col_view = right_table->col(i).view();
        std::vector<rasterdf::column_view> cv = {col_view};
        rasterdf::table_view tv(cv);
        auto gathered = rasterdf::gather(tv, right_idx_view, _ctx.vk_context(), _ctx.dispatcher(), _ctx.workspace_mr());
        auto cols = gathered->extract();
        result->columns[left_table->num_columns() + i] = gpu_column_from_rdf(std::move(*cols[0]));
      }
    }
  }

  RASTERDB_LOG_DEBUG("JOIN result: {} rows x {} cols", match_count, total_cols);

  // Post-filter on remaining conditions (multi-condition join)
  size_t num_left_cols = left_table->num_columns();
  for (size_t ci = 1; ci < op.conditions.size(); ci++) {
    auto& cond  = op.conditions[ci];
    auto lk_idx = unwrap_cast(*cond.left).Cast<duckdb::BoundReferenceExpression>().index;
    auto rk_idx = unwrap_cast(*cond.right).Cast<duckdb::BoundReferenceExpression>().index;

    // In the merged table: left cols at [0..num_left-1], right cols at [num_left..]
    auto& left_key_col  = result->col(lk_idx);
    auto& right_key_col = result->col(num_left_cols + rk_idx);

    uint32_t n = static_cast<uint32_t>(result->num_rows());
    auto mask  = allocate_column(_ctx, {rasterdf::type_id::INT32}, n);

    compare_columns_push_constants cpc{};
    cpc.input_a     = left_key_col.address();
    cpc.input_b     = right_key_col.address();
    cpc.output_addr = mask.address();
    cpc.size        = n;
    cpc.op          = 4;                                                    // EQ
    cpc.type_id     = static_cast<int32_t>(rasterdf::ShaderTypeId::INT32);  // INT32
    _ctx.dispatcher().dispatch_compare_columns(cpc);

    result = apply_filter_mask(*result, mask);
    RASTERDB_LOG_DEBUG("JOIN: {} rows after condition {}", result->num_rows(), ci);
  }

  return result;
}

}  // namespace gpu
}  // namespace rasterdb
