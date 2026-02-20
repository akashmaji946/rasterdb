#pragma once

// sirius
#include <cudf_utils.hpp>

// duckdb
#include <duckdb/common/types.hpp>
#include <duckdb/planner/expression/bound_between_expression.hpp>
#include <duckdb/planner/expression/bound_case_expression.hpp>
#include <duckdb/planner/expression/bound_cast_expression.hpp>
#include <duckdb/planner/expression/bound_comparison_expression.hpp>
#include <duckdb/planner/expression/bound_conjunction_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <duckdb/planner/expression/bound_operator_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/joinside.hpp>

// cudf
#include <cudf/ast/ast_operator.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/types.hpp>

// rmmI
#include <rmm/cuda_stream_view.hpp>

// standard library
#include <optional>

namespace sirius {

struct expression_translator {
  // std::optional cannot wrap a real reference, so we use reference_wrapper instead
  using expr_ref = std::reference_wrapper<cudf::ast::expression const>;

  std::optional<cudf::ast::tree> translate_expression(duckdb::Expression const& expr) {}
  std::optional<cudf::ast::tree> translate_join_condition(duckdb::JoinCondition const& condition) {}

  std::optional<expr_ref> add_expression(
    duckdb::Expression const& expr,
    cudf::ast::table_reference const table_src = cudf::ast::table_reference::LEFT)
  {
    switch (expr.GetExpressionClass()) {
      case duckdb::ExpressionClass::BOUND_BETWEEN:
        return add_expression(expr.Cast<duckdb::BoundBetweenExpression>(), table_src);
      case duckdb::ExpressionClass::BOUND_CASE:
        return std::nullopt;  // CASE expressions cannot be translated to cuDF ASTs
      case duckdb::ExpressionClass::BOUND_CAST:
        return add_expression(expr.Cast<duckdb::BoundCastExpression>(), table_src);
      case duckdb::ExpressionClass::BOUND_COMPARISON:
        return add_expression(expr.Cast<duckdb::BoundComparisonExpression>(), table_src);
      case duckdb::ExpressionClass::BOUND_CONJUNCTION:
        return add_expression(expr.Cast<duckdb::BoundConjunctionExpression>(), table_src);
      case duckdb::ExpressionClass::BOUND_CONSTANT:
        return add_expression(expr.Cast<duckdb::BoundConstantExpression>(), table_src);
      case duckdb::ExpressionClass::BOUND_FUNCTION:
        return add_expression(expr.Cast<duckdb::BoundFunctionExpression>(), table_src);
      case duckdb::ExpressionClass::BOUND_OPERATOR:
        return add_expression(expr.Cast<duckdb::BoundOperatorExpression>(), table_src);
      case duckdb::ExpressionClass::BOUND_REF:
        return add_expression(expr.Cast<duckdb::BoundReferenceExpression>(), table_src);
      default: return std::nullopt;  // Unsupported/unexpected expression class
    }
  }
  std::optional<expr_ref> add_expression(duckdb::BoundBetweenExpression const& expr,
                                         cudf::ast::table_reference const table_src)
  {
    // Add the children
    auto input_expr = add_expression(*expr.input, table_src);
    auto lower_expr = add_expression(*expr.lower, table_src);
    auto upper_expr = add_expression(*expr.upper, table_src);

    // Check for failure in translating children
    if (!input_expr || !lower_expr || !upper_expr) { return std::nullopt; }

    // Construct the BETWEEN expression
    auto lower_cmp_op = ast_tree.emplace<cudf::ast::operation>(
      cudf::ast::ast_operator::GREATER_EQUAL, *input_expr, *lower_expr);
    auto upper_cmp_op = ast_tree.emplace<cudf::ast::operation>(
      cudf::ast::ast_operator::LESS_EQUAL, *input_expr, *upper_expr);
    return ast_tree.emplace<cudf::ast::operation>(
      cudf::ast::ast_operator::LOGICAL_AND, lower_cmp_op, upper_cmp_op);
  }
  std::optional<expr_ref> add_expression(duckdb::BoundCastExpression const& expr,
                                         cudf::ast::table_reference const table_src)
  {
    // Add the child
    auto child_expr = add_expression(*expr.child, table_src);

    // Check for failure in translating child
    if (!child_expr) { return std::nullopt; }

    // Construct the CAST expression, if possible
    // CuDF AST only supports casts to INT64, UINT64, FLOAT64
    auto const cudf_return_type = GetCudfType(expr.return_type);
    switch (cudf_return_type.id()) {
      case cudf::type_id::INT64:
        return ast_tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::CAST_TO_INT64,
                                                      *child_expr);
      case cudf::type_id::UINT64:
        return ast_tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::CAST_TO_UINT64,
                                                      *child_expr);
      case cudf::type_id::FLOAT64:
        return ast_tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::CAST_TO_FLOAT64,
                                                      *child_expr);
      default: return std::nullopt;  // Unsupported cast type
    }
  }
  std::optional<expr_ref> add_expression(duckdb::BoundComparisonExpression const& expr,
                                         cudf::ast::table_reference const table_src)
  {
    // Add the children
    auto left_expr  = add_expression(*expr.left, table_src);
    auto right_expr = add_expression(*expr.right, table_src);

    // Check for failure in translating children
    if (!left_expr || !right_expr) { return std::nullopt; }

    // Construct the comparison expression
    switch (expr.GetExpressionType()) {
      case duckdb::ExpressionType::COMPARE_EQUAL:
        return ast_tree.emplace<cudf::ast::operation>(
          cudf::ast::ast_operator::EQUAL, *left_expr, *right_expr);
      case duckdb::ExpressionType::COMPARE_NOTEQUAL:
        return ast_tree.emplace<cudf::ast::operation>(
          cudf::ast::ast_operator::NOT_EQUAL, *left_expr, *right_expr);
      case duckdb::ExpressionType::COMPARE_LESSTHAN:
        return ast_tree.emplace<cudf::ast::operation>(
          cudf::ast::ast_operator::LESS, *left_expr, *right_expr);
      case duckdb::ExpressionType::COMPARE_GREATERTHAN:
        return ast_tree.emplace<cudf::ast::operation>(
          cudf::ast::ast_operator::GREATER, *left_expr, *right_expr);
      case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
        return ast_tree.emplace<cudf::ast::operation>(
          cudf::ast::ast_operator::LESS_EQUAL, *left_expr, *right_expr);
      case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
        return ast_tree.emplace<cudf::ast::operation>(
          cudf::ast::ast_operator::GREATER_EQUAL, *left_expr, *right_expr);
      default: return std::nullopt;  // Unsupported/unexpected comparison type
    }
  }
  std::optional<expr_ref> add_expression(duckdb::BoundConjunctionExpression const& expr,
                                         cudf::ast::table_reference const table_src)
  {
    // If there are no children, return
    if (expr.children.empty()) { return std::nullopt; }

    // Add the children and combine with AND/OR operations as we go
    std::optional<expr_ref> result = add_expression(*expr.children[0], table_src);
    for (size_t i = 1; i < expr.children.size(); ++i) {
      // Add child expression
      auto child_expr = add_expression(*expr.children[i], table_src);

      // Check for failure in translating child
      if (!child_expr) { return std::nullopt; }

      // Combine with previous children using AND/OR
      if (expr.GetExpressionType() == duckdb::ExpressionType::CONJUNCTION_AND) {
        result = ast_tree.emplace<cudf::ast::operation>(
          cudf::ast::ast_operator::LOGICAL_AND, *result, *child_expr);
      } else if (expr.GetExpressionType() == duckdb::ExpressionType::CONJUNCTION_OR) {
        result = ast_tree.emplace<cudf::ast::operation>(
          cudf::ast::ast_operator::LOGICAL_OR, *result, *child_expr);
      } else {
        // Unexpected conjunction type
        return std::nullopt;
      }
    }
    return result;
  }
  std::optional<expr_ref> add_expression(duckdb::BoundConstantExpression const& expr,
                                         cudf::ast::table_reference const table_src)
  {
    auto const cudf_type = GetCudfType(expr.return_type);
    // TODO: Expand type support as needed. See gpu_execute_constant.cpp.
    switch (cudf_type.id()) {
      case cudf::type_id::INT16: {
        cudf::numeric_scalar<int16_t> scalar(
          expr.value.GetValue<int16_t>(), true, stream, resource_ref);
        return ast_tree.emplace<cudf::ast::literal>(scalar);
      }
      case cudf::type_id::INT32: {
        cudf::numeric_scalar<int32_t> scalar(
          expr.value.GetValue<int32_t>(), true, stream, resource_ref);
        return ast_tree.emplace<cudf::ast::literal>(scalar);
      }
      case cudf::type_id::INT64: {
        cudf::numeric_scalar<int64_t> scalar(
          expr.value.GetValue<int64_t>(), true, stream, resource_ref);
        return ast_tree.emplace<cudf::ast::literal>(scalar);
      }
      case cudf::type_id::FLOAT32: {
        cudf::numeric_scalar<float_t> scalar(
          expr.value.GetValue<float_t>(), true, stream, resource_ref);
        return ast_tree.emplace<cudf::ast::literal>(scalar);
      }
      case cudf::type_id::FLOAT64: {
        cudf::numeric_scalar<double_t> scalar(
          expr.value.GetValue<double_t>(), true, stream, resource_ref);
        return ast_tree.emplace<cudf::ast::literal>(scalar);
      }
      case cudf::type_id::BOOL8: {
        cudf::numeric_scalar<bool> scalar(expr.value.GetValue<bool>(), true, stream, resource_ref);
        return ast_tree.emplace<cudf::ast::literal>(scalar);
      }
      case cudf::type_id::STRING: {
        cudf::string_scalar scalar(expr.value.GetValue<std::string>(), true, stream, resource_ref);
        return ast_tree.emplace<cudf::ast::literal>(scalar);
      }
      // cudf decimal type uses negative scale
      case cudf::type_id::DECIMAL32: {
        auto scalar = cudf::fixed_point_scalar<numeric::decimal32>(
          expr.value.GetValueUnsafe<typename numeric::decimal32::rep>(),
          numeric::scale_type{-duckdb::DecimalType::GetScale(expr.value.type())},
          true,
          stream,
          resource_ref);
        return ast_tree.emplace<cudf::ast::literal>(scalar);
      }
      case cudf::type_id::DECIMAL64: {
        auto scalar = cudf::fixed_point_scalar<numeric::decimal64>(
          expr.value.GetValueUnsafe<typename numeric::decimal64::rep>(),
          numeric::scale_type{-duckdb::DecimalType::GetScale(expr.value.type())},
          true,
          stream,
          resource_ref);
        return ast_tree.emplace<cudf::ast::literal>(scalar);
      }
      default: return std::nullopt;  // Unsupported constant type
    }
  }
  std::optional<expr_ref> add_expression(duckdb::BoundFunctionExpression const& expr,
                                         cudf::ast::table_reference const table_src)
  {
    /// Can do numeric binary ops
    return std::nullopt;
  }
  std::optional<expr_ref> add_expression(duckdb::BoundOperatorExpression const& expr,
                                         cudf::ast::table_reference const table_src)
  {
    switch (expr.type) {                            // COALESCE is not supported
      case duckdb::ExpressionType::COMPARE_IN:      /// TODO
      case duckdb::ExpressionType::COMPARE_NOT_IN:  /// TODO (use fallthrough)
      case duckdb::ExpressionType::OPERATOR_NOT: {
        // Add the child
        auto child_expr = add_expression(*expr.children[0], table_src);

        // Check for failure in translating child
        if (!child_expr) { return std::nullopt; }

        // Add the operator expression
        return ast_tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::NOT, *child_expr);
      }
      case duckdb::ExpressionType::OPERATOR_IS_NULL: {
        // Add the child
        auto child_expr = add_expression(*expr.children[0], table_src);

        // Check for failure in translating child
        if (!child_expr) { return std::nullopt; }

        // Add the operator expression
        return ast_tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::IS_NULL,
                                                      *child_expr);
      }
      case duckdb::ExpressionType::OPERATOR_IS_NOT_NULL: {
        // Add the child
        auto child_expr = add_expression(*expr.children[0], table_src);

        // Check for failure in translating child
        if (!child_expr) { return std::nullopt; }

        // Add IS_NULL followed by NOT to represent IS_NOT_NULL
        auto is_null_op =
          ast_tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::IS_NULL, *child_expr);
        return ast_tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::NOT, is_null_op);
      }
      default: return std::nullopt;  // Unsupported/unexpected operator type
    }
    return std::nullopt;
  }
  std::optional<expr_ref> add_expression(duckdb::BoundReferenceExpression const& expr,
                                         cudf::ast::table_reference const table_src)
  {
    return ast_tree.emplace<cudf::ast::column_reference>(expr.index, table_src);
  }

  cudf::ast::tree ast_tree;
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref resource_ref;
};

}  // namespace sirius