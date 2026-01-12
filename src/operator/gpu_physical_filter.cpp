/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "operator/gpu_physical_filter.hpp"

#include "duckdb/planner/expression/bound_between_expression.hpp"
#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "expression_executor/gpu_expression_executor.hpp"
#include "gpu_expression_executor.hpp"
#include "log/logging.hpp"

namespace duckdb {

GPUPhysicalFilter::GPUPhysicalFilter(vector<LogicalType> types,
                                     vector<unique_ptr<Expression>> select_list,
                                     idx_t estimated_cardinality)
  : GPUPhysicalOperator(PhysicalOperatorType::FILTER, std::move(types), estimated_cardinality)
{
  D_ASSERT(select_list.size() > 0);
  if (select_list.size() > 1) {
    // KEVIN: I don't think this code path is ever entered
    // create a big AND out of the expressions
    auto conjunction = make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND);
    for (auto& expr : select_list) {
      conjunction->children.push_back(std::move(expr));
    }
    expression = std::move(conjunction);
  } else {
    expression = std::move(select_list[0]);
  }
  gpu_expression_executor = new GPUExpressionExecutor();
}

OperatorResultType GPUPhysicalFilter::Execute(GPUIntermediateRelation& input_relation,
                                              GPUIntermediateRelation& output_relation) const
{
  SIRIUS_LOG_DEBUG("Executing expression {}", expression->ToString());
  auto start = std::chrono::high_resolution_clock::now();

  bool use_new_executor = true;
  for (int i = 0; i < input_relation.columns.size(); i++) {
    if (input_relation.columns[i]->row_ids != nullptr) {
      if (input_relation.columns[i]->row_id_count > INT32_MAX) {
        use_new_executor = false;
        break;
      }
    } else {
      if (input_relation.columns[i]->column_length > INT32_MAX) {
        use_new_executor = false;
        break;
      }
    }
  }

  if (use_new_executor) {
    // The new executor...
    SIRIUS_LOG_DEBUG("Using new executor");
    sirius::GpuExpressionExecutor new_gpu_expression_executor(*expression.get());
    new_gpu_expression_executor.Select(input_relation, output_relation);
  } else {
    // The old executor...
    SIRIUS_LOG_DEBUG("Using old executor");
    gpu_expression_executor->FilterRecursiveExpression(
      input_relation, output_relation, *expression, 0);
  }

  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  SIRIUS_LOG_DEBUG("Filter time: {:.2f} ms", duration.count() / 1000.0);
  return OperatorResultType::FINISHED;
}

}  // namespace duckdb
