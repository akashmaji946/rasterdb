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

#include "operator/gpu_physical_projection.hpp"

#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "expression_executor/gpu_expression_executor.hpp"
#include "gpu_expression_executor.hpp"
#include "log/logging.hpp"

namespace duckdb {

GPUPhysicalProjection::GPUPhysicalProjection(vector<LogicalType> types,
                                             vector<unique_ptr<Expression>> select_list,
                                             idx_t estimated_cardinality)
  : GPUPhysicalOperator(PhysicalOperatorType::PROJECTION, std::move(types), estimated_cardinality),
    select_list(std::move(select_list))
{
  gpu_expression_executor = new GPUExpressionExecutor();
}

OperatorResultType GPUPhysicalProjection::Execute(GPUIntermediateRelation& input_relation,
                                                  GPUIntermediateRelation& output_relation) const
{
  SIRIUS_LOG_DEBUG("Executing projection");
  auto start = std::chrono::high_resolution_clock::now();

  // check if input_relation has column size larger than INT32 MAX

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

  // The new executor...
  if (use_new_executor) {
      sirius::GpuExpressionExecutor new_gpu_expression_executor(select_list);
      new_gpu_expression_executor.Execute(input_relation, output_relation);
  } else {
      for (int idx = 0; idx < select_list.size(); idx++) {
            SIRIUS_LOG_DEBUG("Executing old executor expression: {}", select_list[idx]->ToString());
            gpu_expression_executor->ProjectionRecursiveExpression(input_relation, output_relation, *select_list[idx], idx, 0);
      }
  }

  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  SIRIUS_LOG_DEBUG("Projection time: {:.2f} ms", duration.count() / 1000.0);
  return OperatorResultType::FINISHED;
}

}  // namespace duckdb
