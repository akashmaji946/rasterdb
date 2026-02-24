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

#include "memory/sirius_memory_reservation_manager.hpp"
#include "operator_test_utils.hpp"
#include "operator_type_traits.hpp"

#include <catch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <op/sirius_physical_hash_join.hpp>

using namespace duckdb;
using namespace sirius::op;
using namespace cucascade;
using namespace cucascade::memory;

namespace {
using namespace sirius::test::operator_utils;
}  // namespace

TEST_CASE("sirius_physical_hash_join mark join outputs all left rows with bool mark column",
          "[physical_mark_join]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space);

  // Left table: col[0]=id (join key), col[1]=payload
  // id:      {10, 20, 30, 40, 50}
  // payload: { 1,  2,  3,  4,  5}
  std::vector<int32_t> left_ids     = {10, 20, 30, 40, 50};
  std::vector<int32_t> left_payload = {1, 2, 3, 4, 5};
  auto left_batch                   = make_two_column_batch<int32_t, int32_t>(
    *space, left_ids, left_payload, cudf::type_id::INT32, std::nullopt, cudf::type_id::INT32);

  // Right table: col[0]=id (join key) — only {20, 40} exist on right
  std::vector<int32_t> right_ids = {20, 40};
  auto right_batch = make_numeric_batch<int32_t>(*space, right_ids, cudf::type_id::INT32);

  // Stub child operators — types must match the actual batch columns above
  auto left_child = duckdb::make_uniq<sirius_physical_operator>(
    SiriusPhysicalOperatorType::PROJECTION,
    duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER, duckdb::LogicalType::INTEGER},
    0);
  auto right_child = duckdb::make_uniq<sirius_physical_operator>(
    SiriusPhysicalOperatorType::PROJECTION,
    duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER},
    0);

  // Join condition: left.col[0] == right.col[0]
  duckdb::vector<duckdb::JoinCondition> conditions;
  duckdb::JoinCondition cond;
  cond.left       = duckdb::make_uniq<BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0);
  cond.right      = duckdb::make_uniq<BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0);
  cond.comparison = duckdb::ExpressionType::COMPARE_EQUAL;
  conditions.push_back(std::move(cond));

  auto logical_join   = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(duckdb::JoinType::MARK);
  logical_join->types = {
    duckdb::LogicalType::INTEGER, duckdb::LogicalType::INTEGER, duckdb::LogicalType::BOOLEAN};

  sirius_physical_hash_join mark_join(
    *logical_join,
    std::move(left_child),
    std::move(right_child),
    std::move(conditions),
    duckdb::JoinType::MARK,
    duckdb::vector<duckdb::idx_t>{},        // left_projection_map (empty = all)
    duckdb::vector<duckdb::idx_t>{},        // right_projection_map (not used by MARK)
    duckdb::vector<duckdb::LogicalType>{},  // delim_types
    1000,
    nullptr);

  // Execute: batch[0]=left, batch[1]=right
  std::vector<std::shared_ptr<cucascade::data_batch>> inputs{left_batch, right_batch};
  auto outputs = mark_join.execute(operator_data(inputs), cudf::get_default_stream());

  REQUIRE(outputs->get_data_batches().size() == 1);
  auto out_view =
    outputs->get_data_batches()[0]->get_data()->cast<gpu_table_representation>().get_table().view();

  // Output: col[0]=id, col[1]=payload, col[2]=mark(bool)
  REQUIRE(out_view.num_columns() == 3);
  REQUIRE(out_view.num_rows() == static_cast<cudf::size_type>(left_ids.size()));

  auto host_ids     = copy_column_to_host<int32_t>(out_view.column(0));
  auto host_payload = copy_column_to_host<int32_t>(out_view.column(1));
  auto host_mark    = copy_column_to_host<bool>(out_view.column(2));

  // All left rows pass through unchanged
  REQUIRE(host_ids == left_ids);
  REQUIRE(host_payload == left_payload);
  // Mark is true only for rows whose id matched the right side ({20, 40} → rows 1 and 3)
  REQUIRE(host_mark == std::vector<bool>{false, true, false, true, false});
}
