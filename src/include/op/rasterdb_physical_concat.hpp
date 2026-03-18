/*
 * Copyright 2025, RasterDB Contributors.
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

#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "op/rasterdb_physical_grouped_aggregate.hpp"
#include "op/rasterdb_physical_hash_join.hpp"
#include "op/rasterdb_physical_operator.hpp"
#include "op/rasterdb_physical_order.hpp"
#include "op/rasterdb_physical_partition_consumer_operator.hpp"
#include "op/rasterdb_physical_top_n.hpp"
#include "rasterdb_config.hpp"

namespace rasterdb {
namespace op {

class rasterdb_physical_concat : public rasterdb_physical_partition_consumer_operator {
 public:
  static constexpr const RasterDBPhysicalOperatorType TYPE = RasterDBPhysicalOperatorType::CONCAT;

  explicit rasterdb_physical_concat(
    duckdb::vector<duckdb::LogicalType> types,
    duckdb::idx_t estimated_cardinality,
    rasterdb_physical_operator* parent_op,
    bool is_build,
    uint64_t concat_batch_bytes = rasterdb::config::DEFAULT_CONCAT_BATCH_BYTES);

  std::string get_name() const override;

  bool is_source() const override;

  bool is_sink() const override;

  bool is_build_concat();

  std::optional<task_creation_hint> get_next_task_hint() override;

  std::unique_ptr<operator_data> get_next_task_input_data() override;

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  void sink(const operator_data& output_data, rmm::cuda_stream_view stream) override;

  //! Get the parent operator (e.g., HASH_JOIN for build concat)
  rasterdb_physical_operator* get_parent_op() const { return _parent_op; }

 private:
  rasterdb_physical_operator* _parent_op;
  bool _is_build;
  bool _concat_all;
  uint64_t _concat_batch_bytes;
};

}  // namespace op
}  // namespace rasterdb
