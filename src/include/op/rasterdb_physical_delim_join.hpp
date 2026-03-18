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
#include "op/rasterdb_physical_column_data_scan.hpp"
#include "op/rasterdb_physical_operator.hpp"
#include "op/rasterdb_physical_partition.hpp"

namespace rasterdb {

namespace pipeline {
class rasterdb_pipeline;
class rasterdb_meta_pipeline;
}  // namespace pipeline

namespace op {

class rasterdb_physical_grouped_aggregate;

//! rasterdb_physical_delim_join represents a join where either the LHS or RHS will be duplicate
//! eliminated and pushed into a PhysicalColumnDataScan in the other side. Implementations are
//! rasterdb_physical_left_delim_join and rasterdb_physical_right_delim_join
class rasterdb_physical_delim_join : public rasterdb_physical_operator {
 public:
  rasterdb_physical_delim_join(
    RasterDBPhysicalOperatorType type,
    duckdb::vector<duckdb::LogicalType> types,
    duckdb::unique_ptr<rasterdb_physical_operator> original_join,
    duckdb::vector<duckdb::const_reference<rasterdb_physical_operator>> delim_scans,
    duckdb::idx_t estimated_cardinality,
    duckdb::optional_idx delim_idx);

  duckdb::unique_ptr<rasterdb_physical_operator> join;
  duckdb::unique_ptr<rasterdb_physical_grouped_aggregate> distinct;
  duckdb::vector<duckdb::const_reference<rasterdb_physical_operator>> delim_scans;

  duckdb::optional_idx delim_idx;

 public:
  // std::vector<duckdb::const_reference<rasterdb_physical_operator>> get_children() const override;

  bool is_sink() const override { return true; }

  duckdb::OrderPreservationType source_order() const override
  {
    return duckdb::OrderPreservationType::NO_ORDER;
  }
  bool sink_order_dependent() const override { return false; }
};

//! rasterdb_physical_right_delim_join represents a join where the RHS will be duplicate eliminated
//! and pushed into a PhysicalColumnDataScan in the LHS.
class rasterdb_physical_right_delim_join : public rasterdb_physical_delim_join {
 public:
  static constexpr const RasterDBPhysicalOperatorType TYPE =
    RasterDBPhysicalOperatorType::RIGHT_DELIM_JOIN;

 public:
  rasterdb_physical_right_delim_join(
    duckdb::vector<duckdb::LogicalType> types,
    duckdb::unique_ptr<rasterdb_physical_operator> original_join,
    duckdb::vector<duckdb::const_reference<rasterdb_physical_operator>> delim_scans,
    duckdb::idx_t estimated_cardinality,
    duckdb::optional_idx delim_idx);
  rasterdb_physical_partition* partition_join;

 public:
  void build_pipelines(pipeline::rasterdb_pipeline& current,
                       pipeline::rasterdb_meta_pipeline& meta_pipeline) override;

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  void sink(const operator_data& input_data, rmm::cuda_stream_view stream) override;

  std::unique_ptr<operator_data> get_next_task_input_data() override;
};

class rasterdb_physical_left_delim_join : public rasterdb_physical_delim_join {
 public:
  static constexpr const RasterDBPhysicalOperatorType TYPE =
    RasterDBPhysicalOperatorType::LEFT_DELIM_JOIN;

 public:
  rasterdb_physical_left_delim_join(
    duckdb::vector<duckdb::LogicalType> types,
    duckdb::unique_ptr<rasterdb_physical_operator> original_join,
    duckdb::vector<duckdb::const_reference<rasterdb_physical_operator>> delim_scans,
    duckdb::idx_t estimated_cardinality,
    duckdb::optional_idx delim_idx);
  rasterdb_physical_column_data_scan* column_data_scan;

 public:
  void build_pipelines(pipeline::rasterdb_pipeline& current,
                       pipeline::rasterdb_meta_pipeline& meta_pipeline) override;

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  void sink(const operator_data& input_data, rmm::cuda_stream_view stream) override;
};

}  // namespace op
}  // namespace rasterdb
