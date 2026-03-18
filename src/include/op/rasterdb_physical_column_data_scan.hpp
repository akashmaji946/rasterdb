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

#include "duckdb/common/optionally_owned_ptr.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "op/rasterdb_physical_operator.hpp"

namespace rasterdb {

namespace pipeline {
class rasterdb_pipeline;
class rasterdb_meta_pipeline;
}  // namespace pipeline

namespace op {

//! The rasterdb_physical_column_data_scan scans a ColumnDataCollection
class rasterdb_physical_column_data_scan : public rasterdb_physical_operator {
 public:
  static constexpr const RasterDBPhysicalOperatorType TYPE = RasterDBPhysicalOperatorType::INVALID;

 public:
  rasterdb_physical_column_data_scan(
    duckdb::vector<duckdb::LogicalType> types,
    RasterDBPhysicalOperatorType op_type,
    duckdb::idx_t estimated_cardinality,
    duckdb::optionally_owned_ptr<duckdb::ColumnDataCollection> collection);

  rasterdb_physical_column_data_scan(duckdb::vector<duckdb::LogicalType> types,
                                   RasterDBPhysicalOperatorType op_type,
                                   duckdb::idx_t estimated_cardinality,
                                   duckdb::idx_t cte_index);

  //! (optionally owned) column data collection to scan
  duckdb::optionally_owned_ptr<duckdb::ColumnDataCollection> collection;

  duckdb::idx_t cte_index;

  duckdb::optional_idx delim_index;

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

 public:
  bool is_source() const override { return true; }

  void build_pipelines(pipeline::rasterdb_pipeline& current,
                       pipeline::rasterdb_meta_pipeline& meta_pipeline) override;
};

}  // namespace op
}  // namespace rasterdb
