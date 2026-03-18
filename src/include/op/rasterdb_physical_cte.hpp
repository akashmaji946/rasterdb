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

#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "op/rasterdb_physical_operator.hpp"

namespace rasterdb {

namespace pipeline {
class rasterdb_pipeline;
class rasterdb_meta_pipeline;
}  // namespace pipeline

namespace op {

// class recursive_cte_state;

class rasterdb_physical_cte : public rasterdb_physical_operator {
 public:
  static constexpr const RasterDBPhysicalOperatorType TYPE = RasterDBPhysicalOperatorType::CTE;

 public:
  rasterdb_physical_cte(std::string ctename,
                      duckdb::idx_t table_index,
                      duckdb::vector<duckdb::LogicalType> types,
                      duckdb::unique_ptr<rasterdb_physical_operator> top,
                      duckdb::unique_ptr<rasterdb_physical_operator> bottom,
                      duckdb::idx_t estimated_cardinality);
  ~rasterdb_physical_cte() override;

  duckdb::vector<duckdb::const_reference<rasterdb_physical_operator>> cte_scans;

  duckdb::shared_ptr<duckdb::ColumnDataCollection> working_table;

  duckdb::idx_t table_index;
  std::string ctename;

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

 public:
  // Sink interface
  bool is_sink() const override { return true; }

  bool sink_order_dependent() const override { return false; }

 public:
  void build_pipelines(pipeline::rasterdb_pipeline& current,
                       pipeline::rasterdb_meta_pipeline& meta_pipeline) override;

  duckdb::vector<duckdb::const_reference<rasterdb_physical_operator>> get_sources() const override;
};

}  // namespace op
}  // namespace rasterdb
