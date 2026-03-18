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

#include "op/rasterdb_physical_delim_join.hpp"

#include "op/rasterdb_physical_column_data_scan.hpp"
#include "op/rasterdb_physical_dummy_scan.hpp"
#include "op/rasterdb_physical_grouped_aggregate.hpp"
#include "op/rasterdb_physical_hash_join.hpp"
#include "pipeline/rasterdb_meta_pipeline.hpp"
#include "pipeline/rasterdb_pipeline.hpp"

#include <nvtx3/nvtx3.hpp>

namespace rasterdb {
namespace op {

class sirius_left_delim_join_local_state : public duckdb::LocalSinkState {
 public:
  duckdb::unique_ptr<duckdb::LocalSinkState> distinct_state;
  // duckdb::shared_ptr<GPUIntermediateRelation> lhs_data;
  duckdb::ColumnDataAppendState append_state;
};

class sirius_right_delim_join_local_state : public duckdb::LocalSinkState {
 public:
  duckdb::unique_ptr<duckdb::LocalSinkState> join_state;
  duckdb::unique_ptr<duckdb::LocalSinkState> distinct_state;
};

rasterdb_physical_delim_join::rasterdb_physical_delim_join(
  RasterDBPhysicalOperatorType type,
  duckdb::vector<duckdb::LogicalType> types,
  duckdb::unique_ptr<rasterdb_physical_operator> original_join,
  duckdb::vector<duckdb::const_reference<rasterdb_physical_operator>> delim_scans,
  duckdb::idx_t estimated_cardinality,
  duckdb::optional_idx delim_idx)
  : rasterdb_physical_operator(type, std::move(types), estimated_cardinality),
    join(std::move(original_join)),
    delim_scans(std::move(delim_scans))
{
  D_ASSERT(type == RasterDBPhysicalOperatorType::LEFT_DELIM_JOIN ||
           type == RasterDBPhysicalOperatorType::RIGHT_DELIM_JOIN);
}

rasterdb_physical_right_delim_join::rasterdb_physical_right_delim_join(
  duckdb::vector<duckdb::LogicalType> types,
  duckdb::unique_ptr<rasterdb_physical_operator> original_join,
  duckdb::vector<duckdb::const_reference<rasterdb_physical_operator>> delim_scans,
  duckdb::idx_t estimated_cardinality,
  duckdb::optional_idx delim_idx)
  : rasterdb_physical_delim_join(RasterDBPhysicalOperatorType::RIGHT_DELIM_JOIN,
                               std::move(types),
                               std::move(original_join),
                               std::move(delim_scans),
                               estimated_cardinality,
                               delim_idx)
{
  D_ASSERT(join->children.size() == 2);
  children.push_back(std::move(join->children[1]));

  join->children[1] =
    duckdb::make_uniq<rasterdb_physical_dummy_scan>(children[0]->get_types(), estimated_cardinality);
}

rasterdb_physical_left_delim_join::rasterdb_physical_left_delim_join(
  duckdb::vector<duckdb::LogicalType> types,
  duckdb::unique_ptr<rasterdb_physical_operator> original_join,
  duckdb::vector<duckdb::const_reference<rasterdb_physical_operator>> delim_scans,
  duckdb::idx_t estimated_cardinality,
  duckdb::optional_idx delim_idx)
  : rasterdb_physical_delim_join(RasterDBPhysicalOperatorType::LEFT_DELIM_JOIN,
                               std::move(types),
                               std::move(original_join),
                               std::move(delim_scans),
                               estimated_cardinality,
                               delim_idx)
{
  D_ASSERT(join->children.size() == 2);
  children.push_back(std::move(join->children[0]));

  auto cached_chunk_scan = duckdb::make_uniq<rasterdb_physical_column_data_scan>(
    children[0]->get_types(),
    RasterDBPhysicalOperatorType::COLUMN_DATA_SCAN,
    estimated_cardinality,
    nullptr);
  if (delim_idx.IsValid()) { cached_chunk_scan->cte_index = delim_idx.GetIndex(); }
  join->children[0] = std::move(cached_chunk_scan);
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void rasterdb_physical_left_delim_join::build_pipelines(pipeline::rasterdb_pipeline& current,
                                                      pipeline::rasterdb_meta_pipeline& meta_pipeline)
{
  op_state.reset();
  sink_state.reset();

  auto& child_meta_pipeline = meta_pipeline.create_child_meta_pipeline(current, *this);
  child_meta_pipeline.build(*children[0]);

  D_ASSERT(type == RasterDBPhysicalOperatorType::LEFT_DELIM_JOIN);
  // recurse into the actual join
  // any pipelines in there depend on the main pipeline
  // any scan of the duplicate eliminated data on the RHS depends on this pipeline
  // we add an entry to the mapping of (PhysicalOperator*) -> (Pipeline*)
  auto& state = meta_pipeline.get_state();
  for (auto& delim_scan : delim_scans) {
    state.delim_join_dependencies.insert(duckdb::make_pair(
      delim_scan,
      duckdb::reference<pipeline::rasterdb_pipeline>(*child_meta_pipeline.get_base_pipeline())));
  }
  join->build_pipelines(current, meta_pipeline);
}

void rasterdb_physical_right_delim_join::build_pipelines(
  pipeline::rasterdb_pipeline& current, pipeline::rasterdb_meta_pipeline& meta_pipeline)
{
  op_state.reset();
  sink_state.reset();

  auto& child_meta_pipeline = meta_pipeline.create_child_meta_pipeline(current, *this);
  child_meta_pipeline.build(*children[0]);

  D_ASSERT(type == RasterDBPhysicalOperatorType::RIGHT_DELIM_JOIN);
  // recurse into the actual join
  // any pipelines in there depend on the main pipeline
  // any scan of the duplicate eliminated data on the LHS depends on this pipeline
  // we add an entry to the mapping of (PhysicalOperator*) -> (Pipeline*)
  auto& state = meta_pipeline.get_state();
  for (auto& delim_scan : delim_scans) {
    state.delim_join_dependencies.insert(duckdb::make_pair(
      delim_scan,
      duckdb::reference<pipeline::rasterdb_pipeline>(*child_meta_pipeline.get_base_pipeline())));
  }

  rasterdb_physical_hash_join::build_join_pipelines(current, meta_pipeline, *join, false);
}

std::unique_ptr<operator_data> rasterdb_physical_right_delim_join::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"rasterdb_physical_right_delim_join::execute"};
  return std::make_unique<operator_data>(input_data);
}

void rasterdb_physical_right_delim_join::sink(const operator_data& input_data,
                                            rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"rasterdb_physical_right_delim_join::sink"};
  // partition_join stays inline (still part of the delim join)
  auto partition_join_output = partition_join->execute(input_data, stream);
  // distinct stays inline (still part of the delim join)
  auto distinct_output = distinct->execute(input_data, stream);

  stream.synchronize();

  partition_join->sink(*partition_join_output, stream);
  // partition_distinct is external — push distinct output via distinct's next_port_after_sink
  distinct->sink(*distinct_output, stream);
}

std::unique_ptr<operator_data> rasterdb_physical_right_delim_join::get_next_task_input_data()
{
  return partition_join->get_next_task_input_data();
}

std::unique_ptr<operator_data> rasterdb_physical_left_delim_join::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"rasterdb_physical_left_delim_join::execute"};
  return std::make_unique<operator_data>(input_data);
}

void rasterdb_physical_left_delim_join::sink(const operator_data& input_data,
                                           rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"rasterdb_physical_left_delim_join::sink"};
  // column_data_scan stays inline (still part of the delim join)
  auto column_data_scan_output = column_data_scan->execute(input_data, stream);
  // distinct stays inline (still part of the delim join)
  auto distinct_output = distinct->execute(input_data, stream);

  stream.synchronize();

  column_data_scan->sink(*column_data_scan_output, stream);
  // partition_distinct is external — push distinct output via distinct's next_port_after_sink
  distinct->sink(*distinct_output, stream);
}

}  // namespace op
}  // namespace rasterdb
