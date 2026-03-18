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

#include "duckdb/common/common.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/common/pair.hpp"
#include "duckdb/common/reference_map.hpp"
#include "duckdb/execution/task_error_manager.hpp"
#include "op/rasterdb_physical_operator.hpp"
#include "op/rasterdb_physical_result_collector.hpp"
#include "pipeline/rasterdb_meta_pipeline.hpp"
#include "pipeline/rasterdb_pipeline.hpp"

#include <cucascade/data/data_repository_manager.hpp>
namespace duckdb {
class ClientContext;
}  // namespace duckdb

namespace rasterdb {

class rasterdb_interface;

class rasterdb_engine {
  friend class pipeline::rasterdb_pipeline_build_state;
  friend class pipeline::rasterdb_pipeline;
  friend class pipeline::rasterdb_meta_pipeline;

 public:
  explicit rasterdb_engine(duckdb::ClientContext& context, rasterdb_interface& sirius_iface)
    : context(context), sirius_iface(sirius_iface) {};
  ~rasterdb_engine() {}

  duckdb::ClientContext& context;
  rasterdb_interface& sirius_iface;
  duckdb::unique_ptr<op::rasterdb_physical_operator> rasterdb_owned_plan;
  duckdb::optional_ptr<op::rasterdb_physical_operator> rasterdb_physical_plan;

  //! All pipelines of the query plan
  duckdb::vector<duckdb::shared_ptr<pipeline::rasterdb_pipeline>> rasterdb_pipelines;
  //! The root pipelines of the query
  duckdb::vector<duckdb::shared_ptr<pipeline::rasterdb_pipeline>> rasterdb_root_pipelines;
  //! The scheduled pipelines
  duckdb::vector<duckdb::shared_ptr<pipeline::rasterdb_pipeline>> rasterdb_scheduled;
  //! Storage for pipeline breaker created during pipeline splitting
  duckdb::vector<duckdb::unique_ptr<op::rasterdb_physical_operator>> new_pipeline_breakers;
  //! The current root pipeline index
  duckdb::idx_t root_pipeline_idx;
  //! The total amount of pipelines in the query
  duckdb::idx_t total_pipelines;
  //! Insert the repository
  void insert_repository(std::string_view port_id,
                         duckdb::shared_ptr<pipeline::rasterdb_pipeline> input_pipeline,
                         duckdb::shared_ptr<pipeline::rasterdb_pipeline> dependent_pipeline,
                         op::MemoryBarrierType barrier_type = op::MemoryBarrierType::FULL);
  //! Insert the repository
  void insert_repository(std::string_view port_id,
                         op::rasterdb_physical_operator* cur_op,
                         duckdb::shared_ptr<pipeline::rasterdb_pipeline> input_pipeline,
                         duckdb::shared_ptr<pipeline::rasterdb_pipeline> dependent_pipeline,
                         op::MemoryBarrierType barrier_type = op::MemoryBarrierType::FULL);
  //! Whether or not the root of the pipeline is a result collector object
  bool has_result_collector();
  //! Returns the query result - can only be used if `HasResultCollector` returns true
  duckdb::unique_ptr<duckdb::QueryResult> get_result();
  //! Initialize the sirius engine
  void initialize(duckdb::unique_ptr<op::rasterdb_physical_operator> physical_plan);
  //! Initialize the sirius engine internally
  void initialize_internal(op::rasterdb_physical_operator& physical_result_collector);
  //! Execute the sirius engine
  void execute();
  //! Reset the sirius engine
  void reset();
  //! Cancel the tasks
  void cancel_tasks();
  //! Construct the sirius specific operator
  duckdb::unique_ptr<op::rasterdb_physical_operator> construct_sirius_specific_operator(
    op::rasterdb_physical_operator* op);
  //! Create a child pipeline
  duckdb::shared_ptr<pipeline::rasterdb_pipeline> create_child_pipeline(
    pipeline::rasterdb_pipeline& current, op::rasterdb_physical_operator& op);
  duckdb::vector<duckdb::shared_ptr<pipeline::rasterdb_pipeline>> new_scheduled;
  //! Wait for the query to finish
  void wait_for_query_finish();
  //! Mutex for thread-safe access to query finish
  std::mutex query_finish_mutex;
  //! Condition variable for thread-safe access to query finish
  std::condition_variable query_finish_cv;
  //! Whether the query has finished
  bool query_finished;
};

}  // namespace rasterdb
