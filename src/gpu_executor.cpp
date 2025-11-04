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

#include "gpu_context.hpp"
#include "duckdb/execution/operator/set/physical_recursive_cte.hpp"
#include "duckdb/execution/operator/helper/physical_result_collector.hpp"
#include "config.hpp"
#include "fallback.hpp"
#include "gpu_physical_operator.hpp"
#include "operator/gpu_physical_result_collector.hpp"
#include "operator/gpu_physical_hash_join.hpp"
#include "operator/gpu_physical_dummy_source.hpp"
#include "operator/gpu_physical_dummy_sink.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "duckdb/execution/execution_context.hpp"
#include "operator/gpu_physical_table_scan.hpp"
#include "log/logging.hpp"
#include <iostream>
#include <stdio.h>

namespace duckdb {

void 
GPUExecutor::Reset() {
	// lock_guard<mutex> elock(executor_lock);
	gpu_physical_plan = nullptr;
	// cancelled = false;
	gpu_owned_plan.reset();
	// root_executor.reset();
	root_pipelines.clear();
	root_pipeline_idx = 0;
	completed_pipelines = 0;
	total_pipelines = 0;
	// error_manager.Reset();
	pipelines.clear();
	dummy_operators.clear();
	// gpuBufferManager->ResetBuffer();
	// events.clear();
	// to_be_rescheduled_tasks.clear();
	// execution_result = PendingExecutionResult::RESULT_NOT_READY;
}

void GPUExecutor::Initialize(unique_ptr<GPUPhysicalOperator> plan) {
	SIRIUS_LOG_DEBUG("Initializing GPUExecutor");
	Reset();
	gpu_owned_plan = std::move(plan);
	InitializeInternal(*gpu_owned_plan);
}

void GPUExecutor::Execute() {
	// Check if we should fall back to duckdb execution.
	if (Config::ENABLE_FALLBACK_CHECK) {
		FallbackChecker fallback_checker(scheduled);
		fallback_checker.Check();
	}

	// Execution starts here.
	int initial_idx = 0;

	SIRIUS_LOG_DEBUG("Total meta pipelines {}", scheduled.size());

	for (int pipeline_idx = 0; pipeline_idx < scheduled.size(); pipeline_idx++) {
		auto pipeline = scheduled[pipeline_idx];
		SIRIUS_LOG_DEBUG("Executing pipeline {}", pipeline_idx);

		// TODO: This is temporary solution
		// if (pipeline->source->type == PhysicalOperatorType::HASH_JOIN || pipeline->source->type == PhysicalOperatorType::RESULT_COLLECTOR) {
		// 	continue;
		// }

		printf("Executing pipeline %d\n", pipeline_idx);

		vector<shared_ptr<GPUIntermediateRelation>> intermediate_relations;
		shared_ptr<GPUIntermediateRelation> final_relation;
		// vector<unique_ptr<OperatorState>> intermediate_states;
		intermediate_relations.reserve(pipeline->operators.size());
		// intermediate_states.reserve(pipeline->operators.size());

		// SIRIUS_LOG_DEBUG("Executing pipeline op size {}", pipeline->operators.size());
		for (idx_t i = 0; i < pipeline->operators.size(); i++) {
			auto &prev_operator = i == 0 ? *(pipeline->source) : pipeline->operators[i - 1].get();
			auto &current_operator = pipeline->operators[i].get();

			// auto chunk = make_uniq<DataChunk>();
			// chunk->Initialize(Allocator::Get(context.client), prev_operator.GetTypes());
			shared_ptr<GPUIntermediateRelation> inter_rel = make_shared_ptr<GPUIntermediateRelation>(prev_operator.GetTypes().size());
			intermediate_relations.push_back(std::move(inter_rel));

			// auto op_state = current_operator.GetOperatorState(context);
			// intermediate_states.push_back(std::move(op_state));

			// if (current_operator.IsSink() && current_operator.sink_state->state == SinkFinalizeType::NO_OUTPUT_POSSIBLE) {
			// 	// one of the operators has already figured out no output is possible
			// 	// we can skip executing the pipeline
			// 	FinishProcessing();
			// }
		}
		// InitializeChunk(final_chunk);
		auto &last_op = pipeline->operators.empty() ? *pipeline->source : pipeline->operators.back().get();
		final_relation = make_shared_ptr<GPUIntermediateRelation>(last_op.GetTypes().size());

		// auto thread_context = ThreadContext(context);
		// auto exec_context = GPUExecutionContext(context, thread_context, pipeline.get());

		// pipeline->Reset();
		// auto prop = pipeline->executor.context.GetClientProperties();
		// SIRIUS_LOG_DEBUG("Properties: {}", prop.time_zone);
		auto is_empty = pipeline->operators.empty();
		auto &source_relation = is_empty ? final_relation : intermediate_relations[0];
		// auto source_result = FetchFromSource(source_chunk);

		// StartOperator(*pipeline.source);
		// auto interrupt_state = InterruptState();
		// auto local_source_state = pipeline.source->GetLocalSourceState(exec_context, *pipeline.source_state);
		// OperatorSourceInput source_input = {*pipeline.source_state, *local_source_state, interrupt_state};
		// pipeline->source->GetData(exec_context, source_relation, source_input);
		auto source_type = pipeline->source.get()->type;
		SIRIUS_LOG_DEBUG("pipeline source type {}", PhysicalOperatorToString(source_type));
		printf("pipeline source type %s\n", PhysicalOperatorToString(source_type).c_str());
		if (source_type == PhysicalOperatorType::TABLE_SCAN) {
			// initialize pipeline
			Pipeline duckdb_pipeline(*executor);
			ThreadContext thread_context(context);
			ExecutionContext exec_context(context, thread_context, &duckdb_pipeline);
			auto &table_scan = pipeline->source->Cast<GPUPhysicalTableScan>();
			table_scan.GetDataDuckDB(exec_context);
		}
		pipeline->source->GetData(*source_relation);
		// SIRIUS_LOG_DEBUG("source relation size {}", source_relation->columns.size());
		// for (auto col : source_relation->columns) {
		// 	SIRIUS_LOG_DEBUG("source relation column size {} column name {}", col->column_length, col->name);
		// }
		// EndOperator(*pipeline.source, &result);

		//call source
		// SIRIUS_LOG_DEBUG("{}", pipeline->source.get()->GetName());
		for (int current_idx = 1; current_idx <= pipeline->operators.size(); current_idx++) {
			auto op = pipeline->operators[current_idx-1];
			auto op_type = op.get().type;
			SIRIUS_LOG_DEBUG("pipeline operator type {}", PhysicalOperatorToString(op_type));
			printf("pipeline operator type %s\n", PhysicalOperatorToString(op_type).c_str());
			// SIRIUS_LOG_DEBUG("{}", op.get().GetName());
			//call operator

			auto current_intermediate = current_idx;
			auto &current_relation =
				current_intermediate >= intermediate_relations.size() ? final_relation : intermediate_relations[current_intermediate];
			// current_chunk.Reset();

			auto &prev_relation =
			    current_intermediate == initial_idx + 1 ? source_relation : intermediate_relations[current_intermediate - 1];
			auto operator_idx = current_idx - 1;
			auto &current_operator = pipeline->operators[operator_idx];

			// auto op_state = current_operator.GetOperatorState(context);
			// intermediate_states.push_back(std::move(op_state));
			
			// StartOperator(current_operator);
			// auto result = current_operator.get().Execute(exec_context, prev_relation, current_relation, *current_operator.op_state,
			//                                        *intermediate_states[current_intermediate - 1]);

			auto result = current_operator.get().Execute(*prev_relation, *current_relation);
			// EndOperator(current_operator, &current_chunk);
		}
		if (pipeline->sink) {
			auto sink_type = pipeline->sink.get()->type;
			SIRIUS_LOG_DEBUG("pipeline sink type {}", PhysicalOperatorToString(sink_type));
			printf("pipeline sink type %s\n", PhysicalOperatorToString(sink_type).c_str());
			// SIRIUS_LOG_DEBUG("{}", pipeline->sink.get()->GetName());
			//call sink
			auto &sink_relation = final_relation;
			// SIRIUS_LOG_DEBUG("sink relation size {}", final_relation->columns.size());
			// int i = 0;
			// for (auto col : final_relation->columns) {
			// 	if (col == nullptr) SIRIUS_LOG_DEBUG("{}", i);
			// 	i++;
			// 	// SIRIUS_LOG_DEBUG("sink relation column size {}", col->column_length);
			// }
			// auto interrupt_state = InterruptState();
			// auto local_sink_state = pipeline->sink->GetLocalSinkState(exec_context);
			// OperatorSinkInput sink_input {*pipeline->sink->sink_state, *local_sink_state, interrupt_state};
			// pipeline->sink->Sink(exec_context, *sink_relation, sink_input);
			pipeline->sink->Sink(*sink_relation);
		}
	}
}

void GPUExecutor::InitializeInternal(GPUPhysicalOperator &plan) {

	// auto &scheduler = TaskScheduler::GetScheduler(context);
	{
		// lock_guard<mutex> elock(executor_lock);
		gpu_physical_plan = &plan;

		// this->profiler = ClientData::Get(context).profiler;
		// profiler->Initialize(plan);
		// this->producer = scheduler.CreateProducer();

		// build and ready the pipelines
		GPUPipelineBuildState state;
		auto root_pipeline = make_shared_ptr<GPUMetaPipeline>(*this, state, nullptr);
		root_pipeline->Build(*gpu_physical_plan);
		root_pipeline->Ready();

		// ready recursive cte pipelines too
		// TODO: SUPPORT RECURSIVE CTE FOR GPU
		// for (auto &rec_cte_ref : recursive_ctes) {
		// 	auto &rec_cte = rec_cte_ref.get().Cast<PhysicalRecursiveCTE>();
		// 	// rec_cte.recursive_meta_pipeline->Ready();
		// }

		// set root pipelines, i.e., all pipelines that end in the final sink
		root_pipeline->GetPipelines(root_pipelines, false);		
		root_pipeline_idx = 0;

		// collect all meta-pipelines from the root pipeline
		vector<shared_ptr<GPUMetaPipeline>> to_schedule;
		root_pipeline->GetMetaPipelines(to_schedule, true, true);

		// number of 'PipelineCompleteEvent's is equal to the number of meta pipelines, so we have to set it here
		total_pipelines = to_schedule.size();
		
		SIRIUS_LOG_DEBUG("Total meta pipelines {}", to_schedule.size());
		int schedule_count = 0;
		int meta = 0;
		while (schedule_count < to_schedule.size()) {
			vector<shared_ptr<GPUMetaPipeline>> children;
			to_schedule[to_schedule.size() - 1 - meta]->GetMetaPipelines(children, false, true);
			auto base_pipeline = to_schedule[to_schedule.size() - 1 - meta]->GetBasePipeline();
			bool should_schedule = true;

			//already scheduled
			if (find(scheduled.begin(), scheduled.end(), base_pipeline) != scheduled.end()) {
				should_schedule = false;
			} else {
				//check if all children are scheduled
				for (auto &child : children) {
					if (find(scheduled.begin(), scheduled.end(), child->GetBasePipeline()) == scheduled.end()) {
						should_schedule = false;
						break;
					}
				}
				//check if all dependencies are scheduled
				for (int dep = 0; dep < base_pipeline->dependencies.size(); dep++) {
					if (find(scheduled.begin(), scheduled.end(), base_pipeline->dependencies[dep]) == scheduled.end()) {
						should_schedule = false;
						break;
					}
				}
			}
			if (should_schedule) {
				vector<shared_ptr<GPUPipeline>> pipeline_inside;
				to_schedule[to_schedule.size() - 1 - meta]->GetPipelines(pipeline_inside, false);
				for (int pipeline_idx = 0; pipeline_idx < pipeline_inside.size(); pipeline_idx++) {
					auto &pipeline = pipeline_inside[pipeline_idx];
					
					// Check if HASH_JOIN is the sink - if so, don't split
					bool hash_join_is_sink = pipeline->sink && pipeline->sink->type == PhysicalOperatorType::HASH_JOIN;
					bool hash_join_is_source = pipeline->source && pipeline->source->type == PhysicalOperatorType::HASH_JOIN;

					if (hash_join_is_source) continue;
					
					// Find all HASH_JOIN operators in the pipeline (but not if it's the sink)
					vector<idx_t> hash_join_indices;
					if (!hash_join_is_sink) {
						for (idx_t i = 0; i < pipeline->operators.size(); i++) {
							if (pipeline->operators[i].get().type == PhysicalOperatorType::HASH_JOIN) {
								hash_join_indices.push_back(i);
							}
						}
					}
					
					if (hash_join_indices.empty()) {
						// No HASH_JOIN in operators (or HASH_JOIN is sink), schedule as-is
						scheduled.push_back(pipeline);
					} else {
						// Split pipeline at each HASH_JOIN
						vector<shared_ptr<GPUPipeline>> split_pipelines;
						
						// Process each segment
						for (idx_t segment = 0; segment <= hash_join_indices.size(); segment++) {
							auto new_pipeline = make_shared_ptr<GPUPipeline>(*this);
							
							// Determine the range of operators for this segment
							idx_t start_idx = (segment == 0) ? 0 : hash_join_indices[segment - 1] + 1;
							idx_t end_idx = (segment == hash_join_indices.size()) ? pipeline->operators.size() : hash_join_indices[segment] + 1;
							
							// Set source for this segment
							if (segment == 0) {
								// First segment uses original source
								new_pipeline->source = pipeline->source;
							} else {
								// Other segments use DUMMY_SOURCE
								auto &prev_op = pipeline->operators[hash_join_indices[segment - 1]].get();
								auto dummy_source = make_uniq<GPUPhysicalDummySource>(prev_op.GetTypes(), prev_op.estimated_cardinality);
								new_pipeline->source = dummy_source.get();
								dummy_operators.push_back(std::move(dummy_source));
							}
							
							// Add operators for this segment
							for (idx_t i = start_idx; i < end_idx; i++) {
								new_pipeline->operators.push_back(pipeline->operators[i]);
							}
							
							// Set sink for this segment
							if (segment == hash_join_indices.size()) {
								// Last segment uses original sink
								new_pipeline->sink = pipeline->sink;
							} else {
								// Other segments use DUMMY_SINK
								auto &hash_join_op = pipeline->operators[hash_join_indices[segment]].get();
								auto dummy_sink = make_uniq<GPUPhysicalDummySink>(hash_join_op.GetTypes(), hash_join_op.estimated_cardinality);
								new_pipeline->sink = dummy_sink.get();
								dummy_operators.push_back(std::move(dummy_sink));
							}
							
							// Add dependency on previous segment
							if (segment > 0) {
								new_pipeline->AddDependency(split_pipelines[segment - 1]);
							}
							
							split_pipelines.push_back(new_pipeline);
						}
						
						// Schedule all split pipelines
						for (auto &split_pipeline : split_pipelines) {
							scheduled.push_back(split_pipeline);
						}
						
						SIRIUS_LOG_DEBUG("Split pipeline with {} HASH_JOIN operator(s)", hash_join_indices.size());
					}
				}
				// scheduled.push_back(base_pipeline);
				schedule_count++;
			}
			meta = (meta + 1) % to_schedule.size();
		}

		// collect all pipelines from the root pipelines (recursively) for the progress bar and verify them
		root_pipeline->GetPipelines(pipelines, true);
		SIRIUS_LOG_DEBUG("total_pipelines = {}", pipelines.size());

		// finally, verify and schedule
		// VerifyPipelines();
		// ScheduleEvents(to_schedule);
	}
}

void 
GPUExecutor::CancelTasks() {
		pipelines.clear();
		root_pipelines.clear();
}

shared_ptr<GPUPipeline> 
GPUExecutor::CreateChildPipeline(GPUPipeline &current, GPUPhysicalOperator &op) {
    D_ASSERT(!current.operators.empty());
    D_ASSERT(op.IsSource());
    // found another operator that is a source, schedule a child pipeline
    // 'op' is the source, and the sink is the same
    auto child_pipeline = make_shared_ptr<GPUPipeline>(*this);
    child_pipeline->sink = current.sink;
    child_pipeline->source = &op;

    // the child pipeline has the same operators up until 'op'
    for (auto current_op : current.operators) {
        if (&current_op.get() == &op) {
            break;
        }
        child_pipeline->operators.push_back(current_op);
    }

    return child_pipeline;
}

shared_ptr<GPUPipeline> GPUExecutor::SplitPipeline(GPUPipeline &current) {
  D_ASSERT(current.operators.size() >= 2);
  D_ASSERT(current.source);
  D_ASSERT(current.sink);
    
  idx_t split_point = current.operators.size() / 2;
  
  // TODO find a way to make these persist? right now this code definitely does not work
  auto dummy_sink = make_uniq<GPUPhysicalDummySink>();
  auto dummy_source = make_uniq<GPUPhysicalDummySource>();
  auto original_sink = current.sink;
  auto original_source = current.source;

  GPUPhysicalDummySink *dummy_sink_ptr = dummy_sink.get();
  GPUPhysicalDummySource *dummy_source_ptr = dummy_source.get();
  dummy_source_ptr->SetPairedSink(dummy_sink_ptr);
    
  vector<reference<GPUPhysicalOperator>> first_half_ops;
  for (idx_t i = 0; i < split_point; i++) {
    first_half_ops.push_back(current.operators[i]);
  }

  vector<reference<GPUPhysicalOperator>> second_half_ops;
  for (idx_t i = split_point; i < current.operators.size(); i++) {
    second_half_ops.push_back(current.operators[i]);
  }

  pipeline.operators = std::move(first_half_ops);
  pipeline.sink = dummy_sink_ptr;
  pipeline.source = original_source;


  auto second_pipeline = make_shared_ptr<GPUPipeline>();
  second_pipeline.operators = std::move(second_half_ops);
  second_pipeline.sink = original_sink;
  second_pipeline.source = dummy_source_ptr;
  
  second_pipeline->AddDependency(pipeline.shared_from_this());
    
  return second_pipeline;
}

bool 
GPUExecutor::HasResultCollector() {
	return gpu_physical_plan->type == PhysicalOperatorType::RESULT_COLLECTOR;
}

unique_ptr<QueryResult> 
GPUExecutor::GetResult() {
	D_ASSERT(HasResultCollector());
	if (!gpu_physical_plan) throw InvalidInputException("gpu_physical_plan is NULL");
	if (gpu_physical_plan.get() == NULL) throw InvalidInputException("gpu_physical_plan is NULL");
	auto &result_collector = gpu_physical_plan.get()->Cast<GPUPhysicalMaterializedCollector>();
	D_ASSERT(result_collector.sink_state);
	result_collector.sink_state = result_collector.GetGlobalSinkState(context);
	unique_ptr<QueryResult> res = result_collector.GetResult(*(result_collector.sink_state));
	return res;
}

}; // namespace duckdb