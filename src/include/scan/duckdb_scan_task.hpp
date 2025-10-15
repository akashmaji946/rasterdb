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

#pragma once
#include "config.hpp"
#include "operator/gpu_physical_table_scan.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/execution/execution_context.hpp"
#include "parallel/task_executor.hpp"
#include "data/data_batch.hpp"
#include "helper/helper.hpp"

// The implementations have been based on the initial implementation in gpu_phyisical_table_scan.cpp by Yifei
namespace sirius {
namespace parallel {

class DuckDBScanGlobalSourceState : public ITaskGlobalState {
public:

	idx_t max_threads = 0;
	sirius::unique_ptr<duckdb::GlobalTableFunctionState> global_state;
	bool in_out_final = false;
	sirius::unique_ptr<duckdb::TableFilterSet> table_filters;

	duckdb::optional_ptr<duckdb::TableFilterSet> GetTableFilters(const duckdb::GPUPhysicalTableScan &op) const {
		return table_filters ? table_filters.get() : op.fake_table_filters.get();
	}

    // The followings are used in `TableScanCoalesceTask`
    void InitForTableScanCoalesceTask(const duckdb::GPUPhysicalTableScan& op, uint8_t** mask_ptr_p) {
    }

    void NextChunkOffsetsAligned(uint64_t chunk_rows, const sirius::vector<uint64_t>& chunk_column_sizes,
                                uint64_t* out_row_offset, sirius::vector<uint64_t>& out_column_data_offsets) {
    }

    inline void AssignBits(uint8_t from, int from_pos, uint8_t* to, int to_pos, int n) {
    }

    void NextChunkOffsetsUnaligned(uint64_t chunk_rows, const sirius::vector<uint64_t>& chunk_column_sizes,
                                    uint64_t* out_row_offset, sirius::vector<uint64_t>& out_column_data_offsets,
                                    const sirius::vector<uint8_t>& chunk_unaligned_mask_bytes) {
    }

    // For both rows which are null mask aligned and unaligned
    struct {
        sirius::mutex mutex;
        uint64_t row_offset;
        sirius::vector<uint64_t> column_data_offsets;
    } offset_info_aligned, offset_info_unaligned;

    // For compacting null mask bytes of unaligned portion per column. We write starting from last bit
    // since the unaligned portion is written from the end.
    uint8_t** mask_ptr;
    uint64_t unaligned_mask_byte_pos;
    int unaligned_mask_in_byte_pos;
};

class DuckDBScanLocalSourceState : public ITaskLocalState {
public:
	DuckDBScanLocalSourceState(duckdb::ExecutionContext &context, DuckDBScanGlobalSourceState &gstate,
	                     const duckdb::GPUPhysicalTableScan &op) {
		if (op.function.init_local) {
			duckdb::TableFunctionInitInput input(op.bind_data.get(), op.column_ids, op.scanned_ids,
			                             gstate.GetTableFilters(op), op.extra_info.sample_options);
			local_state = op.function.init_local(context, input, gstate.global_state.get());
		}
        num_rows = 0;
        column_size.resize(op.column_ids.size(), 0);
	}

	sirius::unique_ptr<duckdb::LocalTableFunctionState> local_state;

    // Used in `TableScanGetSizeTask`
    uint64_t num_rows;
    sirius::vector<uint64_t> column_size;
};

class DuckDBScanTask : public ITask {
public:
    /**
     * @brief Construct a new DuckDBScanTask object
     *
     * @param task_id Unique identifier for this scan task
     * @param function_p Reference to the DuckDB table function
     * @param context_p Execution context for the scan
     * @param op_p Physical table scan operator
     * @param local_state Local state specific to this task
     * @param global_state Global state shared across scan tasks
     */
    DuckDBScanTask(uint64_t task_id, 
                    duckdb::TableFunction& function_p, 
                    duckdb::ExecutionContext& context_p,
                    duckdb::GPUPhysicalTableScan& op_p,
                    sirius::unique_ptr<ITaskLocalState> local_state,
                    sirius::shared_ptr<ITaskGlobalState> global_state)
        : ITask(std::move(local_state), std::move(global_state)),
          task_id(task_id), function(function_p), context(context_p), op(op_p) {}

	void Execute() override;

    // Convert the output chunk from duckdb to a DataBatch
    void ConvertToDataBatch();

    // push Output DataBatch to Data Repository
    void PushToDataRepository(::sirius::unique_ptr<::sirius::DataBatch> data_batch, size_t pipeline_id, size_t idx);

private:
  int task_id;
  duckdb::TableFunction& function;
  duckdb::ExecutionContext& context;
  duckdb::GPUPhysicalTableScan& op;
  ITaskGlobalState* g_state;
  ITaskLocalState* l_state;
};

class DuckDBScanTaskQueue : public ITaskQueue {
public:
    /**
     * @brief Construct a new DuckDBScanTaskQueue object
     */
    DuckDBScanTaskQueue() = default;

    /**
     * @brief Setups the task queue to start accepting and returning tasks
     */
    void Open() override {
        sirius::lock_guard<sirius::mutex> lock(mutex_);
        is_open_ = true;
    }

    /**
     * @brief Closes the task queue from accepting new tasks or returning tasks
     */
    void Close() override {
        sirius::lock_guard<sirius::mutex> lock(mutex_);
        is_open_ = false;
    }

    /**
     * @brief Push a new task to be scheduled.
     * 
     * @param task The task to be scheduled
     * @throws std::runtime_error If the scheduler is not currently accepting requests
     */
    void Push(sirius::unique_ptr<ITask> task) override {
        // Convert ITask to DuckDBScanTask - since we know it's a DuckDBScanTask
        auto duckdb_scan_task = sirius::unique_ptr<DuckDBScanTask>(static_cast<DuckDBScanTask*>(task.release()));
        Push(std::move(duckdb_scan_task));
    }

    /**
     * @brief DuckDB scan specific push overload for type safety and convenience
     * 
     * @param duckdb_scan_task The DuckDB scan task to be scheduled
     * @throws std::runtime_error If the scheduler is not currently accepting requests
     */
    void Push(sirius::unique_ptr<DuckDBScanTask> duckdb_scan_task) {
        EnqueueTask(std::move(duckdb_scan_task));
    }
    
    /**
     * @brief Pull a task to execute.
     * 
     * Note that this is a non blocking call and will return nullptr if no task is available. In the future we should
     * consider this call blocking. 
     * 
     * @return A unique pointer to the task to execute if there is one, nullptr otherwise
     * @throws std::runtime_error If the scheduler is not currently stopped and thus not returning tasks
     */
    sirius::unique_ptr<ITask> Pull() override {
        // Delegate to DuckDB scan specific version and return as base type
        auto duckdb_scan_task = PullScanTask();
        return std::move(duckdb_scan_task);
    }

    /**
     * @brief DuckDB scan specific pull method for type safety and convenience  
     * 
     * @return A unique pointer to the DuckDB scan task to execute, nullptr otherwise
     * @throws std::runtime_error If the scheduler is not currently stopped and thus not returning tasks
     */
    sirius::unique_ptr<DuckDBScanTask> PullScanTask() {
        return DequeueTask();
    }

    /**
     * @brief Enqueue a DuckDB scan task into the queue
     * 
     * @param duckdb_scan_task The DuckDB scan task to enqueue
     */
    void EnqueueTask(sirius::unique_ptr<DuckDBScanTask> duckdb_scan_task) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (duckdb_scan_task && is_open_) {
            task_queue_.push(std::move(duckdb_scan_task));
        }
    }

    /**
     * @brief Dequeue a DuckDB scan task from the queue
     * 
     * @return A unique pointer to the dequeued DuckDB scan task if there is a task, nullptr otherwise
     */
    sirius::unique_ptr<DuckDBScanTask> DequeueTask() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (task_queue_.empty()) {
            return nullptr;
        }
        auto task = std::move(task_queue_.front());
        task_queue_.pop();
        return task;
    }

    /**
     * @brief Check if the task queue is empty
     * 
     * @return true if the queue is empty, false otherwise
     */
    bool IsEmpty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return task_queue_.empty();
    }

private:
    sirius::queue<sirius::unique_ptr<DuckDBScanTask>> task_queue_; // The underlying queue storing the tasks
    bool is_open_ = false; // Whether the queue is open for accepting and returning tasks
    mutable std::mutex mutex_;  // mutable to allow locking in const methods
};

} // namespace parallel
} // namespace sirius