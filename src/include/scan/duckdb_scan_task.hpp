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
#include "gpu_pipeline_hashmap.hpp"
#include "task_completion.hpp"
#include "parallel/task_executor.hpp"
#include "data/data_repository.hpp"
#include "helper/helper.hpp"

// The implementations have been based on the initial implementation in gpu_phyisical_table_scan.cpp by Yifei
namespace sirius {
namespace parallel {

class DuckDBScanGlobalSourceState : public ITaskGlobalState {
public:
    explicit DuckDBScanGlobalSourceState(DataRepository& data_repository, TaskCompletionMessageQueue& message_queue) : 
        data_repository_(data_repository), message_queue_(message_queue) {}
    DataRepository& data_repository_;
    TaskCompletionMessageQueue& message_queue_; // Message queue to notify TaskCreator about completion of the
};

class DuckDBScanTask : public ITask {
public:
    /**
     * @brief Construct a new DuckDBScanTask object
     *
     * @param scan_metadata Metadata required for the scan operation
     * @param task_id A unique identifier for the task
     * @param pipeline_id The id of the pipeline associated with this task
     * @param local_state Local state specific to this task
     * @param global_state Global state shared across scan tasks
     */
    DuckDBScanTask(uint64_t task_id,
                    uint64_t pipeline_id,
                    sirius::shared_ptr<DuckDBScanMetadata> scan_metadata,
                    sirius::unique_ptr<ITaskLocalState> local_state,
                    sirius::shared_ptr<ITaskGlobalState> global_state)
        : ITask(std::move(local_state), std::move(global_state)),
          scan_metadata_(scan_metadata),
          task_id_(task_id),
          pipeline_id_(pipeline_id) {}

    /**
     * @brief Method to actually execute the downgrade task
     */
	void Execute() override;

    /**
     * @brief Method to convert the scanned data into a DataBatch
     * 
     * This method transforms the raw data obtained from the DuckDB scan into a DataBatch
     * that can be pushed to the Data Repository for further processing.
     */
    void ConvertToDataBatch();

    /**
     * @brief Method to mark that this task is completed
     * 
     * This method informs that TaskCreator that the task is completed so that it can start scheduling
     * tasks that were dependent on this task. This method should be called after pushing the output
     * of this task to the Data Repository.
     */
    void MarkTaskCompletion();

    /**
     * @brief Method to push the resulting DataBatch to the Data Repository
     * 
     * @param data_batch The data batch to push
     * @param pipeline_id The identifier of the pipeline that produced this data batch
     */
    void PushToDataRepository(sirius::unique_ptr<sirius::DataBatch> data_batch, size_t pipeline_id);

private:
  uint64_t task_id_;
  uint64_t pipeline_id_;
  sirius::shared_ptr<DuckDBScanMetadata> scan_metadata_;
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
        sem_.release(); // signal that one item is available
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
        sirius::lock_guard<sirius::mutex> lock(mutex_);
        if (duckdb_scan_task && is_open_) {
            task_queue_.push(std::move(duckdb_scan_task));
        }
        sem_.release(); // signal that one item is available
    }

    /**
     * @brief Dequeue a DuckDB scan task from the queue
     * 
     * @return A unique pointer to the dequeued DuckDB scan task if there is a task, nullptr otherwise
     */
    sirius::unique_ptr<DuckDBScanTask> DequeueTask() {
        sem_.acquire(); // wait until there's something
        sirius::lock_guard<sirius::mutex> lock(mutex_);
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
        sirius::lock_guard<sirius::mutex> lock(mutex_);
        return task_queue_.empty();
    }

private:
    sirius::queue<sirius::unique_ptr<DuckDBScanTask>> task_queue_; // The underlying queue storing the tasks
    bool is_open_ = false; // Whether the queue is open for accepting and returning tasks
    mutable sirius::mutex mutex_;  // mutable to allow locking in const methods
    std::counting_semaphore<> sem_{0}; // starts with 0 available permits
};

} // namespace parallel
} // namespace sirius