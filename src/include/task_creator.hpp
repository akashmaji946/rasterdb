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
#include "data/data_repository.hpp"
#include "pipeline/gpu_pipeline_executor.hpp"
#include "scan/duckdb_scan_executor.hpp"
#include "parallel/task_executor.hpp"
#include "parallel/task.hpp"
#include "parallel/task_creator.hpp"

namespace sirius {

class duckdb::GPUExecutor;

enum class Source {
    SCAN,
    PIPELINE
};

// Message indicating the completion of a task, notifying task creator that it should check if new tasks can be created
struct TaskCompletionMessage {
    uint64_t task_id;
    uint64_t pipeline_id;
    Source source;
};

class TaskCompletionMessageQueue {
public:
    TaskCompletionMessageQueue() = default;
    ~TaskCompletionMessageQueue() = default;
    
    void EnqueueMessage(const TaskCompletionMessage &message) {
        lock_guard<mutex> lock(mutex_);
        message_queue_.push(message);
    }
    
    bool DequeueMessage(TaskCompletionMessage &message) {
        lock_guard<mutex> lock(mutex_);
        if (message_queue_.empty()) {
            return false;
        }
        message = message_queue_.front();
        message_queue_.pop();
        return true;
    }
    
private:
    mutex mutex_;
    sirius::queue<TaskCompletionMessage> message_queue_;
};

class TaskCreator : public ITaskCreator {
public:
    TaskCreator(DataRepository& data_repository,
        parallel::GPUPipelineExecutor &gpu_pipeline_executor,
        parallel::DuckDBScanExecutor& duckdb_scan_executor);
          
    // Destructor
    ~TaskCreator() = default;

    // Non-copyable but movable
    TaskCreator(const TaskCreator&) = delete;
    TaskCreator& operator=(const TaskCreator&) = delete;
    TaskCreator(TaskCreator&&) = default;
    TaskCreator& operator=(TaskCreator&&) = default;

    // pull messages from the message queue and create tasks accordingly
    void PullMessage();

    // scan the data repository for new data batches and submit scan/pipeline tasks
    void ScanRepository();

    void WorkerLoop() override;

    void SetCoordinator(duckdb::GPUExecutor* coordinator) {
        coordinator_ = coordinator;
    }

    // submit scan task to scan executor
    void ScheduleDuckDBScan (sirius::unique_ptr<parallel::DuckDBScanTask> scan_getsize_task);

    // submit pipeline task to pipeline executor
    void SchedulePipelineTask(sirius::unique_ptr<parallel::GPUPipelineTask> gpu_pipeline_task);

    void Run();
    void Signal();
    void Wait();

private:
    TaskCompletionMessageQueue task_completion_message_queue_;
    DataRepository& data_repository_;
    parallel::GPUPipelineExecutor& gpu_pipeline_executor_;
    parallel::DuckDBScanExecutor& duckdb_scan_executor_;
    duckdb::GPUExecutor* coordinator_; // reference to the coordinator (GPUExecutor)

	sirius::mutex mtx;
	std::condition_variable cv;
	bool ready = false;
};

} // namespace sirius