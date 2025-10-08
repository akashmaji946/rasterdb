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
#include "data/data_batch.hpp"
#include "parallel/task_scheduler.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/common/shared_ptr.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/queue.hpp"
#include "duckdb/common/helper.hpp"

namespace duckdb {
namespace sirius {
namespace parallel {

class DowngradeTask : public ITask {
public:
    DowngradeTask(uint64_t task_id, 
                    duckdb::unique_ptr<DataBatch> data_batch,
                    duckdb::unique_ptr<ITaskLocalState> local_state,
                    duckdb::shared_ptr<ITaskGlobalState> global_state)
        : ITask(std::move(local_state), std::move(global_state)),
          task_id_(task_id),
          data_batch_(std::move(data_batch)) {}

    // Implement the pure virtual Execute method from ITask
    void Execute() override {
    }

    // Getter methods for accessing task properties
    uint64_t GetTaskId() const { return task_id_; }
    const DataBatch* GetDataBatch() const { return data_batch_.get(); }

private:
    uint64_t task_id_;
    duckdb::unique_ptr<DataBatch> data_batch_;
};

class DowngradeTaskQueue : public ITaskScheduler {
public:
    DowngradeTaskQueue() = default;

    // Implement ITaskScheduler interface
    void Open() override {
        std::lock_guard<std::mutex> lock(mutex_);
        is_open_ = true;
    }

    void Close() override {
        std::lock_guard<std::mutex> lock(mutex_);
        is_open_ = false;
    }

    void Push(duckdb::unique_ptr<ITask> task) override {
        // Convert ITask to DowngradeTask - since we know it's a DowngradeTask
        auto gpu_task = duckdb::unique_ptr<DowngradeTask>(static_cast<DowngradeTask*>(task.release()));
        Push(std::move(gpu_task));
    }

    // GPU-specific overload for type safety and convenience
    void Push(duckdb::unique_ptr<DowngradeTask> gpu_task) {
        EnqueueTask(std::move(gpu_task));
    }
    
    duckdb::unique_ptr<ITask> Pull() override {
        // Delegate to GPU-specific version and return as base type
        auto gpu_task = PullDowngradeTask();
        return std::move(gpu_task);
    }

    // GPU-specific method for type safety and convenience  
    duckdb::unique_ptr<DowngradeTask> PullDowngradeTask() {
        return DequeueTask();
    }

        // GPU-specific methods
    void EnqueueTask(duckdb::unique_ptr<DowngradeTask> downgrade_task) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (downgrade_task && is_open_) {
            task_queue_.push(std::move(downgrade_task));
        }
    }

    duckdb::unique_ptr<DowngradeTask> DequeueTask() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (task_queue_.empty()) {
            return nullptr;
        }
        auto task = std::move(task_queue_.front());
        task_queue_.pop();
        return task;
    }

    bool IsEmpty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return task_queue_.empty();
    }
    
private:
    duckdb::queue<duckdb::unique_ptr<DowngradeTask>> task_queue_;
    bool is_open_ = false;
    mutable std::mutex mutex_;  // mutable to allow locking in const methods

};

} // namespace parallel
} // namespace sirius
} // namespace duckdb