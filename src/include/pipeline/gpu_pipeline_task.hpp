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
#include <queue>
#include <mutex>
#include "gpu_pipeline.hpp"
#include "data/data_batch.hpp"
#include "parallel/task.hpp"
#include "parallel/task_scheduler.hpp"
#include "helper/helper.hpp"

namespace sirius {
namespace parallel {

class GPUPipelineTask : public ITask {
public:
    GPUPipelineTask(sirius::shared_ptr<duckdb::GPUPipeline> pipeline, 
                    uint64_t task_id, 
                    sirius::unique_ptr<DataBatch> data_batch,
                    sirius::unique_ptr<ITaskLocalState> local_state,
                    sirius::shared_ptr<ITaskGlobalState> global_state)
        : ITask(std::move(local_state), std::move(global_state)),
          pipeline_(std::move(pipeline)),
          task_id_(task_id),
          data_batch_(std::move(data_batch)) {}

    // Implement the pure virtual Execute method from ITask
    void Execute() override {
        // TODO: Implement the actual GPU pipeline task execution logic
        // This is where you would process the data_batch_ using the pipeline_
        if (pipeline_ && data_batch_) {
        }
    }

    // Getter methods for accessing task properties
    uint64_t GetTaskId() const { return task_id_; }
    const DataBatch* GetDataBatch() const { return data_batch_.get(); }
    const duckdb::GPUPipeline* GetPipeline() const { return pipeline_.get(); }

private:
    sirius::shared_ptr<duckdb::GPUPipeline> pipeline_;
    uint64_t task_id_;
    sirius::unique_ptr<DataBatch> data_batch_;
};

class GPUPipelineTaskQueue : public ITaskQueue {
public:
    GPUPipelineTaskQueue() = default;

    // Implement ITaskQueue interface
    void Open() override {
        std::lock_guard<std::mutex> lock(mutex_);
        is_open_ = true;
    }

    void Close() override {
        std::lock_guard<std::mutex> lock(mutex_);
        is_open_ = false;
    }

    void Push(sirius::unique_ptr<ITask> task) override {
        // Convert ITask to GPUPipelineTask - since we know it's a GPUPipelineTask
        auto gpu_task = sirius::unique_ptr<GPUPipelineTask>(static_cast<GPUPipelineTask*>(task.release()));
        Push(std::move(gpu_task));
    }

    // GPU-specific overload for type safety and convenience
    void Push(sirius::unique_ptr<GPUPipelineTask> gpu_task) {
        EnqueueTask(std::move(gpu_task));
    }
    
    sirius::unique_ptr<ITask> Pull() override {
        // Delegate to GPU-specific version and return as base type
        auto gpu_task = PullGPUTask();
        return std::move(gpu_task);
    }

    // GPU-specific method for type safety and convenience  
    sirius::unique_ptr<GPUPipelineTask> PullGPUTask() {
        return DequeueTask();
    }

    // GPU-specific methods
    void EnqueueTask(sirius::unique_ptr<GPUPipelineTask> gpu_pipeline_task) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (gpu_pipeline_task && is_open_) {
            task_queue_.push(std::move(gpu_pipeline_task));
        }
    }

    sirius::unique_ptr<GPUPipelineTask> DequeueTask() {
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
    sirius::queue<sirius::unique_ptr<GPUPipelineTask>> task_queue_;
    bool is_open_ = false;
    mutable std::mutex mutex_;  // mutable to allow locking in const methods
};

} // namespace parallel
} // namespace sirius
