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
#include "parallel/task_executor.hpp"
#include "helper/helper.hpp"

namespace sirius {
namespace parallel {

/**
 * @brief A task representing a unit of work in a GPU pipeline.
 * 
 * This class encaplulates the necessary information to execute a task within a pipeline on the GPU. 
 * These task will be created by the TaskCreator and be scheduled for execution on the GPuPipelineExecutor.
 * 
 * Note that this class will be further derived to represent specific types of tasks such as build, aggregation, etc..
 */
class GPUPipelineTask : public ITask {
public:
    /**
     * @brief Construct a new GPUPipelineTask object
     * 
     * @param pipeline The GPU pipeline associated with this task
     * @param task_id A unique identifier for the task
     * @param data_batch The data batch to be processed by this task
     * @param local_state The local state specific to this task
     * @param global_state The global state shared across multiple tasks
     */
    GPUPipelineTask(sirius::shared_ptr<duckdb::GPUPipeline> pipeline, 
                    uint64_t task_id, 
                    sirius::unique_ptr<DataBatch> data_batch,
                    sirius::unique_ptr<ITaskLocalState> local_state,
                    sirius::shared_ptr<ITaskGlobalState> global_state)
        : ITask(std::move(local_state), std::move(global_state)),
          pipeline_(std::move(pipeline)),
          task_id_(task_id),
          data_batch_(std::move(data_batch)) {}

    /**
     * @brief Method to actually execute the task
     */
    void Execute() override {
        // TODO: Implement the actual GPU pipeline task execution logic
        // This is where you would process the data_batch_ using the pipeline_
        if (pipeline_ && data_batch_) {
        }
    }

    /**
     * @brief Get the unique identifier for this task
     * 
     * @return uint64_t The task ID
     */
    uint64_t GetTaskId() const { return task_id_; }

    /**
     * @brief Get the data batch associated with this task
     * 
     * @return const DataBatch* Pointer to the data batch
     */
    const DataBatch* GetDataBatch() const { return data_batch_.get(); }

    /**
     * @brief Get the GPU pipeline associated with this task
     * 
     * @return const duckdb::GPUPipeline* Pointer to the GPU pipeline
     */
    const duckdb::GPUPipeline* GetPipeline() const { return pipeline_.get(); }

    /**
     * @brief Method to mark that this task is completed
     * 
     * This method informs that TaskCreator that the task is completed so that it can start scheduling
     * tasks that were dependent on this task. This method should be called after pushing the output
     * of this task to the Data Repository.
     */
    void MarkTaskCompletion();

    /**
     * @brief Method to push the output of this task to the Data Repository
     * 
     * @param data_batch The data batch to push
     * @param pipeline_id The id of the pipeline that produced this data batch
     */
    void PushToDataRepository(sirius::unique_ptr<::sirius::DataBatch> data_batch, size_t pipeline_id);

private:
    sirius::shared_ptr<duckdb::GPUPipeline> pipeline_;
    uint64_t task_id_;
    sirius::unique_ptr<DataBatch> data_batch_;
};

/**
 * @brief A task queue specifically for managing GPUPipelineTask instances.
 * 
 * This class provides a thread-safe queue implementation for scheduling and retrieving GPU pipeline tasks.
 * Currently it just uses the sirius::queue (which is just the sirius::queue) but in the future we might want to
 * implement a more sophisticated queue that supports priority scheduling, task stealing, etc..
 */
class GPUPipelineTaskQueue : public ITaskQueue {
public:
    /**
     * @brief Construct a new GPUPipelineTaskQueue object
     */
    GPUPipelineTaskQueue() = default;

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
     * @throws sirius::runtime_error If the scheduler is not currently accepting requests
     */
    void Push(sirius::unique_ptr<ITask> task) override {
        // Convert ITask to GPUPipelineTask - since we know it's a GPUPipelineTask
        auto gpu_task = sirius::unique_ptr<GPUPipelineTask>(static_cast<GPUPipelineTask*>(task.release()));
        Push(std::move(gpu_task));
    }

    /**
     * @brief GPU-specific push overload for type safety and convenience
     * 
     * @param gpu_task The GPU pipeline task to be scheduled
     * @throws sirius::runtime_error If the scheduler is not currently accepting requests
     */
    void Push(sirius::unique_ptr<GPUPipelineTask> gpu_task) {
        EnqueueTask(std::move(gpu_task));
    }
    
    /**
     * @brief Pull a task to execute.
     * 
     * Note that this is a non blocking call and will return nullptr if no task is available. In the future we should
     * consider this call blocking. 
     * 
     * @return A unique pointer to the task to execute if there is one, nullptr otherwise
     * @throws sirius::runtime_error If the scheduler is not currently stopped and thus not returning tasks
     */
    sirius::unique_ptr<ITask> Pull() override {
        // Delegate to GPU-specific version and return as base type
        auto gpu_task = PullGPUTask();
        return std::move(gpu_task);
    }

    /**
     * @brief GPU-specific pull method for type safety and convenience  
     * 
     * @return A unique pointer to the GPU pipeline task to execute, nullptr otherwise
     * @throws sirius::runtime_error If the scheduler is not currently stopped and thus not returning tasks
     */
    sirius::unique_ptr<GPUPipelineTask> PullGPUTask() {
        return DequeueTask();
    }

    /**
     * @brief Enqueue a GPU pipeline task into the queue
     * 
     * @param gpu_pipeline_task The GPU pipeline task to enqueue
     */
    void EnqueueTask(sirius::unique_ptr<GPUPipelineTask> gpu_pipeline_task) {
        sirius::lock_guard<sirius::mutex> lock(mutex_);
        if (gpu_pipeline_task && is_open_) {
            task_queue_.push(std::move(gpu_pipeline_task));
        }
    }

    /**
     * @brief Dequeue a GPU pipeline task from the queue
     * 
     * @return A unique pointer to the dequeued GPU pipeline task if there is a task, nullptr otherwise
     */
    sirius::unique_ptr<GPUPipelineTask> DequeueTask() {
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
    sirius::queue<sirius::unique_ptr<GPUPipelineTask>> task_queue_; // The underlying queue storing the tasks
    bool is_open_ = false; // Whether the queue is open for accepting and returning tasks
    mutable sirius::mutex mutex_;  // mutable to allow locking in const methods
};

} // namespace parallel
} // namespace sirius
