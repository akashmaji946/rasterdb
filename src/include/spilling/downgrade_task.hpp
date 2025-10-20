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
#include "parallel/task_executor.hpp"
#include "task_completion.hpp"
#include "helper/helper.hpp"
#include "data/data_repository.hpp"

namespace sirius {
namespace parallel {

/**
 * @brief Global state shared across all downgrade tasks in an operation.
 * 
 * This class holds references to shared resources that all downgrade tasks within
 * an operation need access to, including the data repository for storing results
 * and the message queue for task completion notifications.
 */
class DowngradeTaskGlobalState : public ITaskGlobalState {
public:
    /**
     * @brief Construct a new DowngradeTaskGlobalState object
     * 
     * @param data_repository Reference to the data repository for storing task outputs
     * @param message_queue Reference to the message queue for task completion notifications
     */
    explicit DowngradeTaskGlobalState(DataRepository& data_repository, TaskCompletionMessageQueue& message_queue) : 
        data_repository_(data_repository), message_queue_(message_queue) {}
    
    DataRepository& data_repository_;           ///< Repository for storing and retrieving data batches
    TaskCompletionMessageQueue& message_queue_; ///< Message queue to notify TaskCreator about task completion
};

/**
 * @brief A task representing a unit of work in a memory downgrade operation.
 * 
 * This class encapsulates the necessary information to execute a task within a memory
 * downgrade operation. Memory downgrading involves moving data from higher-tier (faster)
 * memory to lower-tier (slower) memory to free up space.
 */
class DowngradeTask : public ITask {
public:
    /**
     * @brief Construct a new DowngradeTask object
     * 
     * @param task_id A unique identifier for the task
     * @param pipeline_id The id of the pipeline associated with this task
     * @param data_batch The data batch to be processed by this task
     * @param local_state The local state specific to this task
     * @param global_state The global state shared across multiple tasks
     */
    DowngradeTask(uint64_t task_id, 
                    uint64_t pipeline_id,
                    sirius::unique_ptr<DataBatch> data_batch,
                    sirius::unique_ptr<ITaskLocalState> local_state,
                    sirius::shared_ptr<ITaskGlobalState> global_state)
        : ITask(std::move(local_state), std::move(global_state)),
          task_id_(task_id),
          data_batch_(std::move(data_batch)) {}

    /**
     * @brief Executes the memory downgrade operation for this task
     * 
     * This method performs the actual downgrading of data from a higher memory tier
     * to a lower memory tier.
     */
    void Execute() override;

    /**
     * @brief Get the unique identifier for this task
     * 
     * @return uint64_t The task ID
     */
    uint64_t GetTaskId() const { return task_id_; }

    /**
     * @brief Marks this task as completed and notifies dependent tasks
     * 
     * This method informs the TaskCreator that the task has been completed, allowing
     * it to schedule any tasks that were dependent on this task's completion. This
     * method should be called after successfully pushing the task's output to the
     * Data Repository.
     */
    void MarkTaskCompletion();

    /**
     * @brief Stores the output of this task in the Data Repository
     * 
     * @param data_batch The processed data batch to store in the repository
     * @param pipeline_id The ID of the pipeline that should receive this data batch
     */
    void PushToDataRepository(sirius::unique_ptr<sirius::DataBatch> data_batch, size_t pipeline_id);

    /**
     * @brief Get the data batch associated with this task
     * 
     * @return const DataBatch* Pointer to the data batch
     */
    const DataBatch* GetDataBatch() const { return data_batch_.get(); }

private:
    uint64_t task_id_;                                ///< Unique identifier for this task
    uint64_t pipeline_id_;                            ///< ID of the pipeline associated with this task
    sirius::unique_ptr<DataBatch> data_batch_;        ///< Data batch to be processed by this task
};

/**
 * @brief A thread-safe task queue specifically for managing DowngradeTask instances.
 * 
 * This class provides a thread-safe queue implementation for scheduling and retrieving
 * downgrade tasks. It uses mutexes and semaphores to ensure safe concurrent access.
 * Currently, it uses a simple FIFO queue (sirius::queue), but future versions may
 * implement more sophisticated scheduling algorithms such as priority scheduling,
 * task stealing, or work balancing.
 */
class DowngradeTaskQueue : public ITaskQueue {
public:
    /**
     * @brief Constructs a new DowngradeTaskQueue object
     * 
     * Initializes an empty queue in closed state. Call Open() to begin accepting tasks.
     */
    DowngradeTaskQueue() = default;

    /**
     * @brief Opens the task queue to start accepting and returning tasks
     * 
     * After calling this method, the queue will accept new tasks via Push() and
     * return tasks via Pull(). This method is thread-safe.
     */
    void Open() override {
        sirius::lock_guard<sirius::mutex> lock(mutex_);
        is_open_ = true;
    }

    /**
     * @brief Closes the task queue, stopping task acceptance and retrieval
     * 
     * After calling this method, the queue will no longer accept new tasks or
     * return tasks. Any threads waiting on Pull() will be signaled to wake up.
     * This method is thread-safe.
     */
    void Close() override {
        sem_.release(); // signal that one item is available
        sirius::lock_guard<sirius::mutex> lock(mutex_);
        is_open_ = false;
    }

    /**
     * @brief Pushes a new task to be scheduled (base interface implementation)
     * 
     * @param task The task to be scheduled (must be a DowngradeTask)
     * @throws std::runtime_error If the queue is not currently open for accepting tasks
     */
    void Push(sirius::unique_ptr<ITask> task) override {
        // Convert ITask to DowngradeTask - since we know it's a DowngradeTask
        auto downgrade_task = sirius::unique_ptr<DowngradeTask>(static_cast<DowngradeTask*>(task.release()));
        Push(std::move(downgrade_task));
    }

    /**
     * @brief Downgrade-specific push overload for type safety and convenience
     * 
     * @param downgrade_task The downgrade task to be scheduled
     * @throws std::runtime_error If the queue is not currently open for accepting tasks
     */
    void Push(sirius::unique_ptr<DowngradeTask> downgrade_task) {
        EnqueueTask(std::move(downgrade_task));
    }
    
    /**
     * @brief Pulls a task to execute (base interface implementation)
     * 
     * This is a blocking call that will wait until a task becomes available or the
     * queue is closed. Returns nullptr if the queue is closed and no tasks are available.
     * 
     * @return A unique pointer to the task to execute, or nullptr if none available
     * @throws std::runtime_error If the queue is closed and not returning tasks
     */
    sirius::unique_ptr<ITask> Pull() override {
        // Delegate to downgrade-specific version and return as base type
        auto downgrade_task = PullDowngradeTask();
        return std::move(downgrade_task);
    }

    /**
     * @brief Downgrade-specific pull method for type safety and convenience  
     * 
     * This is a blocking call that will wait until a task becomes available or the
     * queue is closed. Returns nullptr if the queue is closed and no tasks are available.
     * 
     * @return A unique pointer to the downgrade task to execute, or nullptr if none available
     * @throws std::runtime_error If the queue is closed and not returning tasks
     */
    sirius::unique_ptr<DowngradeTask> PullDowngradeTask() {
        return DequeueTask();
    }

    /**
     * @brief Enqueues a downgrade task into the queue
     * 
     * This is the internal implementation for adding tasks to the queue.
     * It's thread-safe and will only accept tasks if the queue is open.
     * 
     * @param downgrade_task The downgrade task to enqueue
     */
    void EnqueueTask(sirius::unique_ptr<DowngradeTask> downgrade_task) {
        sirius::lock_guard<sirius::mutex> lock(mutex_);
        if (downgrade_task && is_open_) {
            task_queue_.push(std::move(downgrade_task));
        }
        sem_.release(); // signal that one item is available
    }

    /**
     * @brief Dequeues a downgrade task from the queue
     * 
     * This is the internal implementation for retrieving tasks from the queue.
     * It's thread-safe and will block until a task is available or the queue is closed.
     * 
     * @return A unique pointer to the dequeued downgrade task, or nullptr if queue is empty and closed
     */
    sirius::unique_ptr<DowngradeTask> DequeueTask() {
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
     * @brief Checks if the task queue is empty
     * 
     * @return true if the queue contains no tasks, false otherwise
     */
    bool IsEmpty() const {
        sirius::lock_guard<sirius::mutex> lock(mutex_);
        return task_queue_.empty();
    }
    
private:
    sirius::queue<sirius::unique_ptr<DowngradeTask>> task_queue_;  ///< FIFO queue storing the downgrade tasks
    bool is_open_ = false;                                         ///< Whether the queue is open for operations
    mutable sirius::mutex mutex_;                                  ///< Mutex for thread-safe access (mutable for const methods)
    std::counting_semaphore<> sem_{0};                             ///< Semaphore for blocking/signaling (starts with 0 permits)
};

} // namespace parallel
} // namespace sirius