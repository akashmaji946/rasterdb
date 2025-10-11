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

#include "task_scheduler.hpp"
#include "helper/helper.hpp"

namespace sirius {
namespace parallel {

/**
 * @brief Struct representing a worker thread in the TaskExecutor
 * 
 * The primary purpose of this struct is to store the state needed by each worker that is not present
 * in the tasks that it executes. 
 */  
struct TaskExecutorThread {
  /**
   * @brief Constructor
   * 
   * @param thread The actual thread object
   */
  explicit TaskExecutorThread(sirius::unique_ptr<std::thread> thread)
    : internal_thread_(std::move(thread)) {}

  sirius::unique_ptr<std::thread> internal_thread_; // The underlying thread
};

/**
 * @brief Struct storing the various configuration options for a TaskExecutor
 */
struct TaskExecutorConfig {
  int num_threads; // The number of worker threads in the executor
  bool retry_on_error; // Whether to retry a task if it fails
};

/**
 * @brief Interface for a thread pool of workers executing tasks.
 * 
 * Each worker in the thread pool should just repeatedly pull tasks from the scheduler and execute them.
 * All logic related to the scheduling of the tasks and thier priority should be handled by the scheduler.
 * Each worker is also responsible for handling any task failures/error based on the configuration provided.
 */
class ITaskExecutor {
public:
  /**
   * @brief Construct a new ITaskExecutor object
   * 
   * @param scheduler The task scheduler to use for scheduling tasks
   * @param config Configuration options for the task executor
   */
  ITaskExecutor(sirius::shared_ptr<ITaskQueue> scheduler, TaskExecutorConfig config)
    : scheduler_(std::move(scheduler)), config_(config), running_(false) {}
  
  /**
   * @brief The desctrutor for the ITaskExecutor.
   */
  virtual ~ITaskExecutor() {
    Stop();
  }

  // Non-copyable and movable
  ITaskExecutor(const ITaskExecutor&) = delete;
  ITaskExecutor& operator=(const ITaskExecutor&) = delete;
  ITaskExecutor(ITaskExecutor&&) = default;
  ITaskExecutor& operator=(ITaskExecutor&&) = default;

  /**
   * @brief Start the task executor and its worker threads.
   */
  virtual void Start();

  /**
   * @brief Stop the task executor and its worker threads.
   * 
   * This method will block the executor from accepting any new requests but will wait for all currently executing tasks to finish.
   * Any task that is scheduled but not yet executing will not be executed.
   */
  virtual void Stop();

  /**
   * @brief Schedule a new task for execution
   * 
   * @param task The task to schedule for execution
   * @throws std::runtime_error If the executor is not currently accepting requests (has not been started or has been stopped) 
   */
  virtual void Schedule(sirius::unique_ptr<ITask> task);

private:
  /**
   * @brief Helper method to initialize any executor related state
   */
  virtual void OnStart();

  /**
   * @brief Helper method to cleanup any executor related state
   */
  virtual void OnStop();

  /**
   * @brief Helper method to handle any task errors
   * 
   * @param worker_id The id of the worker that encountered the error
   * @param task The task that encountered the error
   * @param e The exception that was thrown
   */
  virtual void OnTaskError(int worker_id, sirius::unique_ptr<ITask> task, const std::exception& e);

  /**
   * @brief The main loop executed by each worker thread.
   * 
   * This method repeatedly pulls tasks from the scheduler and executes them until the executor is stopped
   * 
   * @param worker_id The id of the worker thread
   */
  virtual void WorkerLoop(int worker_id);

private:
  sirius::shared_ptr<ITaskQueue> scheduler_; // The task scheduler used by the executor
  TaskExecutorConfig config_; // The configuration options for the executor
  std::atomic<bool> running_; // Whether the executor is currently running
  sirius::vector<sirius::unique_ptr<TaskExecutorThread>> threads_; // The worker threads in the executor
};

} // namespace parallel
} // namespace sirius