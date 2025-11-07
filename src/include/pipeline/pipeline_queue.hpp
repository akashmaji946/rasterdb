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
 #include "gpu_pipeline.hpp"
 #include "data/data_repository.hpp"
 #include "parallel/task_executor.hpp"
 #include "helper/helper.hpp"
 #include "pipeline/gpu_pipeline_task.hpp"
//  #include "task_completion.hpp"
 #include <semaphore>
 #include "config.hpp"
 
 namespace sirius {
 namespace parallel {
 
 /**
  * @brief A task queue specifically for managing gpu_pipeline_task instances.
  * 
  * This class provides a thread-safe queue implementation for scheduling and retrieving GPU pipeline tasks.
  * Currently it just uses the sirius::queue (which is just the sirius::queue) but in the future we might want to
  * implement a more sophisticated queue that supports priority scheduling, task stealing, etc..
  */
 class pipeline_queue : public itask_queue {
 public:
     /**
      * @brief Construct a new pipeline_queue object
     */
    pipeline_queue() = default;

    /**
     * @brief Setups the task queue to start accepting and returning tasks
     */
    void open() override;

    /**
     * @brief Closes the task queue from accepting new tasks or returning tasks
     */
    void close() override;

    /**
     * @brief Push a new task to be scheduled.
     * 
     * @param task The task to be scheduled
     * @throws sirius::runtime_error If the scheduler is not currently accepting requests
     */
    void push(sirius::unique_ptr<itask> task) override;
    
    /**
     * @brief Pull a task to execute.
     * 
     * Note that this is a non blocking call and will return nullptr if no task is available. In the future we should
     * consider this call blocking. 
     * 
     * @return A unique pointer to the task to execute if there is one, nullptr otherwise
     * @throws sirius::runtime_error If the scheduler is not currently stopped and thus not returning tasks
     */
    sirius::unique_ptr<itask> pull() override;

    /**
     * @brief Check if the task queue is empty
     * 
     * @return true if the queue is empty, false otherwise
     */
    bool is_empty() const;

 private:
     sirius::queue<sirius::unique_ptr<gpu_pipeline_task>> _task_queue; // The underlying queue storing the tasks
     bool _is_open = false; // Whether the queue is open for accepting and returning tasks
     mutable sirius::mutex _mutex;  // mutable to allow locking in const methods
     std::counting_semaphore<> _sem{0}; // Semaphore to manage task availability
 };
 
 } // namespace parallel
 } // namespace sirius
 