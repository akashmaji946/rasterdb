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
#include "memory/memory_reservation.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "data/data_repository.hpp"
#include "memory/memory_space.hpp"

namespace sirius {
namespace parallel {

/**
 * @brief Executor specialized for executing GPU pipeline operations.
 * 
 * This executor inherits from itask_executor and uses a gpu_pipeline_task_queue for
 * task scheduling. It manages a pool of threads dedicated to executing GPU pipeline
 * tasks with specialized GPU resource management.
 */
class gpu_pipeline_executor : public itask_executor {
public:
    /**
     * @brief Constructs a new gpu_pipeline_executor with task execution configuration
     * 
     * @param config Configuration for the task executor (thread count, retry policy, etc.)
     */
    explicit gpu_pipeline_executor(task_executor_config config, memory::memory_space* mem_space);

    /**
     * @brief Destructor for the gpu_pipeline_executor.
     */
    ~gpu_pipeline_executor() override = default;

    // Non-copyable but movable
    gpu_pipeline_executor(const gpu_pipeline_executor&) = delete;
    gpu_pipeline_executor& operator=(const gpu_pipeline_executor&) = delete;
    gpu_pipeline_executor(gpu_pipeline_executor&&) = default;
    gpu_pipeline_executor& operator=(gpu_pipeline_executor&&) = default;

    /**
     * @brief Schedules a task for execution with GPU-specific logic
     * 
     * Overrides the base class schedule method to provide specialized scheduling
     * behavior for GPU pipeline operations, including resource allocation and
     * GPU context management.
     * 
     * @param task The task to schedule (must be a gpu_pipeline_task)
     */
    void schedule(sirius::unique_ptr<itask> task) override;

    /**
     * @brief Main worker loop for executing GPU pipeline tasks
     * 
     * Each worker thread runs this loop to continuously pull and execute GPU
     * pipeline tasks from the queue. Handles GPU-specific operations including
     * kernel launches, memory transfers, and synchronization.
     * 
     * @param worker_id The unique identifier for this worker thread
     */
    void worker_loop(int worker_id) override;

    /**
     * @brief Starts the executor and initializes worker threads
     * 
     * Initializes the thread pool and begins accepting tasks for execution.
     */
    void start() override;

    /**
     * @brief Stops the executor and cleanly shuts down worker threads
     * 
     * Stops accepting new tasks and waits for all worker threads to complete
     * their current tasks before shutting down.
     */
    void stop() override;

    /**
     * @brief Get the memory space view associated with this executor
     * 
     * @return memory::memory_space* Pointer to the memory space
     */
    memory::memory_space* get_memory_space_view();

private:
    /**
     * @brief Safely casts itask to gpu_pipeline_task with type validation
     * 
     * @param task The itask pointer to cast
     * @return gpu_pipeline_task* The casted gpu_pipeline_task pointer
     * @throws std::bad_cast if the task is not of type gpu_pipeline_task
     */
    gpu_pipeline_task* cast_to_gpu_pipeline_task(itask* task);
    memory::memory_space* _memory_space_view; // this is supposed to be the memory space associated with this pipeline executor
};

} // namespace parallel
} // namespace sirius