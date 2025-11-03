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

#include "pipeline/gpu_pipeline_executor.hpp"
#include "pipeline/gpu_pipeline_queue.hpp"

namespace sirius {
namespace parallel {

gpu_pipeline_executor::gpu_pipeline_executor(task_executor_config config)
    : itask_executor(sirius::make_unique<gpu_pipeline_task_queue>(), config) {}

void gpu_pipeline_executor::schedule_gpu_task(sirius::unique_ptr<gpu_pipeline_task> gpu_task) {
    // Convert to itask and use parent's schedule method
    schedule(std::move(gpu_task));
}

void gpu_pipeline_executor::schedule(sirius::unique_ptr<itask> task) {
     // GPU-specific scheduling logic
     auto gpu_task = cast_to_gpu_pipeline_task(task.get());
     if (!gpu_task) {
         // If it's not a gpu_pipeline_task, use the parent's implementation
         itask_executor::schedule(std::move(task));
         return;
     }
 
     // Schedule the GPU task using the parent's method
     itask_executor::schedule(std::move(task));
 }
 
 void gpu_pipeline_executor::start() {
     bool expected = false;
     if (!_running.compare_exchange_strong(expected, true)) {
         return;
     }
     on_start();
     _threads.reserve(_config.num_threads);
     for (int i = 0; i < _config.num_threads; ++i) {
         _threads.push_back(
         sirius::make_unique<task_executor_thread>(sirius::make_unique<sirius::thread>(&gpu_pipeline_executor::worker_loop, this, i)));
     }
 }
 
 void gpu_pipeline_executor::stop() {
     bool expected = true;
     if (!_running.compare_exchange_strong(expected, false)) {
         return;
     }
     on_stop();
     for (auto& thread : _threads) {
         if (thread->_internal_thread->joinable()) {
         thread->_internal_thread->join();
         }
     }
     _threads.clear();
 }
 
 void gpu_pipeline_executor::worker_loop(int worker_id) {
     while (true) {
         if (!_running.load()) {
             // Executor is stopped.
             break;
         }
         auto task = _task_queue->pull();
             if (task == nullptr) {
             // Task queue is closed.
             break;
         }
         try {
             task->execute();
         } catch (const std::exception& e) {
             on_task_error(worker_id, std::move(task), e);
         }
     }
 }
 
 gpu_pipeline_task* gpu_pipeline_executor::cast_to_gpu_pipeline_task(itask* task) {
     // Safely cast to gpu_pipeline_task
     return dynamic_cast<gpu_pipeline_task*>(task);
 }
 
 } // namespace parallel
 } // namespace sirius