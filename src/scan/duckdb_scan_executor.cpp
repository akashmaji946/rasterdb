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

#include <scan/duckdb_scan_task.hpp>
#include <scan/duckdb_scan_task_executor.hpp>

namespace sirius
{
namespace parallel
{

void DuckDBScanTaskExecutor::Schedule(sirius::unique_ptr<experimental::DuckDBScanTask> task)
{
  // Here we set the number of threads for duckdb's executor
  task->SetNumThreads(num_duckdb_threads_);
  task_queue_->Push(std::move(task));
  total_tasks_.fetch_add(1);
}

// void DuckDBScanExecutor::Schedule(sirius::unique_ptr<ITask> task) {
//     // Call the parent's Schedule method to handle the task execution
//     ITaskExecutor::Schedule(std::move(task));
// }

// void DuckDBScanExecutor::Start() {
//   bool expected = false;
//   if (!running_.compare_exchange_strong(expected, true)) {
//     return;
//   }
//   OnStart();
//   threads_.reserve(config_.num_threads);
//   for (int i = 0; i < config_.num_threads; ++i) {
//     threads_.push_back(
//       sirius::make_unique<TaskExecutorThread>(sirius::make_unique<sirius::thread>(&DuckDBScanExecutor::WorkerLoop,
//       this, i)));
//   }
// }

// void DuckDBScanExecutor::Stop() {
//   bool expected = true;
//   if (!running_.compare_exchange_strong(expected, false)) {
//     return;
//   }
//   OnStop();
//   for (auto& thread : threads_) {
//     if (thread->internal_thread_->joinable()) {
//       thread->internal_thread_->join();
//     }
//   }
//   threads_.clear();
// }

// void DuckDBScanExecutor::WorkerLoop(int worker_id) {
//     while (true) {
//         if (!running_.load()) {
//             // Executor is stopped.
//             break;
//         }
//         auto task = task_queue_->Pull();
//             if (task == nullptr) {
//             // Task queue is closed.
//             break;
//         }
//         try {
//             task->Execute();
//         } catch (const std::exception& e) {
//             OnTaskError(worker_id, std::move(task), e);
//         }
//     }
// }

} // namespace parallel
} // namespace sirius