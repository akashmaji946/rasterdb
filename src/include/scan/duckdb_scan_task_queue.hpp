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

// sirius
#include <parallel/task_queue.hpp>

// duckdb
#include <blockingconcurrentqueue.h>

// standard library
#include <atomic>

namespace sirius::parallel
{

//===--------------------------------------------------===//
// DuckDBScanTaskQueue
//===--------------------------------------------------===//
class DuckDBScanTaskQueue : public ITaskQueue
{
  static constexpr size_t PULL_TIMEOUT_US = 100;

public:
  DuckDBScanTaskQueue()           = default;
  ~DuckDBScanTaskQueue() override = default;

  void Open() override
  {
    is_open_.store(true, std::memory_order::release);
  }

  void Close() override
  {
    is_open_.store(false, std::memory_order::release);
  }

  void Push(sirius::unique_ptr<ITask> task) override
  {
    // We must assume the queue is open as a precondition, given the implementation in ITaskQueue
    task_queue_.enqueue(std::move(task));
  }

  // Wait until a task is available or the queue is closed.
  unique_ptr<ITask> Pull() override
  {
    unique_ptr<ITask> scan_task;
    while (true)
    {
      if (task_queue_.wait_dequeue_timed(scan_task, PULL_TIMEOUT_US))
      {
        return scan_task;
      }
      // If closed, return
      if (!is_open_.load(std::memory_order::acquire) && task_queue_.size_approx() == 0)
      {
        return nullptr;
      }
    }
  }

private:
  std::atomic<bool> is_open_{false};
  duckdb_moodycamel::BlockingConcurrentQueue<sirius::unique_ptr<ITask>> task_queue_;
};

} // namespace sirius::parallel