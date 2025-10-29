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
#include "parallel/task.hpp"
#include <parallel/task_queue.hpp>

// duckdb
#include <blockingconcurrentqueue.h>

// standard library
#include <atomic>

namespace sirius::parallel
{
class DuckDBScanTaskQueue : public ITaskQueue
{
public:
  DuckDBScanTaskQueue()           = default;
  ~DuckDBScanTaskQueue() override = default;

  void Open() override
  {
    is_open_.store(true, std::memory_order_release);
  }

  void Close() override
  {
    bool was_open = is_open_.exchange(false, std::memory_order_acq_rel);
    if (was_open)
    {
      // First close. Push nullptr sentinel to signal closure.
      task_queue_.enqueue(sirius::unique_ptr<ITask>{});
    }
  }

  void Push(sirius::unique_ptr<ITask> task) override
  {
    // Caller ensures queue is open (as per ITaskQueue contract).
    task_queue_.enqueue(std::move(task));
  }

  // Single-consumer, blocking, drain-and-stop
  sirius::unique_ptr<ITask> Pull() override
  {
    sirius::unique_ptr<ITask> task;
    // Blocks until an task appears (or sentinel is seen).
    task_queue_.wait_dequeue(task);
    if (!task)
    {
      // nullptr sentinel signals closed + drained
      return nullptr;
    }
    return task;
  }

private:
  std::atomic<bool> is_open_{false};
  duckdb_moodycamel::BlockingConcurrentQueue<sirius::unique_ptr<ITask>> task_queue_;
};

} // namespace sirius::parallel