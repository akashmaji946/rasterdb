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

#include "task_queue.hpp"
#include "task_executor.hpp"
#include <condition_variable>
#include "helper/helper.hpp"

namespace sirius {

/**
 * Interface for a Task Creator, can be extended to support various kinds of task creation policies.
 */
class ITaskCreator {
public:
  ITaskCreator()
    : running_(false) {}
  
  virtual ~ITaskCreator() {
    Stop();
  }

  // Non-copyable and movable
  ITaskCreator(const ITaskCreator&) = delete;
  ITaskCreator& operator=(const ITaskCreator&) = delete;
  ITaskCreator(ITaskCreator&&) = default;
  ITaskCreator& operator=(ITaskCreator&&) = default;

  // Start worker threads
  virtual void Start();

  // Stop accepting new tasks, and join worker threads.
  virtual void Stop();

private:
  // Main thread loop.
  virtual void WorkerLoop();

private:
  sirius::atomic<bool> running_;
  sirius::unique_ptr<parallel::TaskExecutorThread> thread_;
};

} // namespace sirius
