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

#include "task.hpp"

namespace sirius {
namespace parallel {

/**
 * @brief Interface for concrete scheduling policies.
 */
class ITaskQueue {
public:
  /**
   * @brief Destructor for the ITaskQueue.
   */
  virtual ~ITaskQueue() = default;

  /**
   * @brief Needs to be called before any tasks can be scheduled or pulled.
   */
  virtual void Open() = 0;

  /**
   * @brief Close the scheduler from accepting new tasks or returning tasks to execute
   */
  virtual void Close() = 0;

  /**
   * @brief Push a new task to be scheduled.
   * 
   * @param task The task to be scheduled
   * @throws std::runtime_error If the scheduler is not currently accepting requests
   */
  virtual void Push(sirius::unique_ptr<ITask> task) = 0;

  /**
   * @brief Pull a task to execute.
   * 
   * Note that this is a blocking call and will wait until a task is available or the scheduler is closed.
   * 
   * @return A unique pointer to the task to execute
   * @throws std::runtime_error If the scheduler is not currently stopped and thus not returning tasks
   */
  virtual sirius::unique_ptr<ITask> Pull() = 0;
};

} // namespace parallel
} // namespace sirius