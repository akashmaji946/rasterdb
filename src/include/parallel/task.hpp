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

#include "helper/helper.hpp"
#include "memory/memory_reservation.hpp"

namespace sirius {
namespace parallel {

using sirius::memory::Reservation;

/**
 * @brief Interface for storing task local states.
 * 
 * This class is used to primarily used to store state local to a specific task (such as its memory reservation). Additionally, DuckDB
 * requires each task to have its own local state object.
 */
class ITaskLocalState {
public:
  virtual ~ITaskLocalState() = default;
};

/**
 * @brief Interface for storing task global states
 * 
 * While this is not strictly necessary, it is useful to have a global state object that can be shared across multiple tasks.
 * This will primarily be used by the DuckDB scan tasks to store common state between the scan task. For storing any common state
 * between tasks we implement, we can either derive a new class from this interface or use the pipeline to store that state.
 */
class ITaskGlobalState {
public:
  virtual ~ITaskGlobalState() = default;
};

/**
 * @brief Interface for concrete executor tasks.
 * 
 * The primary purpose of this interface is to provide a common interface for all tasks. Anyone who wants to submit a task to an
 * executor should create the derived class that the executor supports. 
 */
class ITask {
public:
  /**
   * @brief Construct a new ITask object
   * 
   * Note that the ITask takes ownership of the local state
   * 
   * @param local_state The local state for this task
   * @param global_state The global state shared across multiple tasks
   */
  ITask(sirius::unique_ptr<ITaskLocalState> local_state, sirius::shared_ptr<ITaskGlobalState> global_state)
      : local_state_(std::move(local_state)), global_state_(global_state), reservation_(nullptr) 
  {

  }

  virtual ~ITask() = default;

  // Non-copyable and movable.
  ITask(const ITask&) = delete;
  ITask& operator=(const ITask&) = delete;
  ITask(ITask&&) = default;
  ITask& operator=(ITask&&) = default;

  /**
   * @brief Method that can be used to attach an reservation to the task
   * 
   * Each Executor should ensure that it attaches a reservation to the task before executing it.
   * Note that task should take ownership of the reservation so that the reservation gets released
   * when the task goes out of scope.
   * 
   * @param reservation The reservation to attach to the task
   */
  virtual void AttachReservation(sirius::shared_ptr<memory::Reservation> reservation) = 0;

  /**
   * @brief Function to execute the task
   * 
   * The actual implementation of the task should be implemented by the derived class in this method 
   */
  virtual void Execute() = 0;

protected:
  sirius::unique_ptr<ITaskLocalState> local_state_; // The local state for this task
  sirius::shared_ptr<ITaskGlobalState> global_state_; // The global state shared across multiple tasks
  sirius::unique_ptr<memory::Reservation> reservation_; // The memory reservation attached to the task
};

} // namespace parallel
} // namespace sirius