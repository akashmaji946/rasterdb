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

#include "parallel/task_creator.hpp"

namespace sirius {

// spawn the internal thread that runs the task creator loop
void ITaskCreator::Start() {
  bool expected = false;
  if (!running_.compare_exchange_strong(expected, true)) {
    return;
  }
  thread_ = sirius::make_unique<parallel::TaskExecutorThread>(sirius::make_unique<sirius::thread>(&ITaskCreator::WorkerLoop, this));
}

void ITaskCreator::Stop() {
  bool expected = true;
  if (!running_.compare_exchange_strong(expected, false)) {
    return;
  }
  // stop the internal thread
  if (thread_->internal_thread_ && thread_->internal_thread_->joinable()) {
    thread_->internal_thread_->join();
    thread_->internal_thread_.reset();}
}

void ITaskCreator::WorkerLoop() {
    // Default implementation does nothing
}

} // namespace sirius
