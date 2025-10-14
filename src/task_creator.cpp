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

#include "task_creator.hpp"
#include "gpu_executor.hpp"

namespace sirius {

TaskCreator::TaskCreator(
    DataRepository& data_repository,
    parallel::GPUPipelineTaskQueue &gpu_pipeline_executor
    ) : data_repository_(data_repository),
        gpu_pipeline_executor_(gpu_pipeline_executor),
        task_completion_message_queue_(), 
        coordinator_(nullptr),
        running_(false) {
}

// spawn the internal thread that runs the task creator loop
void
TaskCreator::Start() {
  bool expected = false;
  if (!running_.compare_exchange_strong(expected, true)) {
    return;
  }
  internal_thread_ = sirius::make_unique<sirius::thread>(&TaskCreator::WorkerLoop, this);
}

void
TaskCreator::Stop() {
  bool expected = true;
  if (!running_.compare_exchange_strong(expected, false)) {
    return;
  }
  // stop the internal thread
  if (internal_thread_ && internal_thread_->joinable()) {
    internal_thread_->join();
    internal_thread_.reset();
  }
}

void 
TaskCreator::WorkerLoop() {
    Wait();  // wait for signal from B
    std::cout << "Creator: Got signal from Coordinator\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    std::cout << "Creator: Done processing, signaling Coordinator\n";
    if (coordinator_ == nullptr) {
        throw std::runtime_error("Coordinator is not set in TaskCreator");
    }
    coordinator_->Signal();  // signal B
}

void 
TaskCreator::Signal() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        ready = true;
    }
    cv.notify_one();
}

void 
TaskCreator::Wait() {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [this] { return ready; });
    ready = false;
}

} // namespace sirius