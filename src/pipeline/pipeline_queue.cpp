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

#include "pipeline/pipeline_queue.hpp"
#include "config.hpp"

namespace sirius {
namespace parallel {

void pipeline_queue::open() {
    sirius::lock_guard<sirius::mutex> lock(_mutex);
    _is_open = true;
}

void pipeline_queue::close() {
    // Wake up all waiting threads by releasing the semaphore enough times
    for (int i = 0; i < Config::NUM_PIPELINE_EXECUTOR_THREADS; ++i) {
        _sem.release();
    }
    sirius::lock_guard<sirius::mutex> lock(_mutex);
    _is_open = false;
}

void pipeline_queue::push(sirius::unique_ptr<itask> task) {
    // Convert itask to gpu_pipeline_task - since we know it's a gpu_pipeline_task
    auto gpu_task = sirius::unique_ptr<gpu_pipeline_task>(static_cast<gpu_pipeline_task*>(task.release()));
    sirius::lock_guard<sirius::mutex> lock(_mutex);
    if (gpu_task && _is_open) {
        _task_queue.push(std::move(gpu_task));
    }
    _sem.release(); // signal that one item is available
}

sirius::unique_ptr<itask> pipeline_queue::pull() {
    _sem.acquire(); // wait until there's something
    sirius::lock_guard<sirius::mutex> lock(_mutex);
    
    // If the queue is closed and empty, return nullptr to signal shutdown
    if (!_is_open && _task_queue.empty()) {
        return nullptr;
    }
    
    // If there's a task available, return it
    if (!_task_queue.empty()) {
        auto task = std::move(_task_queue.front());
        _task_queue.pop();
        return task;
    }
    
    // Queue is empty but might be open (spurious semaphore release from close())
    return nullptr;
}

bool pipeline_queue::is_empty() const {
    sirius::lock_guard<sirius::mutex> lock(_mutex);
    return _task_queue.empty();
}

} // namespace parallel
} // namespace sirius

