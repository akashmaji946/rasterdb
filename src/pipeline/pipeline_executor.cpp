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

#include "pipeline/pipeline_executor.hpp"
#include "pipeline/pipeline_queue.hpp"

namespace sirius {
namespace parallel {

pipeline_executor::pipeline_executor(task_executor_config config)
    : itask_executor(sirius::make_unique<pipeline_queue>(), config) {}

void pipeline_executor::schedule(sirius::unique_ptr<itask> task) {
    _task_queue->push(std::move(task));
}
 
void pipeline_executor::start() {
    bool expected = false;
    if (!_running.compare_exchange_strong(expected, true)) {
        return;
    }
    on_start();
    _threads.reserve(_config.num_threads);
    for (int i = 0; i < _config.num_threads; ++i) {
        _threads.push_back(
        sirius::make_unique<task_executor_thread>(sirius::make_unique<sirius::thread>(&pipeline_executor::worker_loop, this, i)));
    }
}

void pipeline_executor::stop() {
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

void pipeline_executor::worker_loop(int worker_id) {
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
            // TODO
            // Make reservation (prioritize GPU with the same memory space as the input)
            // If there is a reservation, dispatch to the corresponding GPU executor
            // If no reservation, use some policy to pick a GPU executor
            // Dispatch to the selected GPU executor
            dispatch_to_gpu_executor(std::move(task), 0); // For now we just dispatch to GPU 0
        } catch (const std::exception& e) {
            on_task_error(worker_id, std::move(task), e);
        }
    }
}

void pipeline_executor::dispatch_to_gpu_executor(sirius::unique_ptr<itask> task, int gpu_id) {
    if (gpu_id < 0 || gpu_id >= static_cast<int>(_gpu_executors.size())) {
        throw std::runtime_error("Invalid GPU ID: " + std::to_string(gpu_id));
    }
    _gpu_executors[gpu_id]->schedule(std::move(task));
}

} // namespace parallel
} // namespace sirius