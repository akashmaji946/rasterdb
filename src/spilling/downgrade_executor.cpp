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

#include "spilling/downgrade_executor.hpp"

namespace sirius {
namespace parallel {

void DowngradeExecutor::Schedule(sirius::unique_ptr<ITask> task) {
    // Downgrade-specific scheduling logic
    auto downgrade_task = CastToDowngradeTask(task.get());
    if (!downgrade_task) {
        // If it's not a DowngradeTask, use the parent's implementation
        ITaskExecutor::Schedule(std::move(task));
        return;
    }

    // Schedule the downgrade task using the parent's method
    ITaskExecutor::Schedule(std::move(task));
}

DowngradeTask* DowngradeExecutor::CastToDowngradeTask(ITask* task) {
    // Safely cast to DowngradeTask
    return dynamic_cast<DowngradeTask*>(task);
}

void DowngradeExecutor::Start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
        return;
    }
    OnStart();
    threads_.reserve(config_.num_threads);
    for (int i = 0; i < config_.num_threads; ++i) {
        threads_.push_back(
        sirius::make_unique<TaskExecutorThread>(sirius::make_unique<sirius::thread>(&DowngradeExecutor::WorkerLoop, this, i)));
    }
}

void DowngradeExecutor::Stop() {
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false)) {
        return;
    }
    OnStop();
    for (auto& thread : threads_) {
        if (thread->internal_thread_->joinable()) {
        thread->internal_thread_->join();
        }
    }
    threads_.clear();
}

void DowngradeExecutor::WorkerLoop(int worker_id) {
    while (true) {
        if (!running_.load()) {
            // Executor is stopped.
            break;
        }
        auto task = task_queue_->Pull();
            if (task == nullptr) {
            // Task queue is closed.
            break;
        }
        try {
            task->Execute();
        } catch (const std::exception& e) {
            OnTaskError(worker_id, std::move(task), e);
        }
    }
}

} // namespace parallel
} // namespace sirius