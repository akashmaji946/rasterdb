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

void DowngradeExecutor::WorkerLoop(int worker_id) {

}

} // namespace parallel
} // namespace sirius