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

#include "spilling/downgrade_task.hpp"
#include "spilling/downgrade_executor.hpp"

namespace sirius {
namespace parallel {

void DowngradeTask::Execute() {
    std::cout << "Downgrade Task is executing\n";
    MarkTaskCompletion();
}

void DowngradeTask::MarkTaskCompletion() {
    // notify TaskCreator about task completion
    std::cout << "Marking Downgrade Task Completion\n";
    auto message = sirius::make_unique<sirius::TaskCompletionMessage>(task_id_, pipeline_id_, sirius::Source::PIPELINE);
    global_state_->Cast<DowngradeTaskGlobalState>().message_queue_.EnqueueMessage(std::move(message));
}


} // namespace parallel
} // namespace sirius