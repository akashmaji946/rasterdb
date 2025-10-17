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

namespace sirius {

enum class Source {
    SCAN,
    PIPELINE
};

// Message indicating the completion of a task, notifying task creator that it should check if new tasks can be created
struct TaskCompletionMessage {
    uint64_t task_id;
    uint64_t pipeline_id;
    Source source;
};

class TaskCompletionMessageQueue {
public:
    TaskCompletionMessageQueue() = default;
    ~TaskCompletionMessageQueue() = default;
    
    void EnqueueMessage(sirius::unique_ptr<TaskCompletionMessage> message) {
        sirius::lock_guard<sirius::mutex> lock(mutex_);
        message_queue_.push(std::move(message));
        sem_.release(); // signal that one item is available
    }
    
    sirius::unique_ptr<TaskCompletionMessage> DequeueMessage() {
        sem_.acquire(); // wait until there's something
        sirius::lock_guard<sirius::mutex> lock(mutex_);
        if (message_queue_.empty()) {
            return nullptr;
        }
        auto message = std::move(message_queue_.front());
        message_queue_.pop();
        return message;
    }

    sirius::unique_ptr<TaskCompletionMessage> PullMessage() {
        return DequeueMessage();
    }
    
private:
    mutex mutex_;
    sirius::queue<sirius::unique_ptr<TaskCompletionMessage>> message_queue_;
    std::counting_semaphore<> sem_{0}; // starts with 0 available permits
};

} // namespace sirius