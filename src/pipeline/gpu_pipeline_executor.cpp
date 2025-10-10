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

#include "pipeline/gpu_pipeline_executor.hpp"

namespace sirius {
namespace parallel {

void GPUPipelineExecutor::Schedule(sirius::unique_ptr<ITask> task) {
    // GPU-specific scheduling logic
    auto gpu_task = CastToGPUPipelineTask(task.get());
    if (!gpu_task) {
        // If it's not a GPUPipelineTask, use the parent's implementation
        ITaskExecutor::Schedule(std::move(task));
        return;
    }

    // Schedule the GPU task using the parent's method
    ITaskExecutor::Schedule(std::move(task));
}

GPUPipelineTask* GPUPipelineExecutor::CastToGPUPipelineTask(ITask* task) {
    // Safely cast to GPUPipelineTask
    return dynamic_cast<GPUPipelineTask*>(task);
}

} // namespace parallel
} // namespace sirius