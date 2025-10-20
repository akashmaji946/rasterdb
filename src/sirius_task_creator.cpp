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

#include "sirius_task_creator.hpp"
#include "gpu_executor.hpp"
#include "scan/duckdb_scan_executor.hpp"
#include "pipeline/gpu_pipeline_executor.hpp"
#include "gpu_pipeline_hashmap.hpp"
#include "gpu_physical_table_scan.hpp"

namespace sirius {

TaskCreator::TaskCreator(
    DataRepository& data_repository,
    parallel::GPUPipelineExecutor& gpu_pipeline_executor,
    parallel::DuckDBScanExecutor& duckdb_scan_executor
    ) : 
    ITaskCreator(), data_repository_(data_repository),
        gpu_pipeline_executor_(gpu_pipeline_executor),
        duckdb_scan_executor_(duckdb_scan_executor),
        task_completion_message_queue_(), 
        coordinator_(nullptr) {
}

sirius::unique_ptr<parallel::DuckDBScanTask>
TaskCreator::CreateScanTask(size_t pipeline_idx) {
    // create new scan task
    auto it = gpu_pipeline_hashmap_->scan_metadata_map_.find(pipeline_idx);
    if (it == gpu_pipeline_hashmap_->scan_metadata_map_.end()) {
        throw std::runtime_error("No scan metadata found for pipeline id " + std::to_string(pipeline_idx));
    }
    auto& scan_metadata = it->second;

    sirius::shared_ptr<parallel::DuckDBScanGlobalSourceState> global_state = 
        sirius::make_shared<parallel::DuckDBScanGlobalSourceState>(data_repository_, task_completion_message_queue_);

    sirius::unique_ptr<parallel::DuckDBScanTask> scan_task = 
        sirius::make_unique<parallel::DuckDBScanTask>(
            GetNextTaskId(), 
            pipeline_idx,
            scan_metadata,
            nullptr, // local state can be initialized inside the task
            global_state  // global state can be initialized inside the task
        );
    return scan_task;
}

sirius::unique_ptr<parallel::GPUPipelineTask>
TaskCreator::CreatePipelineTask(size_t pipeline_idx, sirius::vector<sirius::unique_ptr<DataBatch>> data_batches) {
    // create pipeline tasks for each pipeline in the gpu_pipeline_hashmap_ that can consume the data batch
    auto pipeline = gpu_pipeline_hashmap_->vec_[pipeline_idx];
    sirius::shared_ptr<parallel::GPUPipelineTaskGlobalState> global_state = 
        sirius::make_shared<parallel::GPUPipelineTaskGlobalState>(data_repository_, task_completion_message_queue_);

    // create new pipeline task
    sirius::unique_ptr<parallel::GPUPipelineTask> pipeline_task = 
        sirius::make_unique<parallel::GPUPipelineTask>(
            GetNextTaskId(), 
            pipeline_idx,
            pipeline,
            std::move(data_batches),
            nullptr, // local state can be initialized inside the task
            global_state  // global state can be initialized inside the task
        );
    return pipeline_task;
}

void 
TaskCreator::WorkerLoop() {
    Wait();  // wait for signal from B
    std::cout << "Creator: Got signal from Coordinator\n";
    // std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    for (size_t pipeline_idx = 0; pipeline_idx < gpu_pipeline_hashmap_->vec_.size(); pipeline_idx++) {
        auto pipeline = gpu_pipeline_hashmap_->vec_[pipeline_idx];
        auto source_type = pipeline->GetSource()->type;
        if (source_type == duckdb::PhysicalOperatorType::TABLE_SCAN) {
            std::cout << "Creator: Creating initial scan task for pipeline " << pipeline_idx << "\n";
            auto scan_task = CreateScanTask(pipeline_idx);
            ScheduleDuckDBScan(std::move(scan_task));
        }
    }

    int scan_finished = 0;
    int pipeline_finished = 0;
    int scan_created = 1;
    int pipeline_created = 0;
    while (true) {
        std::cout << "Creator: Waiting for task completion messages\n";
        auto message = task_completion_message_queue_.PullMessage();

        if (message->source == Source::PIPELINE) {
            pipeline_finished++;
            std::cout << "Creator: Received pipeline completion message for pipeline " << message->pipeline_id << "\n";
            if (pipeline_created < scan_created) {
                ScanRepository(message->pipeline_id);
                pipeline_created++;
            }
        } else if (message->source == Source::SCAN) {
            scan_finished++;
            std::cout << "Creator: Received scan completion message for pipeline " << message->pipeline_id << "\n";
            if (scan_created < 10) {
                auto scan_task = CreateScanTask(message->pipeline_id);
                ScheduleDuckDBScan(std::move(scan_task));
                scan_created++;
            }
            if (pipeline_created < scan_created) {
                ScanRepository(message->pipeline_id);
                pipeline_created++;
            }
        }

        std::cout << "Creator: scan_counter = " << scan_finished << ", pipeline_counter = " << pipeline_finished << "\n";
        if (scan_finished >= 10 && pipeline_finished >= 10) {
            break;
        }
    }
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

bool
TaskCreator::ScanRepository(size_t pipeline_idx) {
    sirius::unique_ptr<DataBatch> data_batch = data_repository_.levels_[pipeline_idx]->Cast<SimpleDataRepositoryLevel>().EvictDataBatch();
    if (data_batch != nullptr) {
        std::cout << "Creator: Found data batch for pipeline " << pipeline_idx << "\n";
        sirius::vector<sirius::unique_ptr<DataBatch>> data_batches;
        data_batches.push_back(std::move(data_batch));
        sirius::unique_ptr<parallel::GPUPipelineTask> pipeline_task = CreatePipelineTask(pipeline_idx, std::move(data_batches));
        SchedulePipelineTask(std::move(pipeline_task));
        return true;
    } else {
        std::cout << "Creator: No data batch found for pipeline " << pipeline_idx << "\n";
        return false;
    }
}

void
TaskCreator::ScheduleDuckDBScan(sirius::unique_ptr<parallel::DuckDBScanTask> scan_task) {
    std::cout << "Creator: Scheduling scan task " << "\n";
    duckdb_scan_executor_.Schedule(std::move(scan_task));
}

void
TaskCreator::SchedulePipelineTask(sirius::unique_ptr<parallel::GPUPipelineTask> pipeline_task) {
    std::cout << "Creator: Scheduling pipeline task " << "\n";
    gpu_pipeline_executor_.Schedule(std::move(pipeline_task));
}

} // namespace sirius