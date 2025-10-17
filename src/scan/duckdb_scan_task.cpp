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

#include "scan/duckdb_scan_executor.hpp"
#include "scan/duckdb_scan_task.hpp"

namespace sirius {
namespace parallel {

void DuckDBScanTask::Execute() {
    // create IDataRepresentation
    // create DataBatch
    std::cout << "Scan Task is executing\n";
    auto batch_id = global_state_->Cast<DuckDBScanGlobalSourceState>().data_repository_.GetNextDataBatchId();
    std::cout << "Scan Task is executing\n";
    sirius::unique_ptr<sirius::DataBatch> data_batch = sirius::make_unique<sirius::DataBatch>(batch_id, nullptr);
    // push DataBatch to data_repository_
    global_state_->Cast<DuckDBScanGlobalSourceState>().data_repository_.AddNewDataBatch(pipeline_id_, std::move(data_batch));
    // notify TaskCreator about task completion
    MarkTaskCompletion();
}

void DuckDBScanTask::MarkTaskCompletion() {
    // notify TaskCreator about task completion
    std::cout << "Marking Scan Task Completion\n";
    auto message = sirius::make_unique<sirius::TaskCompletionMessage>(task_id_, pipeline_id_, sirius::Source::SCAN);
    global_state_->Cast<DuckDBScanGlobalSourceState>().message_queue_.EnqueueMessage(std::move(message));
}

} // namespace parallel
} // namespace sirius