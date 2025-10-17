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
#include "sirius_task_creator.hpp"
#include "data/data_repository.hpp"
#include "data/simple_data_repository_level.hpp"
#include "pipeline/gpu_pipeline_executor.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "spilling/downgrade_executor.hpp"
#include "spilling/downgrade_task.hpp"
#include "spilling/downgrade_task_creator.hpp"
#include "scan/duckdb_scan_executor.hpp"
#include "scan/duckdb_scan_task.hpp"
#include "memory/memory_reservation.hpp"
#include "helper/helper.hpp"

namespace sirius {


// Context class holding global Sirius components like TaskCreator, DataRepository, PipelineExecutor, Sc
class SiriusContext {
public:
    explicit SiriusContext();
    ~SiriusContext() = default;
    static SiriusContext& GetInstance() {
        static SiriusContext instance;
        return instance;
    }
    SiriusContext(const SiriusContext&) = delete;
    SiriusContext& operator=(const SiriusContext&) = delete;
    SiriusContext(SiriusContext&&) = delete;
    SiriusContext& operator=(SiriusContext&&) = delete;

    // Accessors for global components
    TaskCreator& GetTaskCreator() {
        return task_creator_;
    }

    sirius::memory::MemoryReservationManager& GetMemoryReservationManager() {
        return memory::MemoryReservationManager::getInstance();
    }

    parallel::GPUPipelineExecutor& GetGPUPipelineExecutor() {
        return gpu_pipeline_executor_;
    }

    parallel::DowngradeExecutor& GetDowngradeExecutor() {
        return downgrade_executor_;
    }

    parallel::DuckDBScanExecutor& GetDuckDBScanExecutor() {
        return duckdb_scan_executor_;
    }

    DowngradeTaskCreator& GetDowngradeTaskCreator() {
        return downgrade_task_creator_;
    }

    DataRepository& GetDataRepository() {
        return data_repository_;
    }

private :

    TaskCreator task_creator_;
    DowngradeTaskCreator downgrade_task_creator_;
    DataRepository data_repository_;
    parallel::GPUPipelineExecutor gpu_pipeline_executor_;
    parallel::DowngradeExecutor downgrade_executor_;
    parallel::DuckDBScanExecutor duckdb_scan_executor_;
};

} // namespace sirius