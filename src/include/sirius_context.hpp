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
#include "scan/duckdb_scan_task_executor.hpp"
#include "scan/duckdb_scan_task.hpp"
#include "memory/memory_reservation.hpp"
#include "helper/helper.hpp"

namespace sirius {

/**
 * @brief Global context class managing all core Sirius system components.
 * 
 * SiriusContext is a singleton that holds and manages all global Sirius components
 * including task creators, executors, data repository, and memory management.
 * It provides centralized access to these components throughout the system and
 * ensures proper initialization and coordination between different subsystems.
 * 
 * This design pattern allows for clean dependency injection and simplifies
 * system-wide resource management and configuration.
 */
class SiriusContext {
public:
    /**
     * @brief Constructs and initializes the SiriusContext
     */
    explicit SiriusContext();
    
    /**
     * @brief Destructor for SiriusContext
     */
    ~SiriusContext() = default;
    
    /**
     * @brief Gets the singleton instance of SiriusContext
     * 
     * @return SiriusContext& Reference to the singleton instance
     */
    static SiriusContext& GetInstance() {
        static SiriusContext instance;
        return instance;
    }
    
    // Non-copyable and non-movable singleton
    SiriusContext(const SiriusContext&) = delete;
    SiriusContext& operator=(const SiriusContext&) = delete;
    SiriusContext(SiriusContext&&) = delete;
    SiriusContext& operator=(SiriusContext&&) = delete;

    /**
     * @brief Gets the main task creator for the system
     * 
     * @return TaskCreator& Reference to the task creator
     */
    TaskCreator& GetTaskCreator() {
        return task_creator_;
    }

    /**
     * @brief Gets the memory reservation manager for the system
     * 
     * @return sirius::memory::MemoryReservationManager& Reference to the memory reservation manager
     */
    sirius::memory::MemoryReservationManager& GetMemoryReservationManager() {
        return memory::MemoryReservationManager::getInstance();
    }

    /**
     * @brief Gets the GPU pipeline executor
     * 
     * @return parallel::GPUPipelineExecutor& Reference to the GPU pipeline executor
     */
    parallel::GPUPipelineExecutor& GetGPUPipelineExecutor() {
        return gpu_pipeline_executor_;
    }

    /**
     * @brief Gets the downgrade executor for memory tier management
     * 
     * @return parallel::DowngradeExecutor& Reference to the downgrade executor
     */
    parallel::DowngradeExecutor& GetDowngradeExecutor() {
        return downgrade_executor_;
    }

    /**
     * @brief Gets the DuckDB scan executor
     * 
     * @return parallel::DuckDBScanExecutor& Reference to the DuckDB scan executor
     */
    parallel::DuckDBScanTaskExecutor& GetDuckDBScanTaskExecutor() {
        return duckdb_scan_task_executor_;
    }

    /**
     * @brief Gets the downgrade task creator
     * 
     * @return DowngradeTaskCreator& Reference to the downgrade task creator
     */
    DowngradeTaskCreator& GetDowngradeTaskCreator() {
        return downgrade_task_creator_;
    }

    /**
     * @brief Gets the central data repository
     * 
     * @return DataRepository& Reference to the data repository
     */
    DataRepository& GetDataRepository() {
        return data_repository_;
    }

private:
    TaskCreator task_creator_;                                   ///< Main task creator for coordinating task scheduling
    DowngradeTaskCreator downgrade_task_creator_;                ///< Specialized task creator for memory downgrade operations
    DataRepository data_repository_;                             ///< Central repository for managing data batches
    parallel::GPUPipelineExecutor gpu_pipeline_executor_;        ///< Executor for GPU pipeline operations
    parallel::DowngradeExecutor downgrade_executor_;             ///< Executor for memory tier downgrade operations
    parallel::DuckDBScanTaskExecutor duckdb_scan_task_executor_; ///< Executor for DuckDB scan operations
};

} // namespace sirius