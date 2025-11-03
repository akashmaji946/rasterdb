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
#include "creator/sirius_task_creator.hpp"
#include "data/data_repository.hpp"
#include "data/simple_data_repository_level.hpp"
#include "pipeline/gpu_pipeline_executor.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "downgrade/downgrade_executor.hpp"
#include "downgrade/downgrade_task.hpp"
#include "creator/downgrade_task_creator.hpp"
#include "scan/duckdb_scan_executor.hpp"
#include "scan/duckdb_scan_task.hpp"
#include "memory/memory_reservation.hpp"
#include "helper/helper.hpp"

namespace sirius {

/**
 * @brief Global context class managing all core Sirius system components.
 * 
 * sirius_context is a singleton that holds and manages all global Sirius components
 * including task creators, executors, data repository, and memory management.
 * It provides centralized access to these components throughout the system and
 * ensures proper initialization and coordination between different subsystems.
 * 
 * This design pattern allows for clean dependency injection and simplifies
 * system-wide resource management and configuration.
 */
class sirius_context {
public:
    /**
     * @brief Constructs and initializes the sirius_context
     */
    explicit sirius_context();
    
    /**
     * @brief Destructor for sirius_context
     */
    ~sirius_context() = default;
    
    /**
     * @brief Gets the singleton instance of sirius_context
     * 
     * @return sirius_context& Reference to the singleton instance
     */
    static sirius_context& get_instance() {
        static sirius_context instance;
        return instance;
    }
    
    // Non-copyable and non-movable singleton
    sirius_context(const sirius_context&) = delete;
    sirius_context& operator=(const sirius_context&) = delete;
    sirius_context(sirius_context&&) = delete;
    sirius_context& operator=(sirius_context&&) = delete;

    /**
     * @brief Gets the main task creator for the system
     * 
     * @return sirius_task_creator& Reference to the task creator
     */
    sirius_task_creator& get_task_creator() {
        return _sirius_task_creator;
    }

    /**
     * @brief Gets the memory reservation manager for the system
     * 
     * @return sirius::memory::MemoryReservationManager& Reference to the memory reservation manager
     */
    sirius::memory::MemoryReservationManager& get_memory_reservation_manager() {
        return memory::MemoryReservationManager::getInstance();
    }

    /**
     * @brief Gets the GPU pipeline executor
     * 
     * @return parallel::gpu_pipeline_executor& Reference to the GPU pipeline executor
     */
    parallel::gpu_pipeline_executor& get_gpu_pipeline_executor() {
        return _gpu_pipeline_executor;
    }

    /**
     * @brief Gets the downgrade executor for memory tier management
     * 
     * @return parallel::downgrade_executor& Reference to the downgrade executor
     */
    parallel::downgrade_executor& get_downgrade_executor() {
        return _downgrade_executor;
    }

    /**
     * @brief Gets the DuckDB scan executor
     * 
     * @return parallel::duckdb_scan_executor& Reference to the DuckDB scan executor
     */
    parallel::duckdb_scan_executor& get_duckdb_scan_executor() {
        return _duckdb_scan_executor;
    }

    /**
     * @brief Gets the downgrade task creator
     * 
     * @return downgrade_task_creator& Reference to the downgrade task creator
     */
    downgrade_task_creator& get_downgrade_task_creator() {
        return _downgrade_task_creator;
    }

    /**
     * @brief Gets the central data repository manager
     * 
     * @return data_repository_manager& Reference to the data repository manager
     */
    data_repository_manager& get_data_repository_manager() {
        return _data_repo_mgr;
    }

private:
    data_repository_manager _data_repo_mgr;                          ///< Central repository for managing data batches
    parallel::gpu_pipeline_executor _gpu_pipeline_executor;     ///< Executor for GPU pipeline operations
    parallel::downgrade_executor _downgrade_executor;          ///< Executor for memory tier downgrade operations
    parallel::duckdb_scan_executor _duckdb_scan_executor;       ///< Executor for DuckDB scan operations
    sirius_task_creator _sirius_task_creator;                                ///< Main task creator for coordinating task scheduling
    downgrade_task_creator _downgrade_task_creator;             ///< Specialized task creator for memory downgrade operations
};

} // namespace sirius