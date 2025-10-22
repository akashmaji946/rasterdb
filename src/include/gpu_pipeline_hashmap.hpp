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
#include "scan/duckdb_physical_table_scan.hpp"
#include "gpu_pipeline.hpp"
#include "helper/helper.hpp"

namespace sirius {

class DuckDBScanMetadata {
public:
    //constructor initializing function and op
    DuckDBScanMetadata(duckdb::ExecutionContext& context, duckdb::DuckDBPhysicalTableScan& op)
        : context_(context), op_(op) {

        }
    //
    ~DuckDBScanMetadata() = default;   
    duckdb::ExecutionContext& context_; // The execution context for the scan operation 
    duckdb::DuckDBPhysicalTableScan& op_; // The GPU physical table scan operator associated with this executor
};

class GPUPipelineHashMap {
public:
    GPUPipelineHashMap(duckdb::vector<duckdb::shared_ptr<duckdb::GPUPipeline>> vec) : vec_(std::move(vec)) {};
    ~GPUPipelineHashMap() = default;
    duckdb::vector<duckdb::shared_ptr<duckdb::GPUPipeline>> vec_;
    sirius::unordered_map<size_t, sirius::shared_ptr<DuckDBScanMetadata>> scan_metadata_map_; // Map of pipeline IDs to their associated scan metadata
};

} //namespace sirius