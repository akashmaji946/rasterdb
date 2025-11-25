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

#include <vector>

#include "data/common.hpp"
#include "memory/fixed_size_host_memory_resource.hpp"
#include "memory/memory_space.hpp"
#include "memory/result_table.hpp"
#include "helper/helper.hpp"

#include "duckdb/common/types/data_chunk.hpp"

namespace sirius {

/**
 * @brief Data representation for the result that get returned to duckdb
 * 
 * This represents a Data Batch that is an output of an query. The reason that this is different from
 * host_table_representation as that represents 
 */
class duckdb_table_representation : public idata_representation { 
public:  

    /**
     * @brief Construct a new duckdb_table_repersentation object
     * 
     * @param result_table Internal representation storing the cudf table in an output friendly format
     * @param data_chunks The Data Chunks representing the table results that can be passed directly to DuckDB
     * @param memory_space The memory space that this table resides in
     */
    duckdb_table_representation(sirius::unique_ptr<sirius::memory::result_table_allocation> result_table, std::vector<sirius::unique_ptr<duckdb::DataChunk>> output_vectors, sirius::memory::memory_space& memory_space);

    /**
     * @brief Get the size of the data representation in bytes
     * 
     * @return std::size_t The number of bytes used to store this representation
     */
    std::size_t get_size_in_bytes() const override;

    /**
     * @brief Returns a reference to the result data chunks
     */
    std::vector<duckdb::unique_ptr<duckdb::DataChunk>>& get_output_chunks() { 
        return _output_chunks;
    }

    /**
     * @brief Convert this CPU table representation to a different memory tier
     * 
     * @param target_memory_space The target memory space to convert to
     * @param stream CUDA stream to use for memory operations
     * @return sirius::unique_ptr<idata_representation> A new data representation in the target tier
     */
    sirius::unique_ptr<idata_representation> convert_to_memory_space(sirius::memory::memory_space& target_memory_space, rmm::cuda_stream_view stream = rmm::cuda_stream_default) override;

private: 
    sirius::unique_ptr<sirius::memory::result_table_allocation> _result_table;
    std::vector<duckdb::unique_ptr<duckdb::DataChunk>> _output_chunks; 
};

}