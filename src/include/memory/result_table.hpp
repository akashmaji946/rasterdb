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

#include <cudf/types.hpp>

#include "fixed_size_host_memory_resource.hpp"
#include "helper/helper.hpp"

namespace sirius {
namespace memory {

/**
 * @brief Structure used to represent the location in a multi block allocation
 */
struct result_table_location { 
    size_t allocation_index = 0; // Index of the block in the allocation
    size_t block_offset = 0; // The offset within the block
};

/**
 * @brief Structure storing metatadata related to a specific column in the result table
 * 
 * Since we are spreading out the contents of a column across multiple blocks, we need to maintain
 * metadata in order to reconstruct all of the details and this column stores those details. Note
 * that all columns first store the validity bitmask and the actual data (not necessarily contigously). 
 *   - For fixed data it is just storing the underlying columnar data
 *   - For variable size strings we first store the offset and then the actual characters. Additionally, we also
 *   - store the duckdb style string alongside the raw characters that we copied over from the GPU
 */
struct result_table_column { 
    cudf::type_id column_type; // The type of the column
    result_table_location valid_mask_loc; // The location where the validity mask of the column is
    result_table_location data_loc; // The location where the data of the column actually is
    size_t valid_mask_bytes; // The number of bytes occupied by the validity mask
    size_t num_rows; // The number of rows in the column
    size_t column_data_bytes; // The number of bytes needed to store the column's data
    result_table_location duckdb_strings_loc; 
};

/**
 * @brief Structure containing both the host memory allocation and additional metadata for create results
 * 
 * Currently, both cudf and duckdb store and expect data to be in an arrow style format (i.e. contingous buffer)
 * but we use fixed sized blocks for our host memory allocation. Thus, we need to store the result table in host memory
 * in a way that they are spread across the allocated blocks but we can pass to DuckDB as if they were contigious and this
 * representation is used to effectivelly perform that mapping/abstraction.
 */
 struct result_table_allocation {
    fixed_size_host_memory_resource::multiple_blocks_allocation allocation;
    std::vector<result_table_column> columns;
    
    result_table_allocation(fixed_size_host_memory_resource::multiple_blocks_allocation alloc, std::vector<result_table_column> columns) 
        : allocation(std::move(alloc)), columns(std::move(columns)) {}
};

} // namespace memory
} // namespace sirius