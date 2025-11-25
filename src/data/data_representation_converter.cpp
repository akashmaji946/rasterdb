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


#include "data/data_representation_converter.hpp"
#include "memory/result_table.hpp"
#include "memory/fixed_size_host_memory_resource.hpp"

#include "duckdb/common/types/string_type.hpp"
#include "duckdb/common/types.hpp"

#include <cuda_runtime.h>
#include <cudf/column/column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>

#include <cmath>
#include <unordered_map>

namespace sirius {

using sirius::memory::fixed_size_host_memory_resource;

constexpr size_t BYTE_TO_BITS = 8;

// Helper map to map CUDF types to DuckDB types. TODO(Devesh): Make sure that all of the mapping make sense because not all of them are 1:1 mappings
const std::unordered_map<cudf::type_id, duckdb::LogicalTypeId> CUDF_TO_DUCKDB_TYPES_MAP = {
    // Signed Integers
    {cudf::type_id::INT8,    duckdb::LogicalTypeId::TINYINT},
    {cudf::type_id::INT16,   duckdb::LogicalTypeId::SMALLINT},
    {cudf::type_id::INT32,   duckdb::LogicalTypeId::INTEGER},
    {cudf::type_id::INT64,   duckdb::LogicalTypeId::BIGINT},

    // Unsigned Integers
    {cudf::type_id::UINT8,   duckdb::LogicalTypeId::UTINYINT},
    {cudf::type_id::UINT16,  duckdb::LogicalTypeId::USMALLINT},
    {cudf::type_id::UINT32,  duckdb::LogicalTypeId::UINTEGER},
    {cudf::type_id::UINT64,  duckdb::LogicalTypeId::UBIGINT},

    // Floating Point
    {cudf::type_id::FLOAT32, duckdb::LogicalTypeId::FLOAT},
    {cudf::type_id::FLOAT64, duckdb::LogicalTypeId::DOUBLE},

    // Boolean & String
    {cudf::type_id::BOOL8,   duckdb::LogicalTypeId::BOOLEAN},
    {cudf::type_id::STRING,  duckdb::LogicalTypeId::VARCHAR},

    // Timestamps
    // cudf::TIMESTAMP_DAYS -> DuckDB DATE (days since epoch)
    {cudf::type_id::TIMESTAMP_DAYS,         duckdb::LogicalTypeId::DATE},
    {cudf::type_id::TIMESTAMP_SECONDS,      duckdb::LogicalTypeId::TIMESTAMP_SEC},
    {cudf::type_id::TIMESTAMP_MILLISECONDS, duckdb::LogicalTypeId::TIMESTAMP_MS},
    {cudf::type_id::TIMESTAMP_MICROSECONDS, duckdb::LogicalTypeId::TIMESTAMP}, // Default TIMESTAMP is us
    {cudf::type_id::TIMESTAMP_NANOSECONDS,  duckdb::LogicalTypeId::TIMESTAMP_NS},
};

inline duckdb::LogicalTypeId cudf_type_to_duckdb_type(cudf::type_id id) {
    auto it = CUDF_TO_DUCKDB_TYPES_MAP.find(id);
    if (it != CUDF_TO_DUCKDB_TYPES_MAP.end()) {
        return it->second;
    }

    return duckdb::LogicalTypeId::INVALID;
}

// Helper function used to determine the number of bytes needed to store a particular column
size_t determine_column_bytes_needed(cudf::column_view& column_view, rmm::cuda_stream_view& stream) { 
    size_t num_rows = column_view.size();

    // First determine the bytes needed to hold the null mask
    size_t null_mask_bytes = cudf::bitmask_allocation_size_bytes(num_rows); // By default aligns to 64 byte boundary

    // Now determine the bytes needed to hold the actual data
    size_t data_bytes = 0;
    cudf::data_type col_type = column_view.type();
    if(col_type.id() == cudf::type_id::STRING) { 
        // For string get the number bytes for the offset and chars column
        cudf::strings_column_view strings_view(column_view);
        size_t offset_bytes = num_rows * cudf::size_of(strings_view.offsets().type());
        size_t chars_bytes = strings_view.chars_size(stream);
        size_t duckdb_string_bytes = num_rows * sizeof(duckdb::string_t);
        data_bytes = offset_bytes + chars_bytes;
    } else {
        int32_t width = cudf::size_of(col_type);
        data_bytes = width * num_rows;
    }

    return null_mask_bytes + data_bytes;
}

// Ensures that the location is aligned to the specificed size within the block
inline void align_location(sirius::memory::result_table_location& location, size_t target_alignment, fixed_size_host_memory_resource::multiple_blocks_allocation& allocated_blocks) { 
    if(location.block_offset % target_alignment != 0) { 
        location.block_offset += target_alignment - (location.block_offset % target_alignment);
        
        // Deal with this shift overflowing into the next block
        if(location.block_offset >= allocated_blocks.block_size) { 
            location.allocation_index += 1;
            location.block_offset = 0;
            allocated_blocks.ensure_capacity(location.allocation_index + 1);
        }
    }
} 

// Helper function to copy data from GPU to CPU in a way that ensures that records are aligned to their size and no record
// is spread across pages
inline void copy_buffer_to_cpu(
    const char* gpu_data_buffer, size_t num_records, size_t record_size, 
    fixed_size_host_memory_resource::multiple_blocks_allocation& allocated_blocks,
    sirius::memory::result_table_location& write_location,
    rmm::cuda_stream_view& stream) { 
    
    size_t records_left = num_records;
    while(records_left > 0) {
        size_t curr_write_offset = write_location.block_offset;
        char* block_copy_ptr = (char*) allocated_blocks[write_location.allocation_index] + curr_write_offset;

        // Determine the number of records we can copy into this block while maintaing alignment
        size_t max_possible_copyable_record = (allocated_blocks.block_size - curr_write_offset)/record_size;
        size_t records_to_copy = std::min(records_left, max_possible_copyable_record); 
        size_t bytes_to_copy = records_to_copy * record_size;

        // Perform the copy
        cudaMemcpyAsync(block_copy_ptr, gpu_data_buffer, bytes_to_copy, cudaMemcpyDeviceToHost, stream.value());
        gpu_data_buffer += bytes_to_copy;
        write_location.block_offset = curr_write_offset + records_to_copy * record_size;
        records_left -= records_to_copy;

        // If we still have records to copy then move to the next block
        if(records_left > 0) { 
            write_location.allocation_index += 1;
            write_location.block_offset = 0;
            allocated_blocks.ensure_capacity(write_location.allocation_index + 1);
        }
    }
}

sirius::unique_ptr<sirius::duckdb_table_representation> data_representation_converter::convert_gpu_table_to_result_format(
    cudf::table_view src_table, 
    sirius::memory::memory_space& host_memory_space,
    rmm::cuda_stream_view stream
) {
    // Get the allocator from the host memory space
    auto host_memory_resource = host_memory_space.get_default_allocator_as<sirius::memory::fixed_size_host_memory_resource>();
    if(host_memory_resource == nullptr) { 
        throw std::runtime_error("Can't cast allocator of host memory resource to fixed sized allocator");
    }

    // Request an initial allocation from the allocator based on the column sizes 
    std::vector<cudf::column_view> table_columns(src_table.begin(), src_table.end());
    size_t num_columns = table_columns.size();
    size_t min_bytes_needed = 0;
    for (int i = 0; i < num_columns; i++) {
        min_bytes_needed += determine_column_bytes_needed(table_columns[i], stream);
    }
    fixed_size_host_memory_resource::multiple_blocks_allocation allocated_blocks = host_memory_resource->allocate_multiple_blocks(min_bytes_needed);

    // Now start copying the table data to CPU a column at a time using the stream
    duckdb::vector<duckdb::LogicalType> duckdb_col_types;

    size_t num_rows = table_columns[0].size();
    std::vector<sirius::memory::result_table_column> result_columns;
    sirius::memory::result_table_location allocation_write_location; 
    for (int i = 0; i < num_columns; i++) {
        // Initialize the result column
        cudf::column_view& cudf_column = table_columns[i];
        sirius::memory::result_table_column result_column;
        cudf::data_type col_type = cudf_column.type();
        result_column.column_type = col_type.id();

        // Copy over its null bitmask
        rmm::device_buffer mask_buffer;
        if (cudf_column.null_mask() != nullptr) {
            mask_buffer = cudf::copy_bitmask(cudf_column); // We perform a copy here because the bitmask may not be aligned by default
        } else { 
            mask_buffer = cudf::create_null_mask(cudf_column.size(), cudf::mask_state::ALL_VALID);
        }

        size_t mask_size = mask_buffer.size();
        result_column.valid_mask_loc = allocation_write_location;
        copy_buffer_to_cpu(static_cast<const char*>(mask_buffer.data()), mask_size, 1, allocated_blocks, allocation_write_location, stream);
        result_column.valid_mask_bytes = mask_size;

        // Copy over the actual column contents
        result_column.num_rows = num_rows;
        duckdb::LogicalTypeId duckdb_col_type = cudf_type_to_duckdb_type(result_column.column_type);
        duckdb_col_types.push_back(duckdb::LogicalType(duckdb_col_type));
        if(col_type.id() == cudf::type_id::STRING) { 
            // First copy the offsets
            cudf::strings_column_view strings_col(cudf_column);
            cudf::column_view strings_col_offsets = strings_col.offsets();
            size_t offsets_width = cudf::size_of(strings_col_offsets.type()); 
            const char* offsets_ptr = reinterpret_cast<const char*>(strings_col_offsets.head()) + offsets_width * strings_col_offsets.offset();
            
            align_location(allocation_write_location, offsets_width, allocated_blocks);
            result_column.data_loc = allocation_write_location;
            copy_buffer_to_cpu(offsets_ptr, num_rows, offsets_width, allocated_blocks, allocation_write_location, stream);

            // Then copy the chars
            size_t chars_bytes = strings_col.chars_size(stream);
            const char* actual_chars_ptr = reinterpret_cast<const char*>(strings_col.chars_begin(stream));
            copy_buffer_to_cpu(actual_chars_ptr, chars_bytes, 1, allocated_blocks, allocation_write_location, stream);
            
            result_column.column_data_bytes = chars_bytes + offsets_width * num_rows;

            // TODO(Devesh): Handle creating of the German style strings needed by DuckDB
        } else { 
            size_t record_width = cudf::size_of(col_type); 
            const char* records_ptr = reinterpret_cast<const char*>(cudf_column.head()) + record_width * cudf_column.offset();

            align_location(allocation_write_location, record_width, allocated_blocks);
            result_column.data_loc = allocation_write_location;
            copy_buffer_to_cpu(records_ptr, num_rows, record_width, allocated_blocks, allocation_write_location, stream);

            result_column.column_data_bytes = record_width * num_rows;
        }
        result_columns.push_back(result_column);
    }

    stream.synchronize();

    // Now also create Data Chunks in a format that we can pass to DuckDB - For now this assumes that we only have integer columns
    std::vector<sirius::memory::result_table_location> column_valid_mask_location;
    std::vector<sirius::memory::result_table_location> column_data_location;
    for(int i = 0; i < num_columns; i++) { 
        column_valid_mask_location.push_back(result_columns[i].valid_mask_loc);
        column_data_location.push_back(result_columns[i].data_loc);
    }
    
    size_t block_size = allocated_blocks.block_size;
    size_t records_left = num_rows;
    std::vector<sirius::unique_ptr<duckdb::DataChunk>> result_chunks;
    while(records_left > 0) { 
        // Determine the number of records we can include without crossing a block boundary
        size_t records_in_chunk = std::min(records_left, static_cast<size_t>(STANDARD_VECTOR_SIZE));

        for(int i = 0; i < num_columns; i++) { 
            cudf::type_id col_type = result_columns[i].column_type;
            // Determine maximum records from null mask
            size_t max_mask_records = (block_size - column_valid_mask_location[i].block_offset) * BYTE_TO_BITS;
            records_in_chunk = std::min(records_in_chunk, max_mask_records);

            if(col_type == cudf::type_id::STRING) { 
                // TODO(Devesh): Extend this implementation to account for string columns
            } else { 
                // Determine maximum amount of data points
                size_t record_width = cudf::size_of(cudf::data_type(col_type));
                size_t curr_record_offset = column_data_location[i].block_offset;
                size_t max_records_from_col = (block_size - curr_record_offset)/record_width; 
                max_records_from_col = max_records_from_col - (max_records_from_col % BYTE_TO_BITS); // We want to ensure that the number of records is a multiple of 8 so that we don't have to splice the null mask 

                records_in_chunk = std::min(records_in_chunk, max_records_from_col); 
            }
        }

        // Create the Data Chunk
        sirius::unique_ptr<duckdb::DataChunk> chunk = sirius::make_unique<duckdb::DataChunk>();
		chunk->InitializeEmpty(duckdb_col_types);
        for(int i = 0; i < num_columns; i++) { 
            cudf::type_id col_type = result_columns[i].column_type;
            if(col_type == cudf::type_id::STRING) { 
                // TODO(Devesh): Extend this implementation to account for string columns
            } else { 
                // Create the vector
                size_t data_block_to_load = column_data_location[i].allocation_index;
                uint8_t* col_records_ptr = reinterpret_cast<uint8_t*>(allocated_blocks[data_block_to_load]) + column_data_location[i].block_offset;
                duckdb::Vector col_vector(duckdb_col_types[i], col_records_ptr);
                
                // Set the null mask
                size_t mask_block_to_load = column_valid_mask_location[i].allocation_index;
                uint8_t* mask_ptr = reinterpret_cast<uint8_t*>(allocated_blocks[mask_block_to_load]) + column_valid_mask_location[i].block_offset;
                duckdb::ValidityMask validity_mask(reinterpret_cast<duckdb::validity_t*>(mask_ptr), records_in_chunk);
                duckdb::FlatVector::SetValidity(col_vector, validity_mask);

                // Add the vector to the chunk
                chunk->data[i].Reference(col_vector);
            }
        }
        chunk->SetCardinality(records_in_chunk);
        records_left -= records_in_chunk;
        result_chunks.push_back(std::move(chunk));

        // Increment the read offsets
        for(int i = 0; i < num_columns; i++) { 
            cudf::type_id col_type = result_columns[i].column_type;
            if(col_type == cudf::type_id::STRING) { 

            } else { 
                size_t record_width = cudf::size_of(cudf::data_type(col_type));

                // Increment the data record location
                column_data_location[i].block_offset += record_width * records_in_chunk;
                if(column_data_location[i].block_offset + record_width >= block_size) { // See if we the remaining records are in the next block
                    column_data_location[i].allocation_index += 1;
                    column_data_location[i].block_offset = 0;
                }

                // Increment the mask ptr
                size_t mask_increment = std::ceil(records_in_chunk/BYTE_TO_BITS);
                column_valid_mask_location[i].block_offset += mask_increment;
                column_valid_mask_location[i].allocation_index += column_valid_mask_location[i].block_offset/block_size;
                column_valid_mask_location[i].block_offset = column_valid_mask_location[i].block_offset % block_size;
            }
        }
    }

    // Create the result
    sirius::unique_ptr<sirius::memory::result_table_allocation> result_table_allocation = sirius::make_unique<sirius::memory::result_table_allocation>(
        std::move(allocated_blocks), std::move(result_columns)
    );
    return sirius::make_unique<sirius::duckdb_table_representation>(std::move(result_table_allocation), std::move(result_chunks), host_memory_space);
}

}