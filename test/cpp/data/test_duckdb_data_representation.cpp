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

#include "catch.hpp"
#include <memory>
#include <vector>

#include "duckdb/common/typedefs.hpp"

#include "data/cpu_data_representation.hpp"
#include "data/gpu_data_representation.hpp"
#include "data/duckdb_data_representation.hpp"
#include "data/common.hpp"
#include "memory/null_device_memory_resource.hpp"
#include "memory/result_table.hpp"
#include "memory/fixed_size_host_memory_resource.hpp"

using namespace sirius;

// Mock memory_space for testing - provides a simple memory_space without real allocators
class mock_memory_space : public memory::memory_space {
public:
    mock_memory_space(memory::Tier tier, std::vector<std::unique_ptr<rmm::mr::device_memory_resource>> allocators, size_t device_id = 0)
        : memory::memory_space(tier, device_id, 1024 * 1024 * 1024, allocators.size() > 0 ? std::move(allocators) : create_null_allocators()) {
        
    }
    
private:
    static std::vector<std::unique_ptr<rmm::mr::device_memory_resource>> create_null_allocators() {
        std::vector<std::unique_ptr<rmm::mr::device_memory_resource>> allocators;
        allocators.push_back(std::make_unique<memory::null_device_memory_resource>());
        return allocators;
    }
};

// Helper function creates a small fixed sized buffer pool 
constexpr std::size_t DEFAULT_BLOCK_SIZE = 2 << 10;
std::unique_ptr<rmm::mr::device_memory_resource> create_host_memory_resource(int num_blocks = 1) { 
    constexpr std::size_t blocks_in_pool = 1;
    return std::make_unique<memory::fixed_size_host_memory_resource>(DEFAULT_BLOCK_SIZE, blocks_in_pool, num_blocks);
}

// Helper method to create test tables
std::unique_ptr<cudf::column> create_fixed_size_column(cudf::type_id col_type_id, int num_rows, bool has_null_mask) { 
    return cudf::make_numeric_column(
        cudf::data_type{col_type_id},
        num_rows,
        has_null_mask ? cudf::mask_state::UNINITIALIZED : cudf::mask_state::UNALLOCATED 
    ); // Note that cudf::mask_state::UNINITIALIZED populates with random data so switch to cudf::mask_state::ALL_VALID/cudf::mask_state::ALL_NULL for determinstic behaviour
}

std::unique_ptr<cudf::column> create_string_column(int num_rows, bool has_null_mask) { 
    cudf::string_scalar row_value("hello_world", true);
    auto col = cudf::make_column_from_scalar(row_value, num_rows);
    if (!has_null_mask) {
        col->set_null_mask(rmm::device_buffer{}, 0);
    }
    return col;
}

sirius::unique_ptr<gpu_table_representation> create_gpu_representation(sirius::memory::memory_space& memory_space, std::vector<cudf::type_id> col_types, std::vector<bool> is_nullable_col, int num_rows) { 
    std::vector<std::unique_ptr<cudf::column>> columns;
    for(int i = 0; i < col_types.size(); i++) { 
        std::unique_ptr<cudf::column> curr_col; 
        if(col_types[i] == cudf::type_id::STRING) { 
            curr_col = create_string_column(num_rows, is_nullable_col[i]);
        } else { 
            curr_col = create_fixed_size_column(col_types[i], num_rows, is_nullable_col[i]);
        }

        columns.push_back(std::move(curr_col));
    }

    cudf::table underlying_table = cudf::table(std::move(columns));
    return sirius::make_unique<gpu_table_representation>(std::move(underlying_table), memory_space);
}

// Helper function to verify that the DataChunks repreents the specified cudf columns
void verify_validity_mask(cudf::column_view cudf_col_view, size_t col, size_t num_records, std::vector<duckdb::unique_ptr<duckdb::DataChunk>>& output_chunks) { 
    // Copy the cudf mask to host if there is one
    std::vector<uint8_t> h_cudf_mask;
    bool cudf_has_mask = cudf_col_view.null_mask() != nullptr;
    if (cudf_has_mask) {
        // Calculate size in 32-bit words needed for bitmask
        rmm::device_buffer mask_buffer = cudf::copy_bitmask(cudf_col_view);
        h_cudf_mask.resize(mask_buffer.size());
        
        // Copy mask from GPU to CPU
        cudaMemcpy(
            h_cudf_mask.data(), 
            mask_buffer.data(), 
            mask_buffer.size(), 
            cudaMemcpyDeviceToHost
        );
    }

    // Iterate through chunk by chunk
    size_t global_record_offset = 0;
    for (int chunk_idx = 0; chunk_idx < output_chunks.size(); chunk_idx++) {
        // Get the specific vector for this column
        duckdb::unique_ptr<duckdb::DataChunk>& result_chunk = output_chunks[chunk_idx];
        size_t chunk_size = static_cast<size_t>(result_chunk->size());
        auto& column_vector = result_chunk->data[col];

        for (size_t chunk_offset = 0; chunk_offset < chunk_size; chunk_offset++) {
            size_t curr_record_offset = global_record_offset + chunk_offset;

            // Get the cudf validity value for this record
            bool cudf_is_valid = true;
            if(cudf_has_mask) { 
                size_t element_index = curr_record_offset / 8;
                size_t bit_index = curr_record_offset % 8;
                cudf_is_valid = static_cast<bool>((h_cudf_mask[element_index] >> bit_index) & 1);
            }

            // Compare this againse the duckdb value
            bool duckdb_is_valid = !column_vector.GetValue(chunk_offset).IsNull();
            if(cudf_is_valid != duckdb_is_valid) { 
                std::cout << "Null Mismatch Detail - Chunk: " << chunk_idx << ", Chunk Offset: " << chunk_offset << ", Global Offset: " << curr_record_offset << ", Cudf Value - " << cudf_is_valid << ", Duckdb Value - " << duckdb_is_valid << std::endl;
            }
            REQUIRE(cudf_is_valid == duckdb_is_valid);
        }

        global_record_offset += chunk_size;
    }
}

void verify_fixed_size_representation(cudf::column_view cudf_col_view, size_t col, size_t num_records, std::vector<duckdb::unique_ptr<duckdb::DataChunk>>& output_chunks) { 
    // Copy the records into a flat buffer on the CPU
    size_t record_width = cudf::size_of(cudf_col_view.type());
    const char* d_records_ptr = reinterpret_cast<const char*>(cudf_col_view.head()) + record_width * cudf_col_view.offset();
    size_t buffer_size_bytes = num_records * record_width;
    std::vector<duckdb::data_t> h_records;
    h_records.resize(buffer_size_bytes/sizeof(duckdb::data_t));

    cudaMemcpy(
        h_records.data(), 
        d_records_ptr, 
        buffer_size_bytes, 
        cudaMemcpyDeviceToHost
    );

    // Compare the data with the column's data
    size_t global_record_offset = 0;
    for (int chunk_idx = 0; chunk_idx < output_chunks.size(); chunk_idx++) {
        // Get the specific vector for this column
        duckdb::unique_ptr<duckdb::DataChunk>& result_chunk = output_chunks[chunk_idx];
        size_t chunk_size = static_cast<size_t>(result_chunk->size());
        auto& column_vector = result_chunk->data[col];

        for (size_t chunk_offset = 0; chunk_offset < chunk_size; chunk_offset++) {
            // Get the ptrs to the data in cudf and duckdb
            size_t curr_record_offset = global_record_offset + chunk_offset;
            duckdb::data_t* cudf_record_ptr = h_records.data() + curr_record_offset * record_width;
            duckdb::data_t* duckdb_record_ptr = column_vector.GetData() + chunk_offset * record_width;

            // Compare the duckdb data with the cudf data based on the record width
            if(record_width == 4) { 
                REQUIRE(reinterpret_cast<uint32_t*>(cudf_record_ptr)[0] == reinterpret_cast<uint32_t*>(duckdb_record_ptr)[0]);
            } else { // Fallback to character at a time comparsion
                size_t num_vals_to_check = record_width/sizeof(duckdb::data_t);
                for(size_t i = 0; i < num_vals_to_check; i++) { 
                    REQUIRE(cudf_record_ptr[i] == duckdb_record_ptr[i]);
                }
            }
        }
    }
}

void verify_representation_conversion(sirius::unique_ptr<duckdb_table_representation> duckdb_representation, sirius::unique_ptr<gpu_table_representation> gpu_representation) { 
    const cudf::table& gpu_table = gpu_representation->get_table();
    std::vector<duckdb::unique_ptr<duckdb::DataChunk>>& output_chunks = duckdb_representation->get_output_chunks();

    // First verify that the row count is correct
    size_t expected_columns = gpu_table.num_columns();
    size_t expected_rows = gpu_table.num_rows();
    size_t data_chunk_records = 0;
    for (const auto& data_chunk : output_chunks) {
        REQUIRE(data_chunk->ColumnCount() == expected_columns);
        data_chunk_records += data_chunk->size();
    }
    REQUIRE(data_chunk_records == expected_rows);

    // Now verify that each column's count is correct
    for(size_t i = 0; i < expected_columns; i++) { 
        // First check that it has the expected bitmask 
        cudf::column_view cudf_col_view = gpu_table.get_column(i).view();
        verify_validity_mask(cudf_col_view, i, expected_rows, output_chunks);

        // Then verify the actual data
        if(cudf_col_view.type().id() == cudf::type_id::STRING) { 

        } else { 
            verify_fixed_size_representation(cudf_col_view, i, expected_rows, output_chunks);
        }
    }
}

TEST_CASE("result_convertor_single_non_nullable_int_conversion", "[duckdb_data_representation][non_null_int_conversion]") {
    // Create the gpu and host memory spaces
    mock_memory_space gpu_memory_space(memory::Tier::GPU, {});
    std::vector<std::unique_ptr<rmm::mr::device_memory_resource>> host_allocators;
    host_allocators.push_back(create_host_memory_resource());
    mock_memory_space host_memory_space(memory::Tier::GPU, std::move(host_allocators));

    // Create the test gpu representation
    int num_test_rows = 400;
    std::vector<cudf::type_id> col_types = {cudf::type_id::INT32};
    std::vector<bool> is_null_column = {false};
    sirius::unique_ptr<gpu_table_representation> gpu_table_representation = create_gpu_representation(gpu_memory_space, col_types, is_null_column, num_test_rows);

    // Perform the conversion and verify
    sirius::unique_ptr<duckdb_table_representation> duckdb_representation = gpu_table_representation->convert_to_result_format(host_memory_space);
    verify_representation_conversion(std::move(duckdb_representation), std::move(gpu_table_representation));
}


TEST_CASE("result_convertor_single_nullable_int_conversion", "[duckdb_data_representation][nullable_int_conversion]") {
    // Create the gpu and host memory spaces
    mock_memory_space gpu_memory_space(memory::Tier::GPU, {});
    std::vector<std::unique_ptr<rmm::mr::device_memory_resource>> host_allocators;
    host_allocators.push_back(create_host_memory_resource());
    mock_memory_space host_memory_space(memory::Tier::GPU, std::move(host_allocators));

    // Create the test gpu representation
    int num_test_rows = 400;
    std::vector<cudf::type_id> col_types = {cudf::type_id::INT32};
    std::vector<bool> is_null_column = {true};
    sirius::unique_ptr<gpu_table_representation> gpu_table_representation = create_gpu_representation(gpu_memory_space, col_types, is_null_column, num_test_rows);

    // Perform the conversion and verify
    sirius::unique_ptr<duckdb_table_representation> duckdb_representation = gpu_table_representation->convert_to_result_format(host_memory_space);
    verify_representation_conversion(std::move(duckdb_representation), std::move(gpu_table_representation));
}

TEST_CASE("result_convertor_single_nullable_int_multiple_block_conversion", "[duckdb_data_representation][nullable_int_conversion]") {
    // Create the gpu and host memory spaces
    int num_test_rows = 2500;
    int total_bytes_needed = num_test_rows * sizeof(int) + std::ceil((1.0 * num_test_rows)/8);
    int blocks_needed = std::ceil((1.0 * total_bytes_needed)/DEFAULT_BLOCK_SIZE);

    mock_memory_space gpu_memory_space(memory::Tier::GPU, {});
    std::vector<std::unique_ptr<rmm::mr::device_memory_resource>> host_allocators;
    host_allocators.push_back(create_host_memory_resource(blocks_needed));
    mock_memory_space host_memory_space(memory::Tier::GPU, std::move(host_allocators));

    // Create the test gpu representation
    std::vector<cudf::type_id> col_types = {cudf::type_id::INT32};
    std::vector<bool> is_null_column = {true};
    sirius::unique_ptr<gpu_table_representation> gpu_table_representation = create_gpu_representation(gpu_memory_space, col_types, is_null_column, num_test_rows);

    // Perform the conversion and verify
    sirius::unique_ptr<duckdb_table_representation> duckdb_representation = gpu_table_representation->convert_to_result_format(host_memory_space);
    verify_representation_conversion(std::move(duckdb_representation), std::move(gpu_table_representation));
}