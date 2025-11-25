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

#include "data/duckdb_data_representation.hpp"

namespace sirius {

duckdb_table_representation::duckdb_table_representation(sirius::unique_ptr<sirius::memory::result_table_allocation> result_table, std::vector<sirius::unique_ptr<duckdb::DataChunk>> output_vectors, sirius::memory::memory_space& memory_space)
    : idata_representation(memory_space), _result_table(std::move(result_table)) {

    // Create the output chunks to reference the data in the result table but using duckdb types
    for(int i = 0; i < output_vectors.size(); i++) { 
        sirius::unique_ptr<duckdb::DataChunk>& result_data_chunk = output_vectors[i];
        duckdb::unique_ptr<duckdb::DataChunk> duckdb_data_chunk = duckdb::make_uniq<duckdb::DataChunk>();

        duckdb_data_chunk->InitializeEmpty(result_data_chunk->GetTypes());
        duckdb_data_chunk->SetCardinality(result_data_chunk->size());
        for(int col = 0; col < result_data_chunk->ColumnCount(); col++) {
            duckdb_data_chunk->data[col].Reference(result_data_chunk->data[col]);
        }
        _output_chunks.push_back(std::move(duckdb_data_chunk));
    }
}

std::size_t duckdb_table_representation::get_size_in_bytes() const {
    return 0;
}

sirius::unique_ptr<idata_representation> duckdb_table_representation::convert_to_memory_space(sirius::memory::memory_space& target_memory_space, rmm::cuda_stream_view stream) {
    throw std::runtime_error("DuckDB table representation currently doesn't support being converted to another memory spaces");
}

} // namespace sirius