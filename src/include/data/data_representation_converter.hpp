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

#include "data/gpu_data_representation.hpp"
#include "data/duckdb_data_representation.hpp"
#include "memory/memory_space.hpp"

#include "cudf/cudf_utils.hpp"

namespace sirius {

/**
 * @brief Utility class for converting between data representations
 */
class data_representation_converter {
public: 

    static sirius::unique_ptr<sirius::duckdb_table_representation> convert_gpu_table_to_result_format(
        cudf::table_view src_table,
        sirius::memory::memory_space& host_memory_space,
        rmm::cuda_stream_view stream
    );
};

}