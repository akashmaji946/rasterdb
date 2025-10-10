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
#include <variant>
#include <memory>
#include <cudf/table/table.hpp>
#include "data/cudf_table_converter.hpp"
#include "helper/helper.hpp"

namespace sirius {

// Enum to indicate where the data is currently residing
enum class Tier {
    CPU,
    GPU
};

class DataBatch {
public:
    // Define the variant type to hold either cudf::table or spilling::allocation
    using DataVariant = std::variant<sirius::unique_ptr<cudf::table>, sirius::unique_ptr<table_allocation>>;

    // Constructor to initialize with cudf::table
    DataBatch(sirius::unique_ptr<cudf::table> gpu_data) 
        : data_(std::move(gpu_data)), location_(Tier::GPU) {}

    // Constructor to initialize with spilling::allocation
    DataBatch(sirius::unique_ptr<table_allocation> cpu_data) 
        : data_(std::move(cpu_data)), location_(Tier::CPU) {}

    // Function to convert data to GPU
    void toGPU() {
    }

    // Function to convert data to CPU
    void toCPU() {
    }

    // Access the underlying data
    const DataVariant& getData() const {
        return data_;
    }

private:
    DataVariant data_;
    Tier location_;

    // Implement these conversion functions according to your specific logic
    sirius::unique_ptr<cudf::table> ConvertToGPU(sirius::unique_ptr<table_allocation> cpu_data) {
        // Conversion logic here
        // ...
        return nullptr; // Replace with actual conversion result
    }

    sirius::unique_ptr<table_allocation> ConvertToCPU(sirius::unique_ptr<cudf::table> gpu_data) {
        // Conversion logic here
        // ...
        return nullptr; // Replace with actual conversion result
    }
};

} // namespace sirius