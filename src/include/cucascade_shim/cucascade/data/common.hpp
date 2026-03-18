/*
 * RasterDB cucascade::data shim — pure C++ types.
 * GPU data management handled by rasterdf (Vulkan + VMA).
 */
#pragma once

#include "../memory/common.hpp"
#include "../memory/memory_space.hpp"

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>
#include <mutex>
#include <functional>

namespace cucascade {

enum class batch_state {
    idle,
    task_created,
    processing,
    in_transit
};

class data_batch;

class data_batch_processing_handle {
public:
    data_batch_processing_handle() = default;
    data_batch_processing_handle(data_batch_processing_handle&&) noexcept = default;
    data_batch_processing_handle& operator=(data_batch_processing_handle&&) noexcept = default;
    ~data_batch_processing_handle() = default;

    void release() {}
    bool valid() const { return false; }
    explicit operator bool() const { return valid(); }
};

// Forward declarations
class idata_representation;
class gpu_table_representation;
class host_data_representation;

class idata_representation {
public:
    virtual ~idata_representation() = default;
    virtual std::size_t num_rows() const = 0;
    virtual std::size_t num_columns() const = 0;
    virtual std::size_t size_bytes() const = 0;
    virtual memory::memory_space_id location() const = 0;
};

} // namespace cucascade
