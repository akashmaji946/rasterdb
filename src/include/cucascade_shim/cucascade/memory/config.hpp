/*
 * RasterDB cucascade::memory config shim.
 */
#pragma once
#include "common.hpp"
#include <string>
#include <vector>

namespace cucascade {
namespace memory {

struct gpu_memory_space_config {
    int32_t device_id{0};
    std::size_t capacity{0};
    DeviceMemoryResourceFactoryFn allocator_factory;
};

struct host_memory_space_config {
    int32_t device_id{0};
    std::size_t capacity{0};
    DeviceMemoryResourceFactoryFn allocator_factory;
};

struct disk_memory_space_config {
    int32_t device_id{0};
    std::size_t capacity{0};
    std::string path;
};

struct memory_space_config {
    std::vector<gpu_memory_space_config> gpu_configs;
    std::vector<host_memory_space_config> host_configs;
    std::vector<disk_memory_space_config> disk_configs;
};

} // namespace memory
} // namespace cucascade
