/*
 * RasterDB cucascade::memory::small_pinned_host_memory_resource shim.
 */
#pragma once
#include "common.hpp"
#include <cstddef>

namespace cucascade {
namespace memory {

class small_pinned_host_memory_resource : public device_memory_resource {
public:
    static constexpr std::size_t MAX_SLAB_SIZE = 256 * 1024;
    small_pinned_host_memory_resource() = default;
    ~small_pinned_host_memory_resource() override = default;
};

} // namespace memory
} // namespace cucascade
