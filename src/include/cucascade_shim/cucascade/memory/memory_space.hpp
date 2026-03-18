/*
 * RasterDB cucascade::memory::memory_space shim.
 * GPU memory managed by rasterdf (Vulkan + VMA). This is a stub.
 */
#pragma once

#include "common.hpp"
#include <string>
#include <mutex>
#include <atomic>

namespace cucascade {
namespace memory {

class memory_space {
public:
    memory_space() = default;
    memory_space(memory_space_id id, std::size_t capacity)
        : _id(id), _capacity(capacity), _used(0) {}
    virtual ~memory_space() = default;

    memory_space_id id() const { return _id; }
    std::size_t capacity() const { return _capacity; }
    std::size_t used() const { return _used.load(); }
    std::size_t available() const {
        auto u = _used.load();
        return u < _capacity ? _capacity - u : 0;
    }

    device_memory_resource* resource() { return &_resource; }

    void* allocate(std::size_t bytes) {
        auto* p = _resource.allocate(bytes);
        _used += bytes;
        return p;
    }
    void deallocate(void* p, std::size_t bytes) {
        _resource.deallocate(p, bytes);
        auto cur = _used.load();
        _used = (cur >= bytes) ? cur - bytes : 0;
    }

private:
    memory_space_id _id{Tier::HOST, 0};
    std::size_t _capacity{0};
    std::atomic<std::size_t> _used{0};
    device_memory_resource _resource;
};

} // namespace memory
} // namespace cucascade
