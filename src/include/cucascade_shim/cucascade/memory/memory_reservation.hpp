/*
 * RasterDB cucascade::memory::reservation shim.
 */
#pragma once
#include "memory_space.hpp"
#include <cstddef>
#include <memory>

namespace cucascade {
namespace memory {

class reservation {
public:
    reservation() = default;
    reservation(memory_space* space, std::size_t bytes)
        : _space(space), _bytes(bytes) {}
    reservation(reservation&& o) noexcept : _space(o._space), _bytes(o._bytes) {
        o._space = nullptr; o._bytes = 0;
    }
    reservation& operator=(reservation&& o) noexcept {
        _space = o._space; _bytes = o._bytes;
        o._space = nullptr; o._bytes = 0;
        return *this;
    }
    ~reservation() = default;

    std::size_t size() const { return _bytes; }
    memory_space* space() const { return _space; }
    memory_space* get_memory_space() const { return _space; }
    bool valid() const { return _space != nullptr && _bytes > 0; }
    explicit operator bool() const { return valid(); }

    void* allocate(std::size_t bytes) {
        return _space ? _space->allocate(bytes) : nullptr;
    }
    void deallocate(void* p, std::size_t bytes) {
        if (_space) _space->deallocate(p, bytes);
    }

private:
    memory_space* _space{nullptr};
    std::size_t _bytes{0};
};

} // namespace memory
} // namespace cucascade
