/*
 * RasterDB cucascade::memory shim — pure C++ types replacing CUDA/RMM memory management.
 * GPU memory is managed by rasterdf (Vulkan + VMA).
 */
#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>
#include <utility>
#include <vector>
#include <stdexcept>

namespace cucascade {
namespace memory {

enum class Tier : int32_t {
    GPU,
    HOST,
    DISK,
    SIZE
};

class memory_space_id {
public:
    Tier tier;
    int32_t device_id;
    explicit memory_space_id(Tier t, int32_t d_id) : tier(t), device_id(d_id) {}
    auto operator<=>(const memory_space_id&) const noexcept = default;
    std::size_t uuid() const noexcept {
        std::size_t key = 0;
        std::memcpy(&key, this, sizeof(key));
        return key;
    }
};

// Stub device_memory_resource — no-op, all GPU memory via rasterdf VMA
class device_memory_resource {
public:
    virtual ~device_memory_resource() = default;
    virtual void* allocate(std::size_t bytes, std::size_t = 256) {
        return ::malloc(bytes);
    }
    virtual void deallocate(void* p, std::size_t, std::size_t = 256) {
        ::free(p);
    }
};

using DeviceMemoryResourceFactoryFn =
    std::function<std::unique_ptr<device_memory_resource>(int device_id, std::size_t capacity)>;

inline std::unique_ptr<device_memory_resource> make_default_gpu_memory_resource(int, std::size_t) {
    return std::make_unique<device_memory_resource>();
}
inline std::unique_ptr<device_memory_resource> make_default_host_memory_resource(int, std::size_t) {
    return std::make_unique<device_memory_resource>();
}
inline DeviceMemoryResourceFactoryFn make_default_allocator_for_tier(Tier) {
    return [](int, std::size_t) { return std::make_unique<device_memory_resource>(); };
}

struct column_metadata {
    int32_t type_id{0};
    int32_t type_size{0};
    int32_t scale{0};
    bool nullable{false};
};

} // namespace memory
} // namespace cucascade

namespace std {
template<> struct hash<cucascade::memory::memory_space_id> {
    size_t operator()(const cucascade::memory::memory_space_id& id) const noexcept {
        return id.uuid();
    }
};
} // namespace std
