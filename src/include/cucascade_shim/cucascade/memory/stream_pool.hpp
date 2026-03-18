/*
 * RasterDB cucascade::memory::stream_pool shim.
 * No CUDA streams — Vulkan uses command buffers via rasterdf.
 */
#pragma once
#include <cstddef>
#include <cstdint>

namespace cucascade {
namespace memory {

class exclusive_stream_pool {
public:
    enum class stream_acquire_policy { ROUND_ROBIN, GROW };

    explicit exclusive_stream_pool(std::size_t = 1, stream_acquire_policy = stream_acquire_policy::GROW) {}
    ~exclusive_stream_pool() = default;

    // Stub — no real streams
    void* acquire() { return nullptr; }
    void release(void*) {}
    std::size_t size() const { return 0; }
};

} // namespace memory
} // namespace cucascade
