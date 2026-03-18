/*
 * RasterDB cucascade::memory::fixed_size_host_memory_resource shim.
 */
#pragma once
#include "common.hpp"
#include <cstddef>
#include <cstdlib>
#include <vector>
#include <mutex>

namespace cucascade {
namespace memory {

class fixed_size_host_memory_resource : public device_memory_resource {
public:
    struct multiple_blocks_allocation {
        void* data{nullptr};
        std::size_t size{0};
        std::size_t num_blocks{0};
        std::size_t _block_size{0};

        void* block(std::size_t idx) const {
            if (!data || idx >= num_blocks) return nullptr;
            return static_cast<char*>(data) + idx * _block_size;
        }
        void* at(std::size_t idx) const { return block(idx); }
        std::size_t block_size() const { return _block_size; }
        std::vector<void*> get_blocks() const {
            std::vector<void*> blocks;
            for (std::size_t i = 0; i < num_blocks; i++) blocks.push_back(block(i));
            return blocks;
        }
        explicit operator bool() const { return data != nullptr; }
    };

    fixed_size_host_memory_resource(std::size_t block_size = 4096, std::size_t max_blocks = 1024)
        : _block_size(block_size), _max_blocks(max_blocks) {}
    ~fixed_size_host_memory_resource() override = default;

    multiple_blocks_allocation allocate_multiple(std::size_t n) {
        multiple_blocks_allocation alloc;
        alloc.size = n * _block_size;
        alloc.num_blocks = n;
        alloc.data = ::malloc(alloc.size);
        return alloc;
    }
    void deallocate_multiple(multiple_blocks_allocation& alloc) {
        ::free(alloc.data);
        alloc.data = nullptr;
        alloc.size = 0;
        alloc.num_blocks = 0;
    }

    std::size_t block_size() const { return _block_size; }

private:
    std::size_t _block_size;
    std::size_t _max_blocks;
};

} // namespace memory
} // namespace cucascade
